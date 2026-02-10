#!/usr/bin/env python3
"""
Garden Classification API

FastAPI application providing REST endpoints for:
1. Single address lookup (eircode/GPS) -> front/back garden pins
2. Batch lookup (GPS + radius) -> all pins in area
3. Point classification (GPS) -> "front_garden" or "back_garden"
4. Precomputation management for large areas

Optimizations:
- Async endpoints with thread pool for CPU-bound work
- Per-coordinate caching (not chunk-based)
- Concurrent request handling
- Small processing radius for single-point queries

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

Endpoints:
    GET  /health                     - Health check
    POST /api/garden-pins            - Get pins for single address
    POST /api/garden-pins/batch      - Get all pins in radius
    POST /api/classify               - Classify a GPS coordinate
    POST /api/precompute             - Precompute a large area
    GET  /api/cache/stats            - Get cache statistics
    DELETE /api/cache                - Clear caches
"""

import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
from enum import Enum
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.tiles import TileSource, fetch_area_image
from src.precompute import PrecomputeManager
from src.fast_classifier import FastGardenClassifier, get_fast_classifier
from src.osm import fetch_buildings

# =============================================================================
# Thread Pool for CPU-bound work
# =============================================================================

# Thread pool for running CPU-intensive classification work
_executor = ThreadPoolExecutor(max_workers=8)


async def run_in_executor(func, *args, **kwargs):
    """Run a blocking function in the thread pool."""
    loop = asyncio.get_event_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(_executor, func, *args)


# =============================================================================
# API Models
# =============================================================================

class TileSourceEnum(str, Enum):
    """Tile source options."""
    google = "google"
    manna = "manna"
    auto = "auto"


class GardenPinRequest(BaseModel):
    """Request for garden pins at a single location."""
    lat: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    lon: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    eircode: Optional[str] = Field(None, description="Irish Eircode (alternative to lat/lon)")
    tile_source: TileSourceEnum = Field(TileSourceEnum.auto, description="Tile source to use")
    skip_cache: bool = Field(False, description="Skip cache and recompute")


class GardenPinResponse(BaseModel):
    """Response with front and back garden pins."""
    front: Optional[Dict[str, Any]] = Field(None, description="Best front garden delivery pin")
    back: Optional[Dict[str, Any]] = Field(None, description="Best back garden delivery pin")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BatchPinRequest(BaseModel):
    """Request for all garden pins within a radius."""
    lat: float = Field(..., ge=-90, le=90, description="Center latitude")
    lon: float = Field(..., ge=-180, le=180, description="Center longitude")
    radius_m: float = Field(500, ge=10, le=5000, description="Search radius in meters")
    garden_type: Optional[str] = Field(None, description="Filter by 'front' or 'back'")
    min_score: float = Field(20.0, ge=0, le=100, description="Minimum score threshold")
    tile_source: TileSourceEnum = Field(TileSourceEnum.auto, description="Tile source to use")
    generate_map: bool = Field(False, description="Generate visualization map")


class BatchPinResponse(BaseModel):
    """Response with all pins in the area."""
    pins: List[Dict[str, Any]] = Field(default_factory=list, description="List of delivery pins")
    count: int = Field(0, description="Number of pins returned")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    map_url: Optional[str] = Field(None, description="URL to visualization map (if generated)")


class ClassifyRequest(BaseModel):
    """Request to classify a GPS coordinate."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    tile_source: TileSourceEnum = Field(TileSourceEnum.auto, description="Tile source to use")
    skip_cache: bool = Field(False, description="Skip cache and recompute")


class ClassifyResponse(BaseModel):
    """Response with classification result."""
    lat: float
    lon: float
    classification: str = Field(..., description="'front_garden', 'back_garden', or other surface type")
    surface_type: Optional[str] = Field(None, description="Type of surface")
    score: Optional[float] = Field(None, description="Delivery suitability score (0-100)")
    distance_to_building_m: Optional[float] = Field(None, description="Distance to nearest building")


class PrecomputeRequest(BaseModel):
    """Request to precompute a large area."""
    lat: float = Field(..., ge=-90, le=90, description="Center latitude")
    lon: float = Field(..., ge=-180, le=180, description="Center longitude")
    radius_m: float = Field(1000, ge=100, le=50000, description="Radius in meters")
    tile_source: TileSourceEnum = Field(TileSourceEnum.auto, description="Tile source to use")
    skip_cache: bool = Field(False, description="Force recompute even if cached")


class PrecomputeResponse(BaseModel):
    """Response from precomputation request."""
    status: str = Field(..., description="'started', 'completed', or 'error'")
    message: str = Field(..., description="Status message")
    summary: Optional[Dict[str, Any]] = Field(None, description="Computation summary")


class CacheStatsResponse(BaseModel):
    """Response with cache statistics."""
    fast_cache_entries: int
    precompute_areas: int
    total_pins: int


# =============================================================================
# Eircode Geocoding (Async)
# =============================================================================

# Irish Eircode routing key to approximate coordinates
# These are center points for each routing area
EIRCODE_ROUTING_COORDS = {
    # Dublin
    "D01": (53.3498, -6.2603), "D02": (53.3382, -6.2591), "D03": (53.3601, -6.2294),
    "D04": (53.3244, -6.2297), "D05": (53.3697, -6.2131), "D06": (53.3209, -6.2654),
    "D6W": (53.3102, -6.2878), "D07": (53.3567, -6.2817), "D08": (53.3387, -6.2928),
    "D09": (53.3782, -6.2334), "D10": (53.3519, -6.3249), "D11": (53.3867, -6.2842),
    "D12": (53.3189, -6.3172), "D13": (53.3923, -6.1968), "D14": (53.2978, -6.2478),
    "D15": (53.3879, -6.3892), "D16": (53.2858, -6.2108), "D17": (53.4012, -6.1521),
    "D18": (53.2689, -6.1789), "D20": (53.3492, -6.3678), "D22": (53.3189, -6.3892),
    "D24": (53.2889, -6.3892),
    # Add more routing keys as needed
}


async def geocode_eircode(eircode: str) -> Optional[Dict[str, float]]:
    """
    Geocode an Irish Eircode to lat/lon coordinates.
    
    Strategy:
    1. Try Google Geocoding API
    2. Try Nominatim with full eircode
    3. Use routing key lookup table for approximate location
    
    Args:
        eircode: Irish Eircode (e.g., "D15 YXN8" or "D15YXN8")
        
    Returns:
        Dict with 'lat' and 'lon' or None if not found
    """
    import config
    
    # Clean up eircode
    eircode_clean = eircode.strip().upper().replace(" ", "")
    
    if len(eircode_clean) != 7:
        print(f"Invalid eircode format: {eircode} (should be 7 characters)")
        return None
    
    eircode_formatted = f"{eircode_clean[:3]} {eircode_clean[3:]}"
    routing_key = eircode_clean[:3]
    
    async with httpx.AsyncClient(timeout=15) as client:
        # Strategy 1: Google Geocoding API
        google_api_key = getattr(config, 'GOOGLE_TILES_API_KEY', None)
        if google_api_key:
            try:
                url = "https://maps.googleapis.com/maps/api/geocode/json"
                params = {
                    "address": f"{eircode_formatted}, Ireland",
                    "key": google_api_key,
                    "components": "country:IE"
                }
                
                response = await client.get(url, params=params)
                data = response.json()
                
                if data.get("status") == "OK" and data.get("results"):
                    result = data["results"][0]
                    location = result["geometry"]["location"]
                    lat, lon = location["lat"], location["lng"]
                    
                    if 51 <= lat <= 56 and -11 <= lon <= -5:
                        return {
                            "lat": lat,
                            "lon": lon,
                            "display_name": result.get("formatted_address", eircode_formatted),
                            "source": "google"
                        }
                elif data.get("status") == "REQUEST_DENIED":
                    print(f"Google Geocoding API not enabled. Enable it at: https://console.cloud.google.com/apis/library/geocoding-backend.googleapis.com")
            except Exception as e:
                print(f"Google geocoding error: {e}")
        
        # Strategy 2: Nominatim with address search
        try:
            url = "https://nominatim.openstreetmap.org/search"
            # Try searching for eircode as address
            params = {
                "q": f"{eircode_formatted} Ireland",
                "format": "json",
                "countrycodes": "ie",
                "limit": 5
            }
            headers = {"User-Agent": "MannaGardenClassifier/1.0"}
            
            response = await client.get(url, params=params, headers=headers)
            results = response.json()
            
            # Find result that's actually in Ireland
            for result in results:
                lat = float(result["lat"])
                lon = float(result["lon"])
                if 51 <= lat <= 56 and -11 <= lon <= -5:
                    return {
                        "lat": lat,
                        "lon": lon,
                        "display_name": result.get("display_name", ""),
                        "source": "nominatim"
                    }
        except Exception as e:
            print(f"Nominatim lookup failed: {e}")
    
    # Strategy 3: Use routing key lookup for approximate location
    if routing_key in EIRCODE_ROUTING_COORDS:
        lat, lon = EIRCODE_ROUTING_COORDS[routing_key]
        print(f"Using routing key approximation for {eircode}")
        return {
            "lat": lat,
            "lon": lon,
            "display_name": f"{routing_key} area, Dublin, Ireland (approximate)",
            "source": "routing_key",
            "approximate": True
        }
    
    print(f"Could not geocode eircode: {eircode}")
    return None


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Garden Classification API",
    description="API for classifying front/back gardens and finding optimal delivery locations",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for serving generated maps
STATIC_DIR = Path(config.OUTPUT_DIR) / "maps"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def generate_pins_map(
    pins: List[Dict[str, Any]],
    center_lat: float,
    center_lon: float,
    radius_m: float
) -> Optional[str]:
    """
    Generate a visualization map showing buildings with garden pins.
    
    Returns the filename of the generated map.
    """
    import cv2
    import numpy as np
    from datetime import datetime
    
    try:
        # Fetch satellite imagery for the area
        image, metadata = fetch_area_image(
            center_lat, center_lon,
            radius_m=radius_m,
            zoom=19,
            tile_source=TileSource.AUTO,
            show_progress=False
        )
        
        if image is None:
            return None
        
        # Make a copy for drawing
        map_image = image.copy()
        geo_bounds = metadata["geo_bounds"]
        height, width = image.shape[:2]
        
        # Fetch buildings to draw
        buildings = fetch_buildings(center_lat, center_lon, radius_m, show_progress=False)
        
        # Draw building outlines
        for _, building in buildings.iterrows():
            try:
                if building.geometry.geom_type == 'Polygon':
                    coords = list(building.geometry.exterior.coords)
                elif building.geometry.geom_type == 'MultiPolygon':
                    coords = list(building.geometry.geoms[0].exterior.coords)
                else:
                    continue
                
                # Convert to pixel coordinates
                pts = []
                for lon, lat in coords:
                    px = int((lon - geo_bounds["west"]) / (geo_bounds["east"] - geo_bounds["west"]) * width)
                    py = int((geo_bounds["north"] - lat) / (geo_bounds["north"] - geo_bounds["south"]) * height)
                    pts.append([px, py])
                
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(map_image, [pts], True, (255, 255, 255), 2)  # White outline
            except Exception:
                continue
        
        # Draw pins
        for pin in pins:
            lat = pin["lat"]
            lon = pin["lon"]
            garden_type = pin.get("garden_type", "unknown")
            score = pin.get("score", 0)
            
            # Convert to pixel coordinates
            px = int((lon - geo_bounds["west"]) / (geo_bounds["east"] - geo_bounds["west"]) * width)
            py = int((geo_bounds["north"] - lat) / (geo_bounds["north"] - geo_bounds["south"]) * height)
            
            # Color based on garden type
            if garden_type == "front":
                color = (0, 255, 0)  # Green for front
            elif garden_type == "back":
                color = (255, 0, 0)  # Blue for back (BGR)
            else:
                color = (0, 255, 255)  # Yellow for unknown
            
            # Size based on score
            radius = max(5, int(score / 10))
            
            # Draw pin marker
            cv2.circle(map_image, (px, py), radius, color, -1)
            cv2.circle(map_image, (px, py), radius + 2, (255, 255, 255), 2)
            
            # Add score label
            label = f"{int(score)}"
            cv2.putText(map_image, label, (px + radius + 3, py + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add legend
        legend_y = 30
        cv2.rectangle(map_image, (10, 10), (180, 90), (0, 0, 0), -1)
        cv2.rectangle(map_image, (10, 10), (180, 90), (255, 255, 255), 1)
        cv2.circle(map_image, (25, legend_y), 8, (0, 255, 0), -1)
        cv2.putText(map_image, "Front Garden", (40, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(map_image, (25, legend_y + 25), 8, (255, 0, 0), -1)
        cv2.putText(map_image, "Back Garden", (40, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(map_image, f"Pins: {len(pins)}", (20, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save the map
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pins_map_{timestamp}.png"
        filepath = STATIC_DIR / filename
        cv2.imwrite(str(filepath), map_image)
        
        return filename
        
    except Exception as e:
        print(f"Error generating map: {e}")
        return None


def get_tile_source(source_enum: TileSourceEnum) -> TileSource:
    """Convert API enum to internal TileSource."""
    return {
        TileSourceEnum.google: TileSource.GOOGLE,
        TileSourceEnum.manna: TileSource.MANNA,
        TileSourceEnum.auto: TileSource.AUTO
    }.get(source_enum, TileSource.AUTO)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.1.0"
    }


@app.post("/api/garden-pins", response_model=GardenPinResponse)
async def get_garden_pins(request: GardenPinRequest):
    """
    Get front and back garden delivery pins for a single location.
    
    Provide either:
    - lat/lon coordinates, OR
    - an Irish eircode
    
    Returns the best delivery pin locations for front and back gardens
    of the nearest house.
    
    Scoring:
    - 100 = grass (best)
    - 75 = driveway
    - 60 = paved area
    - 50 = potential parking area
    """
    start_time = time.time()
    
    # Resolve location
    lat, lon = None, None
    geocode_info = None
    
    if request.eircode:
        # Geocode eircode (async)
        geocode_result = await geocode_eircode(request.eircode)
        if geocode_result is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not geocode eircode: {request.eircode}"
            )
        lat, lon = geocode_result["lat"], geocode_result["lon"]
        geocode_info = geocode_result.get("display_name")
    elif request.lat is not None and request.lon is not None:
        lat, lon = request.lat, request.lon
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'eircode' or both 'lat' and 'lon' must be provided"
        )
    
    # Get classifier
    tile_source = get_tile_source(request.tile_source)
    classifier = FastGardenClassifier(tile_source=tile_source)
    
    # Run classification in thread pool (CPU-bound)
    result = await run_in_executor(
        classifier.get_garden_pins,
        lat, lon,
        not request.skip_cache  # use_cache
    )
    
    elapsed = time.time() - start_time
    
    metadata = {
        "input_lat": lat,
        "input_lon": lon,
        "elapsed_seconds": round(elapsed, 3),
        "tile_source": request.tile_source.value,
        **result.get("metadata", {})
    }
    
    if request.eircode:
        metadata["eircode"] = request.eircode
        if geocode_info:
            metadata["geocoded_address"] = geocode_info
    
    return GardenPinResponse(
        front=result.get("front"),
        back=result.get("back"),
        metadata=metadata
    )


@app.post("/api/garden-pins/batch", response_model=BatchPinResponse)
async def get_garden_pins_batch(request: BatchPinRequest):
    """
    Get all garden delivery pins within a radius.
    
    Returns a list of front and back garden pins for all buildings
    within the specified radius.
    
    PERFORMANCE TIP: For large radii (>100m), precompute first:
        POST /api/precompute with same center and radius
    
    Then subsequent batch queries will be nearly instant (<1s).
    
    Set generate_map=true to create a visualization showing all pins.
    """
    start_time = time.time()
    
    tile_source = get_tile_source(request.tile_source)
    manager = PrecomputeManager(tile_source=tile_source)
    
    # Check if area is precomputed
    is_cached = manager.is_area_cached(request.lat, request.lon, request.radius_m)
    
    # Run in thread pool
    pins = await run_in_executor(
        manager.get_pins_in_radius,
        request.lat,
        request.lon,
        request.radius_m,
        request.garden_type,
        request.min_score
    )
    
    elapsed = time.time() - start_time
    
    # Generate visualization map if requested
    map_url = None
    if request.generate_map and pins:
        map_filename = await run_in_executor(
            generate_pins_map,
            pins,
            request.lat,
            request.lon,
            request.radius_m
        )
        if map_filename:
            map_url = f"/static/{map_filename}"
    
    return BatchPinResponse(
        pins=pins,
        count=len(pins),
        metadata={
            "center_lat": request.lat,
            "center_lon": request.lon,
            "radius_m": request.radius_m,
            "garden_type": request.garden_type,
            "min_score": request.min_score,
            "elapsed_seconds": round(elapsed, 3),
            "tile_source": request.tile_source.value,
            "from_cache": is_cached,
            "tip": None if is_cached else "Precompute this area first for faster queries: POST /api/precompute"
        },
        map_url=map_url
    )


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify_point(request: ClassifyRequest):
    """
    Classify a GPS coordinate as front garden, back garden, or other.
    
    Returns:
    - classification: "front_garden", "back_garden", "grass", "driveway", "building", "road", etc.
    - surface_type: The type of surface at this location
    - score: Delivery suitability score (0-100)
      - 100 = grass (best for delivery)
      - 75 = driveway
      - 60 = paved
      - 50 = parking area
    - distance_to_building_m: Distance to the nearest building
    """
    tile_source = get_tile_source(request.tile_source)
    classifier = FastGardenClassifier(tile_source=tile_source)
    
    # Run in thread pool
    result = await run_in_executor(
        classifier.classify_point,
        request.lat, request.lon,
        not request.skip_cache
    )
    
    return ClassifyResponse(
        lat=result.get("lat", request.lat),
        lon=result.get("lon", request.lon),
        classification=result.get("classification", "unknown"),
        surface_type=result.get("surface_type"),
        score=result.get("score"),
        distance_to_building_m=result.get("distance_to_building_m")
    )


@app.post("/api/precompute", response_model=PrecomputeResponse)
async def precompute_area_endpoint(
    request: PrecomputeRequest,
    background_tasks: BackgroundTasks
):
    """
    Precompute garden classification for a large area (FAST single-pass).
    
    NEW: Fetches all data ONCE for the entire area instead of per-chunk.
    500m radius should take ~1-3 minutes instead of hours.
    
    Subsequent batch queries within this area will be nearly instant.
    """
    tile_source = get_tile_source(request.tile_source)
    manager = PrecomputeManager(tile_source=tile_source)
    
    # Check if already cached
    if not request.skip_cache and manager.is_area_cached(request.lat, request.lon, request.radius_m):
        return PrecomputeResponse(
            status="completed",
            message=f"Area already precomputed. Use /api/garden-pins/batch to query.",
            summary={"from_cache": True}
        )
    
    # Run synchronously in thread pool (much faster now - single pass)
    summary = await run_in_executor(
        manager.precompute_area,
        request.lat,
        request.lon,
        request.radius_m,
        True,  # parallel (unused in new arch but kept for compat)
        True   # show_progress
    )
    
    return PrecomputeResponse(
        status="completed",
        message=f"Precomputed {summary.get('total_buildings', 0)} buildings, "
                f"{summary.get('total_pins', 0)} pins in "
                f"{summary.get('elapsed_seconds', 0):.1f}s",
        summary=summary
    )


@app.get("/api/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get statistics about the caches."""
    from pathlib import Path
    import config
    
    # Count fast cache entries
    fast_cache_dir = Path(config.OUTPUT_DIR) / "fast_cache"
    fast_cache_count = len(list(fast_cache_dir.glob("*.pkl"))) if fast_cache_dir.exists() else 0
    
    # Get precompute stats
    manager = PrecomputeManager()
    precompute_stats = manager.get_cache_stats()
    
    return CacheStatsResponse(
        fast_cache_entries=fast_cache_count,
        precompute_areas=precompute_stats.get("total_areas", 0),
        total_pins=precompute_stats.get("total_pins", 0)
    )


@app.delete("/api/cache")
async def clear_cache():
    """Clear all caches."""
    # Clear fast cache
    classifier = FastGardenClassifier()
    classifier.clear_cache()
    
    # Clear precompute cache
    manager = PrecomputeManager()
    manager.clear_cache()
    
    return {"status": "cleared", "message": "All caches cleared"}


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Garden Classification API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    
    print(f"\n🌿 Garden Classification API v1.1.0")
    print(f"   Starting server on http://{args.host}:{args.port}")
    print(f"   API docs: http://{args.host}:{args.port}/docs")
    print(f"   Workers: {args.workers}")
    print()
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
