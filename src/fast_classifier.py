"""
Fast Garden Classifier Module

Optimized for quick single-point and small-area queries.
Uses caching and reduced processing for speed.

Key optimizations:
- Smaller processing radius for point queries
- Skip visualization steps
- Cache at coordinate level, not chunk level
- Async-friendly design
"""

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import config
from src.tiles import TileSource, fetch_area_image
from src.osm import (
    fetch_all_osm_data, geometry_to_pixel_coords, project_to_meters, 
    OSM_CACHE_DIR, get_osm_cache_key, fetch_buildings, fetch_roads, fetch_driveways,
    fetch_address_polygons
)
from src.garden_detector import detect_green_areas, exclude_buildings_from_mask, exclude_roads_from_mask
from src.classifier import GardenClassifier
from src.delivery_pins import DeliveryPinFinder, SurfaceType

# Cache directory for fast lookups
FAST_CACHE_DIR = Path(config.OUTPUT_DIR) / "fast_cache"


@dataclass
class CachedResult:
    """Cached result for a specific coordinate."""
    lat: float
    lon: float
    front_pin: Optional[Dict[str, Any]]
    back_pin: Optional[Dict[str, Any]]
    classification: str
    surface_type: str
    score: float
    distance_to_building_m: float
    computed_at: float
    processing_time: float


class FastGardenClassifier:
    """
    Optimized classifier for quick API responses.
    
    Key differences from PrecomputeManager:
    - Uses smaller radius (50m) for single-point queries
    - Caches by precise coordinate hash, not chunk grid
    - Skips unnecessary processing steps
    - Designed for concurrent access
    """
    
    # Small radius for point queries - just enough to find nearest building
    POINT_QUERY_RADIUS = 50  # meters
    
    # Cache TTL (time to live) in seconds - 24 hours
    CACHE_TTL = 86400
    
    def __init__(
        self,
        tile_source: TileSource = TileSource.AUTO,
        zoom: int = 19,
        cache_dir: Path = None
    ):
        self.tile_source = tile_source
        self.zoom = zoom
        self.cache_dir = cache_dir or FAST_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, lat: float, lon: float) -> str:
        """Generate precise cache key for coordinates."""
        # Use 5 decimal places (~1m precision)
        key_data = f"{lat:.5f}_{lon:.5f}_{self.zoom}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, lat: float, lon: float) -> Optional[CachedResult]:
        """Load result from cache if valid."""
        cache_key = self._get_cache_key(lat, lon)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    result = pickle.load(f)
                
                # CRITICAL: Validate cached coordinates match request
                # This prevents returning wrong results from hash collisions or old cache
                if abs(result.lat - lat) > 0.0001 or abs(result.lon - lon) > 0.0001:
                    # Cache mismatch - delete stale entry
                    cache_path.unlink()
                    return None
                
                # Check TTL
                if time.time() - result.computed_at < self.CACHE_TTL:
                    return result
                else:
                    # Expired - delete
                    cache_path.unlink()
            except Exception:
                pass
        
        return None
    
    def _save_to_cache(self, result: CachedResult):
        """Save result to cache."""
        cache_key = self._get_cache_key(result.lat, result.lon)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass
    
    def get_garden_pins(
        self,
        lat: float,
        lon: float,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get front and back garden pins for a location.
        
        Optimized for speed - uses small radius and caching.
        
        Args:
            lat: Latitude
            lon: Longitude
            use_cache: Use cached results if available
            
        Returns:
            Dict with 'front', 'back' pins and metadata
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(lat, lon)
            if cached:
                return {
                    "front": cached.front_pin,
                    "back": cached.back_pin,
                    "metadata": {
                        "from_cache": True,
                        "processing_time": cached.processing_time,
                        "cached_at": cached.computed_at
                    }
                }
        
        # Process the location
        try:
            result = self._process_location(lat, lon)
            
            processing_time = time.time() - start_time
            
            # Cache the result
            cached_result = CachedResult(
                lat=lat,
                lon=lon,
                front_pin=result.get("front"),
                back_pin=result.get("back"),
                classification=result.get("classification", "unknown"),
                surface_type=result.get("surface_type", "unknown"),
                score=result.get("score", 0),
                distance_to_building_m=result.get("distance_to_building_m", 0),
                computed_at=time.time(),
                processing_time=processing_time
            )
            self._save_to_cache(cached_result)
            
            return {
                "front": result.get("front"),
                "back": result.get("back"),
                "metadata": {
                    "from_cache": False,
                    "processing_time": round(processing_time, 2)
                }
            }
            
        except Exception as e:
            return {
                "front": None,
                "back": None,
                "metadata": {
                    "error": str(e),
                    "processing_time": round(time.time() - start_time, 2)
                }
            }
    
    def classify_point(
        self,
        lat: float,
        lon: float,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Classify a single GPS coordinate.
        
        Returns classification, surface type, and score.
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(lat, lon)
            if cached:
                return {
                    "lat": lat,
                    "lon": lon,
                    "classification": cached.classification,
                    "surface_type": cached.surface_type,
                    "score": cached.score,
                    "distance_to_building_m": cached.distance_to_building_m,
                    "from_cache": True
                }
        
        # Process
        try:
            result = self._process_location(lat, lon)
            return {
                "lat": lat,
                "lon": lon,
                "classification": result.get("classification", "unknown"),
                "surface_type": result.get("surface_type", "unknown"),
                "score": result.get("score", 0),
                "distance_to_building_m": result.get("distance_to_building_m", 0),
                "from_cache": False
            }
        except Exception as e:
            return {
                "lat": lat,
                "lon": lon,
                "classification": "error",
                "error": str(e)
            }
    
    def _process_location(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Process a location to get garden classification and pins.
        
        Optimized for speed with minimal processing:
        - Small radius (50m)
        - Skip slow OSM queries (exclusion zones, property boundaries, addresses)
        - Use fast classification method
        """
        # Fetch imagery with small radius
        image, metadata = fetch_area_image(
            lat, lon,
            self.POINT_QUERY_RADIUS,
            self.zoom,
            show_progress=False,
            use_cache=True,
            tile_source=self.tile_source
        )
        
        if image is None:
            raise ValueError("Failed to fetch imagery")
        
        # Fetch ONLY essential OSM data (buildings, roads, driveways)
        # Skip slow queries: exclusion_zones, property_boundaries, address_polygons
        osm_data = self._fetch_essential_osm(lat, lon, self.POINT_QUERY_RADIUS)
        buildings = osm_data["buildings"]
        roads = osm_data["roads"]
        driveways = osm_data.get("driveways", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
        
        if buildings.empty:
            return {
                "classification": "no_buildings",
                "surface_type": "unknown",
                "score": 0,
                "distance_to_building_m": 999,
                "front": None,
                "back": None
            }
        
        geo_bounds = metadata["geo_bounds"]
        image_size = tuple(metadata["image_size"])
        
        # Detect vegetation (optimized - skip enhanced detection)
        vegetation_mask, _ = detect_green_areas(image, show_progress=False, enhanced=False)
        
        # Convert to pixel coordinates
        building_polys_px = []
        for _, building in buildings.iterrows():
            coords = geometry_to_pixel_coords(building.geometry, geo_bounds, image_size)
            if coords:
                building_polys_px.append(coords)
        
        road_lines_px = []
        for _, road in roads.iterrows():
            coords = geometry_to_pixel_coords(road.geometry, geo_bounds, image_size)
            if coords:
                road_lines_px.append(coords)
        
        # Exclude buildings and roads
        vegetation_mask = exclude_buildings_from_mask(vegetation_mask, building_polys_px)
        vegetation_mask = exclude_roads_from_mask(vegetation_mask, road_lines_px, road_width_px=8)
        
        # Classify gardens - include address data for accurate front/back detection
        address_polygons = osm_data.get("address_polygons", None)
        
        classifier = GardenClassifier(
            buildings=buildings,
            roads=roads,
            geo_bounds=geo_bounds,
            image_size=image_size,
            center_lat=lat,
            center_lon=lon,
            driveways=driveways,
            exclusion_zones=None,  # Skip for speed
            property_boundaries=None,  # Skip for speed
            address_polygons=address_polygons  # Include for accurate front/back
        )
        
        # Use fast classification (skip region-based for speed)
        classification_mask = classifier.classify_mask_fast(vegetation_mask, show_progress=False)
        
        # Find delivery pins
        pin_finder = DeliveryPinFinder(
            classification_mask=classification_mask,
            vegetation_mask=vegetation_mask,
            buildings=buildings,
            roads=roads,
            driveways=driveways,
            geo_bounds=geo_bounds,
            image_size=image_size,
            center_lat=lat,
            center_lon=lon
        )
        
        # Get pins for nearest building
        pins = pin_finder.find_nearest_building_pins(lat, lon)
        
        # Classify the specific point
        point_class = pin_finder.classify_point(lat, lon)
        
        return {
            "front": pins["front"].to_dict() if pins["front"] else None,
            "back": pins["back"].to_dict() if pins["back"] else None,
            "classification": point_class.get("classification", "unknown"),
            "surface_type": point_class.get("surface_type", "unknown"),
            "score": point_class.get("score", 0),
            "distance_to_building_m": point_class.get("distance_to_building_m", 0)
        }
    
    def _fetch_essential_osm(self, lat: float, lon: float, radius_m: float) -> Dict[str, Any]:
        """
        Fetch essential OSM data for point queries.
        
        Includes address_polygons (critical for accurate front/back detection).
        
        Skips slow queries:
        - exclusion_zones (parks, pitches) - takes 1-6 minutes
        - property_boundaries (walls, fences) - takes 1-2 minutes
        """
        # Check if we have cached essential data
        cache_key = get_osm_cache_key(lat, lon, radius_m)
        fast_cache_path = OSM_CACHE_DIR / f"{cache_key}_fast_v2.pkl"
        
        if fast_cache_path.exists():
            try:
                with open(fast_cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        
        # Fetch essential data (including addresses for front/back detection)
        buildings = fetch_buildings(lat, lon, radius_m, show_progress=False)
        roads = fetch_roads(lat, lon, radius_m, show_progress=False)
        driveways = fetch_driveways(lat, lon, radius_m, show_progress=False)
        
        # Address polygons tell us which street the building faces (critical!)
        try:
            address_polygons = fetch_address_polygons(lat, lon, radius_m, show_progress=False)
        except Exception:
            address_polygons = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        data = {
            "buildings": buildings,
            "roads": roads,
            "driveways": driveways,
            "address_polygons": address_polygons
        }
        
        # Cache for reuse
        try:
            OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(fast_cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass
        
        return data
    
    def clear_cache(self):
        """Clear the fast cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global classifier instance for reuse
_fast_classifier: Optional[FastGardenClassifier] = None


def get_fast_classifier(tile_source: TileSource = TileSource.AUTO) -> FastGardenClassifier:
    """Get or create the fast classifier singleton."""
    global _fast_classifier
    if _fast_classifier is None:
        _fast_classifier = FastGardenClassifier(tile_source=tile_source)
    return _fast_classifier
