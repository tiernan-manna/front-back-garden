"""
Precomputation Module for Batch Processing

REWRITTEN for speed:
- Fetches OSM data ONCE for entire area (not per-chunk)
- Fetches tiles ONCE as a single stitched image
- Processes vegetation detection ONCE
- Classifies ONCE
- Then iterates buildings to find exactly ONE front + ONE back pin per building
- Deduplicates pins so no building gets multiple pins

Previous approach: 4 hours for 500m (25 chunks x separate fetch + process each)
New approach: ~1-3 minutes for 500m (1 fetch + 1 process + iterate buildings)
"""

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set
import threading
import warnings

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from tqdm import tqdm

import config
from src.tiles import (
    TileSource,
    fetch_area_image,
)
from src.osm import (
    fetch_buildings,
    fetch_roads,
    fetch_driveways,
    fetch_address_polygons,
    geometry_to_pixel_coords,
)
from src.garden_detector import detect_green_areas, exclude_buildings_from_mask, exclude_roads_from_mask
from src.classifier import GardenClassifier
from src.delivery_pins import DeliveryPinFinder, DeliveryPin

# Precomputation cache directory
PRECOMPUTE_CACHE_DIR = Path(config.OUTPUT_DIR) / "precompute"


@dataclass
class PrecomputedArea:
    """Cached result for a precomputed area."""
    center_lat: float
    center_lon: float
    radius_m: float
    zoom: int
    tile_source: str
    computed_at: float
    num_buildings: int
    num_front_gardens: int
    num_back_gardens: int
    delivery_pins: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PrecomputedArea':
        return cls(**data)


class PrecomputeManager:
    """
    Manages precomputation of garden data for large areas.

    FAST single-pass approach:
    1. Fetch OSM buildings/roads/driveways ONCE for the whole area
    2. Fetch satellite imagery ONCE as a stitched image
    3. Detect vegetation ONCE
    4. Classify front/back ONCE
    5. Find exactly ONE front + ONE back pin per building
    """

    def __init__(
        self,
        cache_dir: Path = None,
        tile_source: TileSource = TileSource.AUTO,
        zoom: int = None,
        max_workers: int = 4
    ):
        self.cache_dir = cache_dir or PRECOMPUTE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tile_source = tile_source
        self.zoom = zoom or config.ZOOM_LEVEL
        self.max_workers = max_workers

        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, dict]:
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        with self._lock:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)

    # ------------------------------------------------------------------
    # Cache key / path helpers
    # ------------------------------------------------------------------

    def _get_area_key(self, lat: float, lon: float, radius_m: float) -> str:
        key_data = f"{lat:.5f}_{lon:.5f}_{radius_m:.0f}_{self.zoom}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def _get_area_cache_path(self, area_key: str) -> Path:
        return self.cache_dir / f"area_{area_key}.pkl"

    # Backwards-compat: old chunk key for is_area_cached check
    def _get_chunk_key(self, lat: float, lon: float) -> str:
        return f"{round(lat, 4):.4f}_{round(lon, 4):.4f}_{self.zoom}"

    def _get_chunk_cache_path(self, chunk_key: str) -> Path:
        return self.cache_dir / f"chunk_{chunk_key}.pkl"

    # ------------------------------------------------------------------
    # Main precompute - SINGLE PASS
    # ------------------------------------------------------------------

    def precompute_area(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        parallel: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Precompute all data for an area in a SINGLE PASS.

        Steps:
        1. Check cache
        2. Fetch imagery (one stitched image)
        3. Fetch OSM data (one set of API calls)
        4. Detect vegetation
        5. Classify front/back
        6. Find one pin per building per garden type
        7. Cache results
        """
        start_time = time.time()

        area_key = self._get_area_key(center_lat, center_lon, radius_m)
        cache_path = self._get_area_cache_path(area_key)

        # Check cache
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if show_progress:
                    print(f"Loaded from cache: {len(cached.delivery_pins)} pins")
                elapsed = time.time() - start_time
                return {
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "radius_m": radius_m,
                    "total_pins": len(cached.delivery_pins),
                    "total_buildings": cached.num_buildings,
                    "elapsed_seconds": round(elapsed, 2),
                    "tile_source": self.tile_source.value,
                    "from_cache": True,
                }
            except Exception:
                pass

        # ---- STEP 1: Fetch imagery (ONCE) ----
        if show_progress:
            print(f"[1/5] Fetching satellite imagery for {radius_m}m radius...")
        t0 = time.time()

        image, metadata = fetch_area_image(
            center_lat, center_lon,
            radius_m=radius_m,
            zoom=self.zoom,
            show_progress=show_progress,
            use_cache=True,
            tile_source=self.tile_source,
        )

        if image is None:
            return {"error": "Failed to fetch imagery", "elapsed_seconds": time.time() - start_time}

        geo_bounds = metadata["geo_bounds"]
        image_size = tuple(metadata["image_size"])

        if show_progress:
            print(f"    Image: {image_size[0]}x{image_size[1]} ({time.time()-t0:.1f}s)")

        # ---- STEP 2: Fetch OSM data (ONCE) ----
        if show_progress:
            print(f"[2/5] Fetching OSM data...")
        t0 = time.time()

        buildings = fetch_buildings(center_lat, center_lon, radius_m, show_progress=False)
        roads = fetch_roads(center_lat, center_lon, radius_m, show_progress=False)
        driveways = fetch_driveways(center_lat, center_lon, radius_m, show_progress=False)

        try:
            address_polygons = fetch_address_polygons(center_lat, center_lon, radius_m, show_progress=False)
        except Exception:
            address_polygons = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        if show_progress:
            print(f"    {len(buildings)} buildings, {len(roads)} roads ({time.time()-t0:.1f}s)")

        if buildings.empty:
            return {
                "center_lat": center_lat,
                "center_lon": center_lon,
                "radius_m": radius_m,
                "total_pins": 0,
                "total_buildings": 0,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "tile_source": self.tile_source.value,
                "from_cache": False,
            }

        # ---- STEP 3: Detect vegetation (ONCE) ----
        if show_progress:
            print(f"[3/5] Detecting vegetation...")
        t0 = time.time()

        vegetation_mask, _ = detect_green_areas(image, show_progress=False, enhanced=False)

        # Convert to pixel coords and exclude buildings/roads from vegetation
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

        vegetation_mask = exclude_buildings_from_mask(vegetation_mask, building_polys_px)
        vegetation_mask = exclude_roads_from_mask(vegetation_mask, road_lines_px, road_width_px=8)

        if show_progress:
            print(f"    Vegetation detected ({time.time()-t0:.1f}s)")

        # ---- STEP 4: Classify front/back (ONCE) ----
        if show_progress:
            print(f"[4/5] Classifying front/back gardens...")
        t0 = time.time()

        classifier = GardenClassifier(
            buildings=buildings,
            roads=roads,
            geo_bounds=geo_bounds,
            image_size=image_size,
            center_lat=center_lat,
            center_lon=center_lon,
            driveways=driveways,
            exclusion_zones=None,        # Skip for speed
            property_boundaries=None,    # Skip for speed
            address_polygons=address_polygons,
        )

        classification_mask = classifier.classify_mask_fast(vegetation_mask, show_progress=False)

        if show_progress:
            print(f"    Classification complete ({time.time()-t0:.1f}s)")

        # ---- STEP 5: Find pins - ONE per building per garden type ----
        if show_progress:
            print(f"[5/5] Finding delivery pins for {len(buildings)} buildings...")
        t0 = time.time()

        pin_finder = DeliveryPinFinder(
            classification_mask=classification_mask,
            vegetation_mask=vegetation_mask,
            buildings=buildings,
            roads=roads,
            driveways=driveways,
            geo_bounds=geo_bounds,
            image_size=image_size,
            center_lat=center_lat,
            center_lon=center_lon,
            building_directions=getattr(classifier, 'building_directions', None),
        )

        all_pins: List[DeliveryPin] = []
        seen_buildings: Set[str] = set()

        for idx in range(len(buildings)):
            # Get a stable building ID
            building = buildings.iloc[idx]
            building_id = None
            if "osm_id" in building.index:
                building_id = str(building["osm_id"])
            elif hasattr(building, 'name') and building.name is not None:
                building_id = str(building.name)
            else:
                building_id = f"bld_{idx}"

            # Skip if already processed (dedup)
            if building_id in seen_buildings:
                continue
            seen_buildings.add(building_id)

            # Find ONE front pin and ONE back pin for this building
            front_pin = pin_finder.find_best_pin_for_building(idx, "front")
            if front_pin and front_pin.score >= 15:
                front_pin.building_id = building_id
                all_pins.append(front_pin)

            back_pin = pin_finder.find_best_pin_for_building(idx, "back")
            if back_pin and back_pin.score >= 15:
                back_pin.building_id = building_id
                all_pins.append(back_pin)

        if show_progress:
            print(f"    Found {len(all_pins)} pins ({time.time()-t0:.1f}s)")

        # ---- Cache results ----
        num_front = sum(1 for p in all_pins if p.garden_type == "front")
        num_back = sum(1 for p in all_pins if p.garden_type == "back")

        precomputed = PrecomputedArea(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_m=radius_m,
            zoom=self.zoom,
            tile_source=self.tile_source.value if hasattr(self.tile_source, 'value') else str(self.tile_source),
            computed_at=time.time(),
            num_buildings=len(buildings),
            num_front_gardens=num_front,
            num_back_gardens=num_back,
            delivery_pins=[pin.to_dict() for pin in all_pins],
        )

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(precomputed, f)
        except Exception as e:
            warnings.warn(f"Failed to save cache: {e}")

        # Update index
        with self._lock:
            self.index[area_key] = {
                "lat": center_lat,
                "lon": center_lon,
                "radius_m": radius_m,
                "computed_at": precomputed.computed_at,
                "num_pins": len(all_pins),
                "num_buildings": len(buildings),
            }
        self._save_index()

        elapsed = time.time() - start_time

        if show_progress:
            print(f"\nPrecompute complete:")
            print(f"   Buildings: {len(buildings)}")
            print(f"   Front pins: {num_front}")
            print(f"   Back pins: {num_back}")
            print(f"   Total time: {elapsed:.1f}s")

        return {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_m": radius_m,
            "total_pins": len(all_pins),
            "total_buildings": len(buildings),
            "num_front": num_front,
            "num_back": num_back,
            "elapsed_seconds": round(elapsed, 2),
            "tile_source": self.tile_source.value if hasattr(self.tile_source, 'value') else str(self.tile_source),
            "from_cache": False,
        }

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def is_area_cached(self, center_lat: float, center_lon: float, radius_m: float) -> bool:
        """Check if the exact area is cached."""
        area_key = self._get_area_key(center_lat, center_lon, radius_m)
        return self._get_area_cache_path(area_key).exists()

    def get_pins_in_radius(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        garden_type: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get all delivery pins within a radius.

        Tries exact cache first. If not found, checks for any cached area
        that covers the requested radius. Falls back to computing on-the-fly.
        """
        # Try exact cache
        area_key = self._get_area_key(center_lat, center_lon, radius_m)
        cache_path = self._get_area_cache_path(area_key)

        precomputed = None

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    precomputed = pickle.load(f)
            except Exception:
                pass

        # Try to find a larger cached area that covers this request
        if precomputed is None:
            precomputed = self._find_covering_cache(center_lat, center_lon, radius_m)

        # Compute on-the-fly if no cache
        if precomputed is None:
            summary = self.precompute_area(center_lat, center_lon, radius_m, show_progress=True)
            if "error" in summary:
                return []
            # Re-load from cache
            try:
                with open(cache_path, "rb") as f:
                    precomputed = pickle.load(f)
            except Exception:
                return []

        # Filter pins by distance, type, and score
        pins = []
        for pin in precomputed.delivery_pins:
            dist = self._haversine_distance(center_lat, center_lon, pin["lat"], pin["lon"])
            if dist > radius_m:
                continue
            if garden_type and pin.get("garden_type") != garden_type:
                continue
            if pin.get("score", 0) < min_score:
                continue
            pins.append(pin)

        return pins

    def _find_covering_cache(self, lat: float, lon: float, radius_m: float) -> Optional[PrecomputedArea]:
        """Find a cached area that fully covers the requested area."""
        for key, info in self.index.items():
            cached_lat = info.get("lat", 0)
            cached_lon = info.get("lon", 0)
            cached_radius = info.get("radius_m", 0)

            # Check if the cached area fully contains our requested area
            dist_between_centers = self._haversine_distance(lat, lon, cached_lat, cached_lon)
            if dist_between_centers + radius_m <= cached_radius:
                cache_path = self._get_area_cache_path(key)
                if cache_path.exists():
                    try:
                        with open(cache_path, "rb") as f:
                            return pickle.load(f)
                    except Exception:
                        continue
        return None

    def get_nearest_building_pins(
        self,
        lat: float,
        lon: float
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get front and back garden pins for the building nearest to a coordinate."""
        # Search for any cached area containing this point
        for key, info in self.index.items():
            cached_lat = info.get("lat", 0)
            cached_lon = info.get("lon", 0)
            cached_radius = info.get("radius_m", 0)

            dist = self._haversine_distance(lat, lon, cached_lat, cached_lon)
            if dist <= cached_radius:
                cache_path = self._get_area_cache_path(key)
                if cache_path.exists():
                    try:
                        with open(cache_path, "rb") as f:
                            precomputed = pickle.load(f)

                        best_front = None
                        best_back = None
                        min_front_dist = float("inf")
                        min_back_dist = float("inf")

                        for pin in precomputed.delivery_pins:
                            d = self._haversine_distance(lat, lon, pin["lat"], pin["lon"])
                            if pin["garden_type"] == "front" and d < min_front_dist:
                                min_front_dist = d
                                best_front = pin
                            elif pin["garden_type"] == "back" and d < min_back_dist:
                                min_back_dist = d
                                best_back = pin

                        return {"front": best_front, "back": best_back}
                    except Exception:
                        continue

        return {"front": None, "back": None}

    def classify_point(self, lat: float, lon: float) -> Dict[str, Any]:
        """Classify a single GPS coordinate using cached data if available."""
        # Check if any cached area covers this point
        for key, info in self.index.items():
            cached_lat = info.get("lat", 0)
            cached_lon = info.get("lon", 0)
            cached_radius = info.get("radius_m", 0)

            dist = self._haversine_distance(lat, lon, cached_lat, cached_lon)
            if dist <= cached_radius:
                cache_path = self._get_area_cache_path(key)
                if cache_path.exists():
                    try:
                        with open(cache_path, "rb") as f:
                            precomputed = pickle.load(f)

                        # Find nearest pin to classify
                        nearest_pin = None
                        min_dist = float("inf")
                        for pin in precomputed.delivery_pins:
                            d = self._haversine_distance(lat, lon, pin["lat"], pin["lon"])
                            if d < min_dist:
                                min_dist = d
                                nearest_pin = pin

                        if nearest_pin and min_dist < 30:
                            return {
                                "lat": lat,
                                "lon": lon,
                                "classification": f"{nearest_pin['garden_type']}_garden",
                                "surface_type": nearest_pin.get("surface_type", "unknown"),
                                "score": nearest_pin.get("score", 0),
                                "distance_to_building_m": nearest_pin.get("distance_to_building_m", 0),
                            }
                    except Exception:
                        continue

        return {"lat": lat, "lon": lon, "classification": "unknown", "score": 0}

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def clear_cache(self):
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self._save_index()
        print("Precompute cache cleared.")

    def get_cache_stats(self) -> Dict[str, Any]:
        total_chunks = len(self.index)
        total_pins = sum(entry.get("num_pins", 0) for entry in self.index.values())
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl")) if self.cache_dir.exists() else 0
        return {
            "total_areas": total_chunks,
            "total_pins": total_pins,
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "cached_areas": [
                {"key": k, "lat": v.get("lat"), "lon": v.get("lon"), "radius_m": v.get("radius_m"), "pins": v.get("num_pins", 0)}
                for k, v in self.index.items()
            ],
        }


def precompute_area(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    tile_source: str = "auto",
    zoom: int = None,
    parallel: bool = True
) -> Dict[str, Any]:
    """Convenience function to precompute an area."""
    source_map = {
        "google": TileSource.GOOGLE,
        "manna": TileSource.MANNA,
        "auto": TileSource.AUTO,
    }
    manager = PrecomputeManager(
        tile_source=source_map.get(tile_source, TileSource.AUTO),
        zoom=zoom,
    )
    return manager.precompute_area(
        center_lat=center_lat,
        center_lon=center_lon,
        radius_m=radius_m,
        parallel=parallel,
    )
