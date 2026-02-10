"""
Delivery Pin Scoring and Selection Module

Finds optimal delivery locations (pins) for front and back gardens.
Scoring prioritizes:
1. Grass (highest score)
2. Driveway near house (medium-high score)
3. Car parking area near house (medium score)

Ignores:
- Non-grass/non-driveway spaces
- Obstacles (for now - will be added later)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from src.osm import project_to_meters, geo_to_pixel


class SurfaceType(Enum):
    """Types of surfaces for delivery scoring."""
    UNKNOWN = 0
    GRASS = 1
    DRIVEWAY = 2
    PAVED = 3
    CAR_PARKING = 4
    BUILDING = 5
    ROAD = 6


@dataclass
class DeliveryPin:
    """
    Represents a potential delivery location.
    
    Attributes:
        lat: Latitude
        lon: Longitude  
        garden_type: "front" or "back"
        score: Delivery suitability score (0-100)
        surface_type: Type of surface at this location
        distance_to_building_m: Distance to nearest building in meters
        building_id: ID of the associated building (if any)
        metadata: Additional metadata
    """
    lat: float
    lon: float
    garden_type: str  # "front" or "back"
    score: float  # 0-100
    surface_type: str
    distance_to_building_m: float
    building_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Ensure all values are native Python types (not numpy)
        safe_metadata = {}
        for k, v in self.metadata.items():
            if hasattr(v, 'item'):  # numpy scalar
                safe_metadata[k] = v.item()
            else:
                safe_metadata[k] = v
        
        return {
            "lat": float(self.lat),
            "lon": float(self.lon),
            "garden_type": self.garden_type,
            "score": round(float(self.score), 2),
            "surface_type": self.surface_type,
            "distance_to_building_m": round(float(self.distance_to_building_m), 2),
            "building_id": self.building_id,
            "metadata": safe_metadata
        }


class DeliveryPinFinder:
    """
    Finds optimal delivery pin locations for front and back gardens.
    
    Scoring weights:
    - Grass: 100 (highest)
    - Driveway near house: 75
    - Paved area: 60
    - Car parking near house: 50
    
    Distance penalties:
    - Too close to building (<2m): -20
    - Too far from building (>20m): -10 per 5m over
    - Near road: -5
    
    Key validation:
    - Back garden pins must be BEHIND the building (opposite to road)
    - Pins must NOT be in front of any OTHER building (neighbor's front garden)
    """
    
    # Scoring weights for surface types
    SURFACE_SCORES = {
        SurfaceType.GRASS: 100,
        SurfaceType.DRIVEWAY: 75,
        SurfaceType.PAVED: 60,
        SurfaceType.CAR_PARKING: 50,
        SurfaceType.UNKNOWN: 30,
        SurfaceType.BUILDING: 0,
        SurfaceType.ROAD: 0,
    }
    
    # Optimal distance range from building (meters)
    OPTIMAL_DISTANCE_MIN = 2.0
    OPTIMAL_DISTANCE_MAX = 15.0
    
    def __init__(
        self,
        classification_mask: np.ndarray,
        vegetation_mask: np.ndarray,
        buildings: gpd.GeoDataFrame,
        roads: gpd.GeoDataFrame,
        driveways: gpd.GeoDataFrame,
        geo_bounds: dict,
        image_size: Tuple[int, int],
        center_lat: float,
        center_lon: float,
        meters_per_pixel: float = None,
        building_directions: Dict = None
    ):
        """
        Initialize the delivery pin finder.
        
        Args:
            classification_mask: Classification mask (1=front, 2=back)
            vegetation_mask: Binary vegetation mask
            buildings: GeoDataFrame with building polygons
            roads: GeoDataFrame with road geometries
            driveways: GeoDataFrame with driveway geometries
            geo_bounds: Geographic bounds dict
            image_size: (width, height) in pixels
            center_lat: Center latitude
            center_lon: Center longitude
            meters_per_pixel: Meters per pixel (calculated if not provided)
            building_directions: Dict mapping building idx to direction info (from GardenClassifier)
        """
        self.classification_mask = classification_mask
        self.vegetation_mask = vegetation_mask
        self.buildings = buildings
        self.roads = roads
        self.driveways = driveways
        self.geo_bounds = geo_bounds
        self.image_size = image_size
        self.center_lat = center_lat
        self.center_lon = center_lon
        
        # Calculate meters per pixel if not provided
        if meters_per_pixel is None:
            # Approximate using latitude
            lat_range = geo_bounds["north"] - geo_bounds["south"]
            meters_lat = lat_range * 111320  # ~111km per degree latitude
            self.meters_per_pixel = meters_lat / image_size[1]
        else:
            self.meters_per_pixel = meters_per_pixel
        
        # Store or compute building directions
        self.building_directions = building_directions or {}
        if not self.building_directions and not buildings.empty and not roads.empty:
            self._compute_building_directions()
        
        # Pre-compute building and road masks
        self._precompute_masks()
        
        # Pre-compute distance transforms
        self._precompute_distances()
    
    def _compute_building_directions(self):
        """Compute road-facing direction for each building."""
        from shapely.ops import nearest_points
        from src.osm import project_to_meters
        
        # Project to meters for accurate distance calculations
        buildings_m = project_to_meters(self.buildings, self.center_lat, self.center_lon)
        roads_m = project_to_meters(self.roads, self.center_lat, self.center_lon)
        
        if roads_m.empty:
            return
        
        # Combine all roads into one geometry
        from shapely.ops import unary_union
        all_roads = unary_union(roads_m.geometry.tolist())
        
        for idx, building in buildings_m.iterrows():
            try:
                centroid = building.geometry.centroid
                
                # Find nearest point on road
                nearest_road_point = nearest_points(centroid, all_roads)[1]
                
                # Direction vector from building to road (this is the "front" direction)
                dx = nearest_road_point.x - centroid.x
                dy = nearest_road_point.y - centroid.y
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    direction = (dx / length, dy / length)
                else:
                    direction = (0, 1)
                
                # Convert centroid to pixel coords
                bld_wgs = self.buildings.loc[idx]
                c = bld_wgs.geometry.centroid
                width, height = self.image_size
                cx = int((c.x - self.geo_bounds["west"]) / 
                        (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
                cy = int((self.geo_bounds["north"] - c.y) / 
                        (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
                
                self.building_directions[idx] = {
                    "centroid_px": (cx, cy),
                    "direction_to_road": direction,  # Unit vector pointing toward road
                    "distance_to_road": centroid.distance(nearest_road_point)
                }
            except Exception:
                continue
    
    def _precompute_masks(self):
        """Pre-compute building and road masks for fast lookup."""
        from src.osm import geometry_to_pixel_coords
        
        height, width = self.classification_mask.shape
        
        # Building mask
        self.building_mask = np.zeros((height, width), dtype=np.uint8)
        for _, building in self.buildings.iterrows():
            coords = geometry_to_pixel_coords(
                building.geometry, self.geo_bounds, self.image_size
            )
            if coords and len(coords) >= 3:
                pts = np.array(coords, dtype=np.int32)
                cv2.fillPoly(self.building_mask, [pts], 255)
        
        # Road mask
        self.road_mask = np.zeros((height, width), dtype=np.uint8)
        for _, road in self.roads.iterrows():
            coords = geometry_to_pixel_coords(
                road.geometry, self.geo_bounds, self.image_size
            )
            if coords and len(coords) >= 2:
                pts = np.array(coords, dtype=np.int32)
                cv2.polylines(self.road_mask, [pts], False, 255, thickness=8)
        
        # Driveway mask
        self.driveway_mask = np.zeros((height, width), dtype=np.uint8)
        if self.driveways is not None and not self.driveways.empty:
            for _, driveway in self.driveways.iterrows():
                coords = geometry_to_pixel_coords(
                    driveway.geometry, self.geo_bounds, self.image_size
                )
                if coords and len(coords) >= 2:
                    pts = np.array(coords, dtype=np.int32)
                    cv2.polylines(self.driveway_mask, [pts], False, 255, thickness=6)
    
    def _precompute_distances(self):
        """Pre-compute distance transforms for buildings and roads."""
        # Distance from buildings (in pixels)
        building_binary = self.building_mask > 0
        self.distance_to_building = distance_transform_edt(~building_binary)
        
        # Distance from roads (in pixels)
        road_binary = self.road_mask > 0
        self.distance_to_road = distance_transform_edt(~road_binary)
    
    def _precompute_building_zones(self):
        """
        FAST + LOW MEMORY: Precompute a per-pixel map of 'which building owns this pixel'.
        
        Uses a single labeled mask + one distance_transform_edt call with return_indices.
        Memory: ~3 arrays of image size, NOT per-building arrays.
        
        Result: self.building_owner[y, x] = building_list_index (or -1 if none).
        """
        from src.osm import geometry_to_pixel_coords
        
        height, width = self.classification_mask.shape
        
        # Create a labeled mask: each building gets its own label (idx + 1)
        labeled_mask = np.zeros((height, width), dtype=np.int16)
        self._building_centroids_px = {}
        
        for idx in range(len(self.buildings)):
            building = self.buildings.iloc[idx]
            coords = geometry_to_pixel_coords(building.geometry, self.geo_bounds, self.image_size)
            
            if not coords or len(coords) < 3:
                continue
            
            pts = np.array(coords, dtype=np.int32)
            cv2.fillPoly(labeled_mask, [pts], idx + 1)  # Label = idx + 1 (0 = no building)
            
            # Store centroid
            c = building.geometry.centroid
            cx = int((c.x - self.geo_bounds["west"]) / (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
            cy = int((self.geo_bounds["north"] - c.y) / (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
            self._building_centroids_px[idx] = (cx, cy)
        
        # Single distance transform: for every non-building pixel, find the nearest building pixel
        # return_indices gives us the coordinates of the nearest building pixel
        is_background = (labeled_mask == 0)
        _, nearest_indices = distance_transform_edt(is_background, return_indices=True)
        
        # Look up the building label at the nearest indices
        nearest_labels = labeled_mask[nearest_indices[0], nearest_indices[1]]
        
        # Convert labels back to building indices (label - 1), -1 for no building
        self.building_owner = (nearest_labels - 1).astype(np.int16)
        self.building_owner[nearest_labels == 0] = -1
    
    def _pixel_to_geo(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates."""
        width, height = self.image_size
        lon = self.geo_bounds["west"] + (px / width) * (self.geo_bounds["east"] - self.geo_bounds["west"])
        lat = self.geo_bounds["north"] - (py / height) * (self.geo_bounds["north"] - self.geo_bounds["south"])
        return lat, lon
    
    def _get_surface_type(self, px: int, py: int) -> SurfaceType:
        """
        Determine the surface type at a pixel location.
        
        IMPORTANT: Only returns GRASS if the vegetation mask confirms it.
        Non-vegetation areas get PAVED or DRIVEWAY scores (lower than grass).
        """
        # Check bounds
        height, width = self.classification_mask.shape
        if py < 0 or py >= height or px < 0 or px >= width:
            return SurfaceType.UNKNOWN
        
        if self.building_mask[py, px] > 0:
            return SurfaceType.BUILDING
        if self.road_mask[py, px] > 0:
            return SurfaceType.ROAD
        if self.driveway_mask[py, px] > 0:
            return SurfaceType.DRIVEWAY
        
        # CRITICAL: Only mark as GRASS if vegetation mask confirms it
        # This prevents non-grass front gardens from getting 100 score
        if self.vegetation_mask[py, px] > 0:
            return SurfaceType.GRASS
        
        # Non-vegetation areas - check if it might be a paved driveway/patio
        # by looking at distance to driveway or building
        dist_to_building = self.distance_to_building[py, px] * self.meters_per_pixel
        
        if dist_to_building < 5:
            # Very close to building - might be patio or parking
            return SurfaceType.CAR_PARKING
        
        # Default to paved for non-vegetation areas
        return SurfaceType.PAVED
    
    def _calculate_score(
        self,
        px: int,
        py: int,
        surface_type: SurfaceType
    ) -> float:
        """
        Calculate delivery suitability score for a location.
        
        Scoring:
        - GRASS: 100 base (highest - best delivery surface)
        - DRIVEWAY: 75 base (good - near house, accessible)
        - PAVED: 60 base (okay - patios, concrete areas)
        - CAR_PARKING: 50 base (acceptable - might have cars)
        - UNKNOWN: 30 base (uncertain surface)
        - BUILDING/ROAD: 0 (not suitable)
        
        Args:
            px: Pixel X coordinate
            py: Pixel Y coordinate
            surface_type: Type of surface at this location
            
        Returns:
            Score from 0-100
        """
        height, width = self.classification_mask.shape
        if py < 0 or py >= height or px < 0 or px >= width:
            return 0
        
        # Base score from surface type - GRASS gets 100, others get less
        base_score = self.SURFACE_SCORES.get(surface_type, 30)
        
        # If not grass, the max score is capped at the surface type's base score
        # This ensures non-grass areas NEVER get 100
        if surface_type != SurfaceType.GRASS:
            # Non-grass areas are capped at their base score
            max_possible = base_score
        else:
            max_possible = 100
        
        # Distance to building penalty
        dist_building_px = self.distance_to_building[py, px]
        dist_building_m = dist_building_px * self.meters_per_pixel
        
        distance_penalty = 0
        if dist_building_m < self.OPTIMAL_DISTANCE_MIN:
            # Too close - penalty
            distance_penalty = -20
        elif dist_building_m > self.OPTIMAL_DISTANCE_MAX:
            # Too far - gradual penalty
            excess_m = dist_building_m - self.OPTIMAL_DISTANCE_MAX
            distance_penalty = -min(30, excess_m * 2)  # -2 per meter over, max -30
        
        # Distance to road penalty (don't want to be ON the road)
        dist_road_px = self.distance_to_road[py, px]
        dist_road_m = dist_road_px * self.meters_per_pixel
        
        road_penalty = 0
        if dist_road_m < 2:
            road_penalty = -15  # Too close to road
        
        # Calculate final score
        score = base_score + distance_penalty + road_penalty
        
        # Cap at max possible for this surface type
        score = min(score, max_possible)
        
        return max(0, min(100, score))
    
    def find_best_pin_for_building(
        self,
        building_idx: int,
        garden_type: str = "front"
    ) -> Optional[DeliveryPin]:
        """
        Find the best delivery pin for a specific building.
        
        OPTIMIZED: Uses precomputed building_owner array instead of
        per-pixel Shapely distance calls.
        
        Args:
            building_idx: Index of the building in self.buildings
            garden_type: "front" or "back"
            
        Returns:
            DeliveryPin or None if no suitable location found
        """
        if building_idx >= len(self.buildings):
            return None
        
        # Lazy-init the zone map on first call
        if not hasattr(self, 'building_owner'):
            self._precompute_building_zones()
        
        building = self.buildings.iloc[building_idx]
        building_idx_original = self.buildings.index[building_idx]
        
        # Get centroid
        if building_idx in self._building_centroids_px:
            bld_cx, bld_cy = self._building_centroids_px[building_idx]
        else:
            c = building.geometry.centroid
            w, h = self.image_size
            bld_cx = int((c.x - self.geo_bounds["west"]) / (self.geo_bounds["east"] - self.geo_bounds["west"]) * w)
            bld_cy = int((self.geo_bounds["north"] - c.y) / (self.geo_bounds["north"] - self.geo_bounds["south"]) * h)
        
        target_class = 1 if garden_type == "front" else 2
        height, width = self.classification_mask.shape
        
        # Search radius (15m)
        search_radius_px = int(15 / self.meters_per_pixel)
        min_y = max(0, bld_cy - search_radius_px)
        max_y = min(height, bld_cy + search_radius_px)
        min_x = max(0, bld_cx - search_radius_px)
        max_x = min(width, bld_cx + search_radius_px)
        
        # Extract the search region as numpy slices (FAST)
        region_owner = self.building_owner[min_y:max_y, min_x:max_x]
        region_class = self.classification_mask[min_y:max_y, min_x:max_x]
        region_veg = self.vegetation_mask[min_y:max_y, min_x:max_x]
        region_bld = self.building_mask[min_y:max_y, min_x:max_x]
        region_road = self.road_mask[min_y:max_y, min_x:max_x]
        region_dist = self.distance_to_building[min_y:max_y, min_x:max_x]
        
        # Build candidate mask: owned by this building + correct garden type + not building/road
        candidates = (
            (region_owner == building_idx) &
            (region_class == target_class) &
            (region_bld == 0) &
            (region_road == 0) &
            (region_dist * self.meters_per_pixel <= 12)
        )
        
        # Get direction info for this building
        direction_info = self.building_directions.get(building_idx_original)
        
        if direction_info is not None:
            road_dir = direction_info.get("direction_to_road")
            if road_dir is not None:
                try:
                    rd_x = float(road_dir[0])
                    rd_y = float(road_dir[1])
                    
                    # Create coordinate grids relative to building centroid
                    ys = np.arange(min_y, max_y) - bld_cy
                    xs = np.arange(min_x, max_x) - bld_cx
                    gx, gy = np.meshgrid(xs, ys)
                    
                    lengths = np.sqrt(gx**2 + gy**2)
                    lengths[lengths < 1] = 1  # avoid div by zero
                    
                    # Normalize direction vectors
                    nx = gx / lengths
                    ny = gy / lengths
                    
                    # Dot product with road direction (Y flipped for pixel space)
                    dot = nx * rd_x + ny * (-rd_y)
                    
                    # Directional filter
                    if garden_type == "front":
                        candidates &= (dot > -0.3)
                    else:
                        candidates &= (dot < 0.3)
                        
                        # Lateral alignment: project onto perpendicular axis
                        # and check within building width + tolerance
                        lat_x = -rd_y  # perpendicular
                        lat_y = rd_x
                        lat_dir_px = (lat_x, -lat_y)  # flip Y
                        
                        lateral_proj = gx * lat_dir_px[0] + gy * lat_dir_px[1]
                        
                        # Get building width along lateral axis
                        poly = self._geometry_to_pixel_polygon(building.geometry)
                        if poly is not None:
                            try:
                                coords_list = list(poly.exterior.coords)
                                lat_positions = [
                                    (c[0] - bld_cx) * lat_dir_px[0] + (c[1] - bld_cy) * lat_dir_px[1]
                                    for c in coords_list
                                ]
                                tolerance_px = 5.0 / self.meters_per_pixel
                                min_lat = min(lat_positions) - tolerance_px
                                max_lat = max(lat_positions) + tolerance_px
                                candidates &= (lateral_proj >= min_lat) & (lateral_proj <= max_lat)
                            except Exception:
                                pass
                except (IndexError, TypeError, ValueError):
                    pass
        
        # For grass candidates, filter isolated trees (need connected green area)
        if np.any(candidates & (region_veg > 0)):
            # Prefer grass candidates
            grass_candidates = candidates & (region_veg > 0)
            # Check neighborhood green density
            from scipy.ndimage import uniform_filter
            green_density = uniform_filter((region_veg > 0).astype(np.float32), size=11)
            grass_candidates &= (green_density > 0.15)  # At least 15% green in 11x11 area
            
            if np.any(grass_candidates):
                candidates = grass_candidates
        
        # Find best scoring candidate
        if not np.any(candidates):
            return None
        
        # Score all candidates at once
        candidate_ys, candidate_xs = np.where(candidates)
        
        if len(candidate_ys) == 0:
            return None
        
        # Vectorized scoring: grass=100, driveway=75, else=50
        scores = np.full(len(candidate_ys), 50.0)
        for i, (cy_r, cx_r) in enumerate(zip(candidate_ys, candidate_xs)):
            py_abs = min_y + cy_r
            px_abs = min_x + cx_r
            if self.vegetation_mask[py_abs, px_abs] > 0:
                scores[i] = 100.0
            elif self.driveway_mask[py_abs, px_abs] > 0:
                scores[i] = 75.0
            
            # Distance penalty
            dist_m = self.distance_to_building[py_abs, px_abs] * self.meters_per_pixel
            if dist_m < 2:
                scores[i] -= 20
            elif dist_m > 15:
                scores[i] -= min(30, (dist_m - 15) * 2)
        
        best_i = np.argmax(scores)
        best_py = min_y + candidate_ys[best_i]
        best_px = min_x + candidate_xs[best_i]
        best_score = scores[best_i]
        
        if best_score <= 0:
            return None
        
        # Determine surface type
        if self.vegetation_mask[best_py, best_px] > 0:
            surface = "grass"
        elif self.driveway_mask[best_py, best_px] > 0:
            surface = "driveway"
        else:
            surface = "paved"
        
        lat, lon = self._pixel_to_geo(best_px, best_py)
        dist_m = self.distance_to_building[best_py, best_px] * self.meters_per_pixel
        
        building_id = None
        if "osm_id" in building.index:
            building_id = str(building["osm_id"])
        elif hasattr(building, 'name') and building.name is not None:
            building_id = str(building.name)
        
        return DeliveryPin(
            lat=lat, lon=lon,
            garden_type=garden_type,
            score=round(max(0, min(100, best_score)), 2),
            surface_type=surface,
            distance_to_building_m=round(dist_m, 2),
            building_id=building_id,
            metadata={"pixel_x": best_px, "pixel_y": best_py}
        )
    
    def find_all_pins(
        self,
        include_front: bool = True,
        include_back: bool = True,
        min_score: float = 20.0,
        show_progress: bool = True
    ) -> List[DeliveryPin]:
        """
        Find delivery pins for all buildings.
        
        Args:
            include_front: Include front garden pins
            include_back: Include back garden pins
            min_score: Minimum score threshold
            show_progress: Show tqdm progress bar
            
        Returns:
            List of DeliveryPin objects
        """
        from tqdm import tqdm
        
        # Precompute building zones once (fast numpy operation)
        if not hasattr(self, 'building_owner'):
            self._precompute_building_zones()
        
        pins = []
        iterator = range(len(self.buildings))
        if show_progress:
            iterator = tqdm(iterator, desc="Finding pins", unit="bld")
        
        for idx in iterator:
            if include_front:
                front_pin = self.find_best_pin_for_building(idx, "front")
                if front_pin and front_pin.score >= min_score:
                    pins.append(front_pin)
            
            if include_back:
                back_pin = self.find_best_pin_for_building(idx, "back")
                if back_pin and back_pin.score >= min_score:
                    pins.append(back_pin)
        
        return pins
    
    def classify_point(self, lat: float, lon: float) -> dict:
        """
        Classify a single GPS coordinate as front or back garden.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dict with classification result
        """
        width, height = self.image_size
        
        # Convert to pixel coordinates
        px = int((lon - self.geo_bounds["west"]) / 
                (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
        py = int((self.geo_bounds["north"] - lat) / 
                (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
        
        # Check bounds
        if px < 0 or px >= width or py < 0 or py >= height:
            return {
                "lat": lat,
                "lon": lon,
                "classification": "out_of_bounds",
                "confidence": 0.0
            }
        
        # Get classification
        class_value = self.classification_mask[py, px]
        
        if class_value == 1:
            classification = "front_garden"
        elif class_value == 2:
            classification = "back_garden"
        elif class_value == 3:
            classification = "unknown"
        else:
            # Check if it's a building, road, etc.
            surface_type = self._get_surface_type(px, py)
            classification = surface_type.name.lower()
        
        # Get surface type and score
        surface_type = self._get_surface_type(px, py)
        score = self._calculate_score(px, py, surface_type)
        
        return {
            "lat": lat,
            "lon": lon,
            "classification": classification,
            "surface_type": surface_type.name.lower(),
            "score": round(score, 2),
            "distance_to_building_m": round(
                self.distance_to_building[py, px] * self.meters_per_pixel, 2
            )
        }
    
    def find_nearest_building_pins(
        self,
        lat: float,
        lon: float,
        max_distance_m: float = 50.0
    ) -> Dict[str, Optional[DeliveryPin]]:
        """
        Find the best front and back garden pins for the building at/nearest to a coordinate.
        
        IMPROVED: First checks if the point is INSIDE or very close to a building.
        Only searches for gardens that are directly adjacent to THAT building.
        
        Args:
            lat: Latitude
            lon: Longitude
            max_distance_m: Maximum distance to search for buildings
            
        Returns:
            Dict with 'front' and 'back' DeliveryPin objects (or None if not found)
        """
        if self.buildings.empty:
            return {"front": None, "back": None}
        
        # Convert input point to pixel coordinates
        width, height = self.image_size
        input_px = int((lon - self.geo_bounds["west"]) / 
                      (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
        input_py = int((self.geo_bounds["north"] - lat) / 
                      (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
        
        # Project buildings to meters for distance calculation
        buildings_m = project_to_meters(self.buildings, self.center_lat, self.center_lon)
        
        # Create point and project it
        point = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        )
        point_m = project_to_meters(point, self.center_lat, self.center_lon)
        point_geom = point_m.geometry.iloc[0]
        
        # First: Check if point is INSIDE a building
        containing_idx = None
        for idx, building in buildings_m.iterrows():
            if building.geometry.contains(point_geom) or building.geometry.distance(point_geom) < 3:
                containing_idx = idx
                break
        
        # If point is inside a building, use that building
        if containing_idx is not None:
            nearest_idx = containing_idx
            min_dist = 0
        else:
            # Find nearest building
            min_dist = float("inf")
            nearest_idx = None
            
            for idx, building in buildings_m.iterrows():
                dist = point_geom.distance(building.geometry)
                if dist < min_dist and dist < max_distance_m:
                    min_dist = dist
                    nearest_idx = idx
        
        if nearest_idx is None:
            return {"front": None, "back": None}
        
        # Get the building in WGS84
        building_wgs = self.buildings.loc[nearest_idx]
        building_centroid = building_wgs.geometry.centroid
        
        # Convert building centroid to pixel coordinates
        bld_cx = int((building_centroid.x - self.geo_bounds["west"]) / 
                    (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
        bld_cy = int((self.geo_bounds["north"] - building_centroid.y) / 
                    (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
        
        # Find the index in the original DataFrame
        building_list_idx = list(self.buildings.index).index(nearest_idx)
        
        # Find front and back pins for THIS specific building
        front_pin = self._find_pin_for_specific_building(
            building_list_idx, bld_cx, bld_cy, input_px, input_py, "front"
        )
        back_pin = self._find_pin_for_specific_building(
            building_list_idx, bld_cx, bld_cy, input_px, input_py, "back"
        )
        
        return {
            "front": front_pin,
            "back": back_pin
        }
    
    def _find_pin_for_specific_building(
        self,
        building_idx: int,
        bld_cx: int,
        bld_cy: int,
        input_px: int,
        input_py: int,
        garden_type: str
    ) -> Optional[DeliveryPin]:
        """
        Find a pin for a specific building, ensuring it's in the correct garden.
        
        Key improvements:
        1. Creates a Voronoi-style zone for this building
        2. Only considers pixels closest to THIS building
        3. Filters out isolated green (likely trees) - must be connected to building-adjacent area
        4. Requires green area to be contiguous with building edge
        """
        if building_idx >= len(self.buildings):
            return None
        
        building = self.buildings.iloc[building_idx]
        target_class = 1 if garden_type == "front" else 2
        
        height, width = self.classification_mask.shape
        
        # Build a map of "closest building" for each pixel (Voronoi-style)
        building_zone_mask = self._get_building_zone_mask(building_idx, bld_cx, bld_cy)
        
        # Search radius in pixels (about 15 meters - tighter to avoid neighbor gardens)
        search_radius_px = int(15 / self.meters_per_pixel)
        
        min_y = max(0, bld_cy - search_radius_px)
        max_y = min(height, bld_cy + search_radius_px)
        min_x = max(0, bld_cx - search_radius_px)
        max_x = min(width, bld_cx + search_radius_px)
        
        best_score = -1
        best_px, best_py = None, None
        best_surface = None
        
        # Get target building polygon for additional checks
        target_building_poly = self._geometry_to_pixel_polygon(building.geometry)
        
        # Get our building's direction info
        building_idx_original = self.buildings.index[building_idx]
        our_direction_info = self.building_directions.get(building_idx_original)
        
        # Sample every 2 pixels for better accuracy
        for py in range(min_y, max_y, 2):
            for px in range(min_x, max_x, 2):
                # CRITICAL: Must be in THIS building's zone
                if not building_zone_mask[py, px]:
                    continue
                
                # Must be correct garden type
                if self.classification_mask[py, px] != target_class:
                    continue
                
                # DIRECTIONAL CHECK: Verify pixel is on correct side of building
                if our_direction_info and not self._is_correct_side(
                    px, py, bld_cx, bld_cy, our_direction_info, garden_type
                ):
                    continue
                
                # LATERAL ALIGNMENT CHECK: For back gardens, pixel must be directly
                # behind the building, not shifted sideways into neighbor's garden
                if garden_type == "back" and not self._is_laterally_aligned(
                    px, py, building_idx, garden_type
                ):
                    continue
                
                # NEIGHBOR CHECK: Ensure this isn't in front of another building
                if self._is_front_of_other_building(px, py, building_idx_original, garden_type):
                    continue
                
                # Distance to our building in meters (using the distance transform)
                dist_to_any_building_m = self.distance_to_building[py, px] * self.meters_per_pixel
                
                # Also check distance to OUR specific building
                if target_building_poly:
                    point = Point(px, py)
                    dist_to_our_building_px = point.distance(target_building_poly)
                    dist_to_our_building_m = dist_to_our_building_px * self.meters_per_pixel
                else:
                    dist_to_our_building_m = dist_to_any_building_m
                
                # Skip if too far from OUR building (stricter - 10m max)
                if dist_to_our_building_m > 10:
                    continue
                
                # Get surface type
                surface_type = self._get_surface_type(px, py)
                
                # Skip buildings and roads
                if surface_type in (SurfaceType.BUILDING, SurfaceType.ROAD):
                    continue
                
                # For grass, verify it's a substantial area (not an isolated tree)
                if surface_type == SurfaceType.GRASS:
                    if not self._is_valid_grass_area(px, py):
                        continue
                
                # Calculate score
                score = self._calculate_score(px, py, surface_type)
                
                if score > best_score:
                    best_score = score
                    best_px, best_py = px, py
                    best_surface = surface_type
        
        if best_px is None:
            # Try fallback with driveway/paved (non-grass) in this zone
            return self._find_non_grass_pin(building_idx, bld_cx, bld_cy, building_zone_mask, garden_type)
        
        # Convert to geographic coordinates
        lat, lon = self._pixel_to_geo(best_px, best_py)
        
        # Calculate distance to building
        dist_m = self.distance_to_building[best_py, best_px] * self.meters_per_pixel
        
        # Get building ID
        building_id = None
        if "osm_id" in building.index:
            building_id = str(building["osm_id"])
        elif hasattr(building, 'name') and building.name is not None:
            building_id = str(building.name)
        
        return DeliveryPin(
            lat=lat,
            lon=lon,
            garden_type=garden_type,
            score=best_score,
            surface_type=best_surface.name.lower(),
            distance_to_building_m=dist_m,
            building_id=building_id,
            metadata={
                "pixel_x": best_px,
                "pixel_y": best_py,
            }
        )
    
    def _get_building_zone_mask(self, target_idx: int, target_cx: int, target_cy: int) -> np.ndarray:
        """
        Create a mask where pixels are True ONLY if they're in the target building's garden zone.
        
        Approach:
        1. Pixel must be closer to target building than ANY other building
        2. Pixel must NOT be between another building and its likely garden area
        3. Uses building edges, not just centroids, for accurate distance calculation
        """
        height, width = self.classification_mask.shape
        mask = np.zeros((height, width), dtype=bool)  # Start with all False
        
        # Get target building geometry in pixel coords
        target_building = self.buildings.iloc[target_idx]
        target_poly_px = self._geometry_to_pixel_polygon(target_building.geometry)
        
        # Get all other building polygons in pixel coords
        other_buildings_px = []
        for idx, bld in self.buildings.iterrows():
            if list(self.buildings.index).index(idx) == target_idx:
                continue
            poly_px = self._geometry_to_pixel_polygon(bld.geometry)
            if poly_px is not None:
                c = bld.geometry.centroid
                cx = int((c.x - self.geo_bounds["west"]) / 
                        (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
                cy = int((self.geo_bounds["north"] - c.y) / 
                        (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
                other_buildings_px.append((poly_px, cx, cy))
        
        # Search radius - only consider pixels close to our building
        search_radius = int(20 / self.meters_per_pixel)
        min_y = max(0, target_cy - search_radius)
        max_y = min(height, target_cy + search_radius)
        min_x = max(0, target_cx - search_radius)
        max_x = min(width, target_cx + search_radius)
        
        for py in range(min_y, max_y):
            for px in range(min_x, max_x):
                # Distance to target building EDGE (not centroid)
                point = Point(px, py)
                dist_to_target = point.distance(target_poly_px) if target_poly_px else float('inf')
                
                # Check if this pixel is closer to our building than others
                is_closest_to_target = True
                for other_poly, other_cx, other_cy in other_buildings_px:
                    dist_to_other = point.distance(other_poly)
                    
                    # Must be significantly closer to target (1.5x rule)
                    # This prevents selecting areas that are ambiguously between buildings
                    if dist_to_other < dist_to_target * 1.5:
                        is_closest_to_target = False
                        break
                
                if is_closest_to_target:
                    mask[py, px] = True
        
        return mask
    
    def _geometry_to_pixel_polygon(self, geom) -> Optional[Polygon]:
        """Convert a shapely geometry to pixel coordinates."""
        from shapely.geometry import Polygon, MultiPolygon
        
        width, height = self.image_size
        
        def transform_coords(coords):
            result = []
            for lon, lat in coords:
                px = (lon - self.geo_bounds["west"]) / (self.geo_bounds["east"] - self.geo_bounds["west"]) * width
                py = (self.geo_bounds["north"] - lat) / (self.geo_bounds["north"] - self.geo_bounds["south"]) * height
                result.append((px, py))
            return result
        
        try:
            if geom.geom_type == 'Polygon':
                return Polygon(transform_coords(geom.exterior.coords))
            elif geom.geom_type == 'MultiPolygon':
                # Return the largest polygon
                largest = max(geom.geoms, key=lambda p: p.area)
                return Polygon(transform_coords(largest.exterior.coords))
            else:
                return None
        except Exception:
            return None
    
    def _is_valid_grass_area(self, px: int, py: int, min_connected_pixels: int = 50) -> bool:
        """
        Check if a grass pixel is part of a substantial grass area.
        
        Filters out isolated trees which appear as small circular green blobs.
        Valid grass should be:
        - Part of a larger contiguous area
        - Connected to building-adjacent areas
        """
        height, width = self.vegetation_mask.shape
        
        # Quick check: count vegetation pixels in small neighborhood
        radius = 5
        min_y = max(0, py - radius)
        max_y = min(height, py + radius)
        min_x = max(0, px - radius)
        max_x = min(width, px + radius)
        
        neighborhood = self.vegetation_mask[min_y:max_y, min_x:max_x]
        green_count = np.sum(neighborhood > 0)
        
        # If very few green pixels nearby, likely isolated tree
        if green_count < 20:
            return False
        
        # Check if building is nearby (grass should be adjacent to a building)
        building_dist = self.distance_to_building[py, px] * self.meters_per_pixel
        if building_dist > 15:
            return False
        
        return True
    
    def _is_correct_side(
        self,
        px: int,
        py: int,
        bld_cx: int,
        bld_cy: int,
        direction_info: Dict,
        garden_type: str
    ) -> bool:
        """
        Check if pixel is on the correct side of the building for the garden type.
        
        - Front garden: pixel should be in the direction of the road
        - Back garden: pixel should be OPPOSITE to the road direction
        """
        road_dir = direction_info.get("direction_to_road")
        if road_dir is None:
            return True  # No direction info, allow it
        
        # Ensure road_dir is indexable (could be tuple, list, or numpy array)
        try:
            rd_x = float(road_dir[0])
            rd_y = float(road_dir[1])
        except (IndexError, TypeError, ValueError):
            return True
        
        # Vector from building centroid to pixel
        dx = px - bld_cx
        dy = py - bld_cy
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 1:
            return True  # Too close to centroid, allow it
        
        # Normalize
        px_dir_x = dx / length
        px_dir_y = dy / length
        
        # Dot product with road direction
        # Note: road_dir is in meters space, px_dir is in pixel space
        # In pixel space, Y is inverted (increases downward), so we negate dy for road_dir
        dot = px_dir_x * rd_x + px_dir_y * (-rd_y)  # Flip Y for pixel coordinates
        
        if garden_type == "front":
            # Front garden should be toward the road (positive dot product)
            return dot > -0.3  # Allow some tolerance
        else:
            # Back garden should be away from the road (negative dot product)
            return dot < 0.3  # Allow some tolerance
    
    def _is_front_of_other_building(
        self,
        px: int,
        py: int,
        our_building_idx,
        garden_type: str
    ) -> bool:
        """
        Check if this pixel is in the FRONT garden zone of another building.
        
        If looking for a back garden pin, we should reject pixels that are
        actually in front of a neighboring building (their front garden).
        """
        if garden_type != "back":
            return False  # Only check for back gardens
        
        width, height = self.image_size
        
        for idx, bld in self.buildings.iterrows():
            if idx == our_building_idx:
                continue
            
            direction_info = self.building_directions.get(idx)
            if not direction_info:
                continue
            
            # Get this building's centroid in pixel coords
            other_cx, other_cy = direction_info.get("centroid_px", (None, None))
            if other_cx is None:
                c = bld.geometry.centroid
                other_cx = int((c.x - self.geo_bounds["west"]) / 
                              (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
                other_cy = int((self.geo_bounds["north"] - c.y) / 
                              (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
            
            # Distance from pixel to this building
            dist_to_other = np.sqrt((px - other_cx)**2 + (py - other_cy)**2)
            dist_to_other_m = dist_to_other * self.meters_per_pixel
            
            # Only check buildings within 15m
            if dist_to_other_m > 15:
                continue
            
            # Check if pixel is in FRONT of this building
            road_dir = direction_info.get("direction_to_road")
            if road_dir is None:
                continue
            
            try:
                rd_x = float(road_dir[0])
                rd_y = float(road_dir[1])
            except (IndexError, TypeError, ValueError):
                continue
            
            # Vector from other building to pixel
            dx = px - other_cx
            dy = py - other_cy
            length = np.sqrt(dx**2 + dy**2)
            
            if length < 1:
                continue
            
            px_dir_x = dx / length
            px_dir_y = dy / length
            
            dot = px_dir_x * rd_x + px_dir_y * (-rd_y)  # Flip Y for pixel coords
            
            # If pixel is in front of this other building (positive dot product)
            # and close to it, reject it - it's the neighbor's front garden
            if dot > 0.5 and dist_to_other_m < 12:
                return True  # This is in front of another building
        
        return False
    
    def _is_laterally_aligned(
        self,
        px: int,
        py: int,
        building_idx: int,
        garden_type: str,
        tolerance_m: float = 5.0
    ) -> bool:
        """
        Check if a pixel is laterally aligned with the building.
        
        For back gardens in terraced/semi-detached houses, the garden should be
        directly behind the building, not shifted sideways into neighbor's space.
        
        We project the building's width perpendicular to the road direction,
        and check if the pixel falls within that corridor (plus tolerance).
        """
        if building_idx >= len(self.buildings):
            return True
        
        building = self.buildings.iloc[building_idx]
        building_idx_original = self.buildings.index[building_idx]
        direction_info = self.building_directions.get(building_idx_original)
        
        if not direction_info:
            return True  # No direction info, allow it
        
        road_dir = direction_info.get("direction_to_road")
        if road_dir is None:
            return True
        
        try:
            rd_x = float(road_dir[0])
            rd_y = float(road_dir[1])
        except (IndexError, TypeError, ValueError):
            return True
        
        # Get building bounds in pixel coords
        building_poly = self._geometry_to_pixel_polygon(building.geometry)
        if building_poly is None:
            return True
        
        # Get building centroid
        bld_cx, bld_cy = direction_info.get("centroid_px", (None, None))
        if bld_cx is None:
            bounds = building_poly.bounds
            bld_cx = (bounds[0] + bounds[2]) / 2
            bld_cy = (bounds[1] + bounds[3]) / 2
        
        # The "lateral" direction is perpendicular to the road direction
        # road_dir is (dx, dy) pointing toward road
        # lateral is (-dy, dx) or (dy, -dx)
        lateral_dir = (-rd_y, rd_x)  # Perpendicular in meters space
        lateral_dir_px = (lateral_dir[0], -lateral_dir[1])  # Flip Y for pixel space
        
        # Calculate the building's width along the lateral axis
        # by projecting building corners onto the lateral axis
        try:
            coords = list(building_poly.exterior.coords)
            lateral_positions = []
            for coord in coords:
                # Vector from centroid to corner
                dx = coord[0] - bld_cx
                dy = coord[1] - bld_cy
                # Project onto lateral axis
                proj = dx * lateral_dir_px[0] + dy * lateral_dir_px[1]
                lateral_positions.append(proj)
            
            min_lateral = min(lateral_positions)
            max_lateral = max(lateral_positions)
            
            # Add tolerance
            tolerance_px = tolerance_m / self.meters_per_pixel
            min_lateral -= tolerance_px
            max_lateral += tolerance_px
            
            # Project the pixel onto the lateral axis
            dx = px - bld_cx
            dy = py - bld_cy
            pixel_lateral = dx * lateral_dir_px[0] + dy * lateral_dir_px[1]
            
            # Check if pixel is within the building's lateral corridor
            return min_lateral <= pixel_lateral <= max_lateral
            
        except Exception:
            return True  # On error, allow it
    
    def _find_non_grass_pin(
        self,
        building_idx: int,
        bld_cx: int,
        bld_cy: int,
        zone_mask: np.ndarray,
        garden_type: str
    ) -> Optional[DeliveryPin]:
        """
        Fallback: find a driveway or paved area pin when no grass is available.
        """
        target_class = 1 if garden_type == "front" else 2
        height, width = self.classification_mask.shape
        
        search_radius_px = int(12 / self.meters_per_pixel)
        
        min_y = max(0, bld_cy - search_radius_px)
        max_y = min(height, bld_cy + search_radius_px)
        min_x = max(0, bld_cx - search_radius_px)
        max_x = min(width, bld_cx + search_radius_px)
        
        best_score = -1
        best_px, best_py = None, None
        best_surface = None
        
        for py in range(min_y, max_y, 2):
            for px in range(min_x, max_x, 2):
                if not zone_mask[py, px]:
                    continue
                if self.classification_mask[py, px] != target_class:
                    continue
                
                # Lateral alignment check for back gardens
                if garden_type == "back" and not self._is_laterally_aligned(
                    px, py, building_idx, garden_type
                ):
                    continue
                
                surface_type = self._get_surface_type(px, py)
                
                # Accept driveway, paved, or car parking
                if surface_type in (SurfaceType.BUILDING, SurfaceType.ROAD, SurfaceType.UNKNOWN):
                    continue
                
                score = self._calculate_score(px, py, surface_type)
                
                if score > best_score:
                    best_score = score
                    best_px, best_py = px, py
                    best_surface = surface_type
        
        if best_px is None:
            return None
        
        building = self.buildings.iloc[building_idx]
        lat, lon = self._pixel_to_geo(best_px, best_py)
        dist_m = self.distance_to_building[best_py, best_px] * self.meters_per_pixel
        
        building_id = None
        if "osm_id" in building.index:
            building_id = str(building["osm_id"])
        elif hasattr(building, 'name') and building.name is not None:
            building_id = str(building.name)
        
        return DeliveryPin(
            lat=lat,
            lon=lon,
            garden_type=garden_type,
            score=best_score,
            surface_type=best_surface.name.lower(),
            distance_to_building_m=dist_m,
            building_id=building_id,
            metadata={"pixel_x": best_px, "pixel_y": best_py}
        )


def find_delivery_pins_for_area(
    classification_mask: np.ndarray,
    vegetation_mask: np.ndarray,
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    driveways: gpd.GeoDataFrame,
    geo_bounds: dict,
    image_size: Tuple[int, int],
    center_lat: float,
    center_lon: float,
    min_score: float = 20.0
) -> List[DeliveryPin]:
    """
    Convenience function to find all delivery pins for an area.
    
    Args:
        classification_mask: Classification mask
        vegetation_mask: Vegetation mask
        buildings: Buildings GeoDataFrame
        roads: Roads GeoDataFrame
        driveways: Driveways GeoDataFrame
        geo_bounds: Geographic bounds
        image_size: Image size
        center_lat: Center latitude
        center_lon: Center longitude
        min_score: Minimum score threshold
        
    Returns:
        List of DeliveryPin objects
    """
    finder = DeliveryPinFinder(
        classification_mask=classification_mask,
        vegetation_mask=vegetation_mask,
        buildings=buildings,
        roads=roads,
        driveways=driveways,
        geo_bounds=geo_bounds,
        image_size=image_size,
        center_lat=center_lat,
        center_lon=center_lon
    )
    
    return finder.find_all_pins(min_score=min_score)
