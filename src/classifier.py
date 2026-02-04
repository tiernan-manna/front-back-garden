"""
Front/Back Garden Classifier module

Classifies detected garden areas as front or back gardens based on
their geometric relationship to buildings and roads.

Logic:
- Front garden: Between a building and the nearest road
- Back garden: On the opposite side of the building from the nearest road

Optimized with:
- KDTree for fast nearest-neighbor lookups
- Vectorized numpy operations
- Pre-computed direction maps
"""

import cv2
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
from typing import Tuple, List, Dict, Optional
import warnings

from src.osm import project_to_meters, geo_to_pixel


class GardenClassifier:
    """
    Classifies garden pixels as front or back based on building/road geometry.
    
    Improved approach:
    1. Uses driveways when available (best indicator of "front")
    2. Falls back to nearest road direction
    3. Excludes parks, pitches, and other public spaces
    4. Only considers main buildings (not sheds/garages)
    """
    
    # Classification labels
    NOT_GARDEN = 0      # Not a garden (buildings, roads, etc.)
    FRONT_GARDEN = 1    # Front garden (facing road)
    BACK_GARDEN = 2     # Back garden (away from road)
    UNKNOWN = 3         # Unknown/ambiguous (will show as red)
    
    def __init__(
        self,
        buildings: gpd.GeoDataFrame,
        roads: gpd.GeoDataFrame,
        geo_bounds: dict,
        image_size: Tuple[int, int],
        center_lat: float,
        center_lon: float,
        driveways: gpd.GeoDataFrame = None,
        exclusion_zones: gpd.GeoDataFrame = None,
        property_boundaries: gpd.GeoDataFrame = None,
        address_polygons: gpd.GeoDataFrame = None
    ):
        """
        Initialize the classifier.
        
        Args:
            buildings: GeoDataFrame with building polygons (EPSG:4326)
            roads: GeoDataFrame with road geometries (EPSG:4326)
            geo_bounds: Dict with 'north', 'south', 'east', 'west'
            image_size: Tuple of (width, height) in pixels
            center_lat: Center latitude for CRS projection
            center_lon: Center longitude for CRS projection
            driveways: GeoDataFrame with driveway LineStrings (optional but recommended)
            exclusion_zones: GeoDataFrame with areas to exclude (parks, pitches, etc.)
            property_boundaries: GeoDataFrame with walls/fences (property separators)
            address_polygons: GeoDataFrame with building footprints + street names
        """
        self.geo_bounds = geo_bounds
        self.image_size = image_size
        self.center_lat = center_lat
        self.center_lon = center_lon
        
        # Project to meters for accurate distance calculations
        self.buildings_m = project_to_meters(buildings, center_lat, center_lon)
        self.roads_m = project_to_meters(roads, center_lat, center_lon)
        
        # Handle driveways
        if driveways is not None and not driveways.empty:
            self.driveways_m = project_to_meters(driveways, center_lat, center_lon)
        else:
            self.driveways_m = gpd.GeoDataFrame(geometry=[], crs=self.buildings_m.crs)
        
        # Handle exclusion zones
        if exclusion_zones is not None and not exclusion_zones.empty:
            self.exclusion_zones_m = project_to_meters(exclusion_zones, center_lat, center_lon)
            self.exclusion_union = self.exclusion_zones_m.geometry.unary_union
        else:
            self.exclusion_zones_m = gpd.GeoDataFrame(geometry=[], crs=self.buildings_m.crs)
            self.exclusion_union = None
        
        # Handle property boundaries (walls, fences)
        if property_boundaries is not None and not property_boundaries.empty:
            self.property_boundaries_wgs = property_boundaries
        else:
            self.property_boundaries_wgs = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Handle address polygons (building footprints with street info)
        if address_polygons is not None and not address_polygons.empty:
            self.address_polygons_wgs = address_polygons
            self.address_polygons_m = project_to_meters(address_polygons, center_lat, center_lon)
        else:
            self.address_polygons_wgs = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            self.address_polygons_m = gpd.GeoDataFrame(geometry=[], crs=self.buildings_m.crs if not self.buildings_m.empty else "EPSG:4326")
        
        # Keep original WGS84 for pixel conversion
        self.buildings_wgs = buildings
        self.roads_wgs = roads
        
        # Pre-compute building centroids and road-facing directions
        self._precompute_building_directions()
    
    def _precompute_building_directions(self):
        """
        For each building, compute the direction to the front.
        
        IMPROVED: Now also stores building polygon for edge-based classification.
        
        Priority:
        1. Use address street name to find the correct road (best!)
        2. Use driveway direction if a driveway is near the building
        3. Fall back to nearest road direction
        """
        from tqdm import tqdm
        
        self.building_directions = {}
        self.building_polygons_px = {}  # Store building polygons in pixel coords
        
        if self.buildings_m.empty or self.roads_m.empty:
            return
        
        # Merge all roads into a single geometry for distance calculations
        all_roads = self.roads_m.geometry.unary_union
        
        # Also merge driveways if available
        has_driveways = not self.driveways_m.empty
        if has_driveways:
            all_driveways = self.driveways_m.geometry.unary_union
        
        # Build a lookup from street name to road geometry
        has_addresses = not self.address_polygons_wgs.empty and "addr:street" in self.address_polygons_wgs.columns
        street_roads = {}
        if "name" in self.roads_wgs.columns:
            for idx, road in self.roads_wgs.iterrows():
                try:
                    # Safely get the name column value
                    name = None
                    if hasattr(road, 'name') and not callable(getattr(road, 'name', None)):
                        name = road['name'] if 'name' in road.index else None
                    
                    if name and isinstance(name, str) and len(name) > 1:
                        # Normalize street name (strip, lowercase for matching)
                        name_lower = name.strip().lower()
                        if name_lower not in street_roads:
                            street_roads[name_lower] = []
                        # Get the projected version for distance calculations
                        if idx in self.roads_m.index:
                            street_roads[name_lower].append(self.roads_m.loc[idx].geometry)
                except Exception:
                    pass
        
        if street_roads:
            print(f"    Built street lookup with {len(street_roads)} named streets")
        
        address_count = 0
        driveway_count = 0
        road_count = 0
        
        width, height = self.image_size
        
        for idx, building in tqdm(self.buildings_m.iterrows(), 
                                   total=len(self.buildings_m), 
                                   desc="Processing buildings"):
            centroid = building.geometry.centroid
            
            # Store building polygon in pixel coordinates for edge-based classification
            if idx in self.buildings_wgs.index:
                building_wgs = self.buildings_wgs.loc[idx]
                try:
                    if building_wgs.geometry.geom_type == 'Polygon':
                        coords = list(building_wgs.geometry.exterior.coords)
                        pixel_coords = []
                        for lon, lat in coords:
                            px = int((lon - self.geo_bounds["west"]) / 
                                    (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
                            py = int((self.geo_bounds["north"] - lat) / 
                                    (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
                            pixel_coords.append((px, py))
                        self.building_polygons_px[idx] = pixel_coords
                except Exception:
                    pass
            
            try:
                direction = None
                source = "road"
                
                # Priority 1: Use address street name to find the correct road
                if has_addresses and street_roads:
                    try:
                        # Check if there's an address polygon that matches this building
                        for addr_idx, addr in self.address_polygons_m.iterrows():
                            if building.geometry.intersects(addr.geometry) or building.geometry.distance(addr.geometry) < 5:
                                if addr_idx in self.address_polygons_wgs.index:
                                    addr_wgs = self.address_polygons_wgs.loc[addr_idx]
                                    
                                    # Safely get street name
                                    street_name = None
                                    if "addr:street" in addr_wgs.index:
                                        street_name = addr_wgs["addr:street"]
                                    
                                    if street_name and isinstance(street_name, str):
                                        # Normalize for matching
                                        street_name_lower = street_name.strip().lower()
                                        
                                        if street_name_lower in street_roads:
                                            # Find direction to the named street
                                            street_geoms = street_roads[street_name_lower]
                                            min_dist = float("inf")
                                            best_point = None
                                            
                                            for geom in street_geoms:
                                                try:
                                                    point = nearest_points(centroid, geom)[1]
                                                    dist = centroid.distance(point)
                                                    if dist < min_dist:
                                                        min_dist = dist
                                                        best_point = point
                                                except Exception:
                                                    pass
                                            
                                            if best_point is not None and min_dist < 150:  # Increased distance
                                                dx = best_point.x - centroid.x
                                                dy = best_point.y - centroid.y
                                                length = np.sqrt(dx**2 + dy**2)
                                                
                                                if length > 0.1:
                                                    direction = np.array([dx / length, dy / length])
                                                    source = "address"
                                                    address_count += 1
                                break
                    except Exception as e:
                        pass
                
                # Priority 2: Try to find a driveway near this building
                if direction is None and has_driveways:
                    try:
                        nearest_driveway_point = nearest_points(centroid, all_driveways)[1]
                        driveway_dist = centroid.distance(nearest_driveway_point)
                        
                        if driveway_dist < 30:
                            dx = nearest_driveway_point.x - centroid.x
                            dy = nearest_driveway_point.y - centroid.y
                            length = np.sqrt(dx**2 + dy**2)
                            
                            if length > 0.1:
                                direction = np.array([dx / length, dy / length])
                                source = "driveway"
                                driveway_count += 1
                    except Exception:
                        pass
                
                # Priority 3: Fall back to nearest road
                if direction is None:
                    nearest_road_point = nearest_points(centroid, all_roads)[1]
                    
                    dx = nearest_road_point.x - centroid.x
                    dy = nearest_road_point.y - centroid.y
                    length = np.sqrt(dx**2 + dy**2)
                    
                    if length > 0.1:
                        direction = np.array([dx / length, dy / length])
                        road_count += 1
                    else:
                        direction = np.array([0, 0])
                
                # Compute distance to nearest road for reference
                nearest_road_point = nearest_points(centroid, all_roads)[1]
                distance = centroid.distance(nearest_road_point)
                
                self.building_directions[idx] = {
                    "centroid": centroid,
                    "direction_to_road": direction,
                    "distance_to_road": distance,
                    "source": source,
                }
                
            except Exception as e:
                warnings.warn(f"Could not compute direction for building {idx}: {e}")
        
        print(f"    Front direction: {address_count} from address, {driveway_count} from driveways, {road_count} from roads")
    
    def classify_pixel(
        self,
        px: int,
        py: int
    ) -> int:
        """
        Classify a single pixel as front garden, back garden, or unknown.
        
        Args:
            px: Pixel X coordinate
            py: Pixel Y coordinate
            
        Returns:
            Classification label (FRONT_GARDEN, BACK_GARDEN, or UNKNOWN)
        """
        if self.buildings_m.empty or not self.building_directions:
            return self.UNKNOWN
        
        # Convert pixel to geographic coordinates
        width, height = self.image_size
        lon = self.geo_bounds["west"] + (px / width) * (self.geo_bounds["east"] - self.geo_bounds["west"])
        lat = self.geo_bounds["north"] - (py / height) * (self.geo_bounds["north"] - self.geo_bounds["south"])
        
        # Convert to projected coordinates
        point_wgs = Point(lon, lat)
        
        # Create a temporary GeoDataFrame to project the point
        temp_gdf = gpd.GeoDataFrame(geometry=[point_wgs], crs="EPSG:4326")
        temp_gdf_m = project_to_meters(temp_gdf, self.center_lat, self.center_lon)
        point_m = temp_gdf_m.geometry.iloc[0]
        
        # Find nearest building
        min_dist = float("inf")
        nearest_building_idx = None
        
        for idx, building in self.buildings_m.iterrows():
            dist = point_m.distance(building.geometry)
            if dist < min_dist:
                min_dist = dist
                nearest_building_idx = idx
        
        if nearest_building_idx is None or min_dist > 50:  # Max 50m from building
            return self.UNKNOWN
        
        # Get the building's road-facing direction
        if nearest_building_idx not in self.building_directions:
            return self.UNKNOWN
        
        building_info = self.building_directions[nearest_building_idx]
        centroid = building_info["centroid"]
        road_direction = building_info["direction_to_road"]
        
        if np.linalg.norm(road_direction) < 0.1:
            return self.UNKNOWN
        
        # Vector from building centroid to this pixel
        pixel_direction = np.array([
            point_m.x - centroid.x,
            point_m.y - centroid.y
        ])
        
        pixel_dist = np.linalg.norm(pixel_direction)
        if pixel_dist < 0.1:
            return self.UNKNOWN
        
        pixel_direction = pixel_direction / pixel_dist
        
        # Dot product: positive = same direction as road (front), negative = opposite (back)
        dot = np.dot(pixel_direction, road_direction)
        
        if dot > 0.3:  # Facing toward road
            return self.FRONT_GARDEN
        elif dot < -0.3:  # Facing away from road
            return self.BACK_GARDEN
        else:
            # Side gardens - classify based on slight bias
            return self.FRONT_GARDEN if dot >= 0 else self.BACK_GARDEN
    
    def _create_exclusion_mask(self, height: int, width: int) -> np.ndarray:
        """
        Create a binary mask of areas to exclude from classification.
        
        These are parks, sports pitches, and other public spaces.
        """
        mask = np.zeros((height, width), dtype=bool)
        
        if self.exclusion_zones_m.empty:
            return mask
        
        from src.osm import geometry_to_pixel_coords
        import cv2
        
        for _, zone in self.exclusion_zones_m.iterrows():
            try:
                # Convert zone geometry back to WGS84 for pixel conversion
                zone_wgs = gpd.GeoDataFrame(
                    geometry=[zone.geometry], 
                    crs=self.exclusion_zones_m.crs
                ).to_crs("EPSG:4326")
                
                coords = geometry_to_pixel_coords(
                    zone_wgs.geometry.iloc[0], 
                    self.geo_bounds, 
                    self.image_size
                )
                
                if coords and len(coords) >= 3:
                    pts = np.array(coords, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], True)
            except Exception:
                pass
        
        return mask
    
    def classify_mask_fast(
        self,
        garden_mask: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Fast classification using pre-computed direction map.
        
        Instead of per-pixel geometric calculations, this method:
        1. Creates a rasterized "direction to road" map for each building
        2. Uses that to classify all garden pixels at once
        3. Excludes parks, pitches, and other public spaces
        
        Args:
            garden_mask: Binary mask where 255 = garden pixel
            show_progress: Show progress bar
            
        Returns:
            Classification mask
        """
        from tqdm import tqdm
        
        height, width = garden_mask.shape
        result = np.zeros((height, width), dtype=np.uint8)
        
        if self.buildings_m.empty or self.roads_m.empty:
            return result
        
        # Step 0: Create exclusion mask for parks/pitches
        if not self.exclusion_zones_m.empty:
            if show_progress:
                print("Creating exclusion mask for public spaces...")
            exclusion_mask = self._create_exclusion_mask(height, width)
            # Remove excluded areas from garden mask
            working_mask = garden_mask.copy()
            working_mask[exclusion_mask] = 0
            excluded_pixels = np.sum(garden_mask > 0) - np.sum(working_mask > 0)
            if show_progress and excluded_pixels > 0:
                print(f"    Excluded {excluded_pixels:,} pixels in parks/pitches")
        else:
            working_mask = garden_mask
        
        # Step 1: Create building centroid and road direction arrays in pixel space
        if show_progress:
            print("Building direction map...")
        
        building_data = []  # (centroid_x, centroid_y, road_dir_x, road_dir_y)
        
        for idx in tqdm(self.building_directions.keys(), desc="Processing buildings", disable=not show_progress):
            info = self.building_directions[idx]
            
            # Get building in WGS84 for pixel conversion
            if idx not in self.buildings_wgs.index:
                continue
                
            building_wgs = self.buildings_wgs.loc[idx]
            centroid_wgs = building_wgs.geometry.centroid
            
            # Convert centroid to pixel coordinates
            cx = int((centroid_wgs.x - self.geo_bounds["west"]) / 
                    (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
            cy = int((self.geo_bounds["north"] - centroid_wgs.y) / 
                    (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
            
            # Get road direction (already computed in meters, direction is unitless)
            road_dir = info["direction_to_road"]
            
            if np.linalg.norm(road_dir) > 0.1:
                building_data.append((cx, cy, road_dir[0], road_dir[1]))
        
        if not building_data:
            return result
        
        building_arr = np.array(building_data)  # Shape: (n_buildings, 4)
        
        # Step 2: Get all garden pixel coordinates (excluding public spaces)
        garden_ys, garden_xs = np.where(working_mask > 0)
        n_pixels = len(garden_xs)
        
        if n_pixels == 0:
            return result
        
        if show_progress:
            print(f"Classifying {n_pixels:,} garden pixels...")
        
        # Step 3: Build KDTree of building centroids for fast nearest lookup
        building_centroids = building_arr[:, :2]  # (x, y)
        tree = cKDTree(building_centroids)
        
        # Step 4: Find nearest building for each garden pixel (vectorized!)
        garden_points = np.column_stack([garden_xs, garden_ys])
        
        # Query KDTree for nearest building - this is O(n log m) instead of O(n*m)
        distances, nearest_indices = tree.query(garden_points, k=1, workers=-1)
        
        # Step 5: Classify based on direction to road
        # For each pixel, compute direction from building centroid to pixel
        # Dot product with road direction determines front/back
        
        if show_progress:
            pbar = tqdm(total=3, desc="Computing classifications")
        
        # Get the building data for each pixel's nearest building
        nearest_buildings = building_arr[nearest_indices]  # Shape: (n_pixels, 4)
        
        # Direction from building centroid to pixel
        pixel_dir_x = garden_xs - nearest_buildings[:, 0]
        pixel_dir_y = garden_ys - nearest_buildings[:, 1]
        
        if show_progress:
            pbar.update(1)
        
        # Normalize pixel directions
        pixel_lengths = np.sqrt(pixel_dir_x**2 + pixel_dir_y**2)
        pixel_lengths = np.maximum(pixel_lengths, 0.1)  # Avoid division by zero
        pixel_dir_x = pixel_dir_x / pixel_lengths
        pixel_dir_y = pixel_dir_y / pixel_lengths
        
        if show_progress:
            pbar.update(1)
        
        # Road directions for each pixel's nearest building
        road_dir_x = nearest_buildings[:, 2]
        road_dir_y = nearest_buildings[:, 3]
        
        # Dot product: positive = toward road (front), negative = away (back)
        # Note: Y is inverted in image coordinates, so we negate road_dir_y
        dot_products = pixel_dir_x * road_dir_x + pixel_dir_y * (-road_dir_y)
        
        # Classify based on dot product
        # Also filter by distance (only classify pixels within reasonable distance of a building)
        max_distance = 200  # pixels - increased to catch more gardens
        valid_mask = distances < max_distance
        
        front_mask = (dot_products > 0.2) & valid_mask
        back_mask = (dot_products < -0.2) & valid_mask
        
        # Apply classifications
        result[garden_ys[front_mask], garden_xs[front_mask]] = self.FRONT_GARDEN
        result[garden_ys[back_mask], garden_xs[back_mask]] = self.BACK_GARDEN
        
        # For ambiguous pixels (side gardens), use slight bias
        ambiguous_mask = (~front_mask) & (~back_mask) & valid_mask
        result[garden_ys[ambiguous_mask & (dot_products >= 0)], 
               garden_xs[ambiguous_mask & (dot_products >= 0)]] = self.FRONT_GARDEN
        result[garden_ys[ambiguous_mask & (dot_products < 0)], 
               garden_xs[ambiguous_mask & (dot_products < 0)]] = self.BACK_GARDEN
        
        if show_progress:
            pbar.update(1)
            pbar.close()
        
        return result
    
    def _create_building_mask(self, height: int, width: int) -> np.ndarray:
        """
        Create a mask of building footprints.
        
        Returns: Binary mask where True = building, False = not building
        """
        import cv2
        from src.osm import geometry_to_pixel_coords
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for _, building in self.buildings_wgs.iterrows():
            try:
                coords = geometry_to_pixel_coords(
                    building.geometry,
                    self.geo_bounds,
                    self.image_size
                )
                if coords and len(coords) >= 3:
                    pts = np.array(coords, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
            except Exception:
                pass
        
        return mask
    
    def _create_property_boundaries(
        self,
        height: int,
        width: int,
        building_buffer_px: int = 5,  # Increased buffer
        road_width_px: int = 10,  # Wider roads as boundaries
        fence_width_px: int = 4   # Slightly wider fences
    ) -> np.ndarray:
        """
        Create a mask of property boundaries using buildings, roads, and walls/fences.
        
        These act as separators to prevent gardens from merging across properties.
        
        IMPROVED: 
        - Try to create enclosed property polygons from connected fences
        - Use watershed or connected components to find property regions
        """
        import cv2
        from src.osm import geometry_to_pixel_coords
        
        # Start with empty mask (0 = boundary, 255 = not boundary)
        boundary_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Draw buildings with small buffer as boundaries
        for _, building in self.buildings_wgs.iterrows():
            try:
                coords = geometry_to_pixel_coords(
                    building.geometry,
                    self.geo_bounds,
                    self.image_size
                )
                if coords and len(coords) >= 3:
                    pts = np.array(coords, dtype=np.int32)
                    # Fill building interior
                    cv2.fillPoly(boundary_mask, [pts], 0)
                    # Buffer around building
                    cv2.polylines(boundary_mask, [pts], True, 0, thickness=building_buffer_px)
            except Exception:
                pass
        
        # Draw roads as boundaries - these are strong separators
        for _, road in self.roads_wgs.iterrows():
            try:
                coords = geometry_to_pixel_coords(
                    road.geometry,
                    self.geo_bounds,
                    self.image_size
                )
                if coords and len(coords) >= 2:
                    pts = np.array(coords, dtype=np.int32)
                    cv2.polylines(boundary_mask, [pts], False, 0, thickness=road_width_px)
            except Exception:
                pass
        
        # Draw driveways as lighter boundaries (they separate gardens but are narrower)
        if not self.driveways_m.empty:
            for idx, driveway in self.driveways_m.iterrows():
                try:
                    # Project back to WGS84
                    driveway_gdf = gpd.GeoDataFrame(
                        geometry=[driveway.geometry], 
                        crs=self.driveways_m.crs
                    ).to_crs("EPSG:4326")
                    
                    coords = geometry_to_pixel_coords(
                        driveway_gdf.geometry.iloc[0],
                        self.geo_bounds,
                        self.image_size
                    )
                    if coords and len(coords) >= 2:
                        pts = np.array(coords, dtype=np.int32)
                        cv2.polylines(boundary_mask, [pts], False, 0, thickness=4)
                except Exception:
                    pass
        
        # Draw property boundaries (walls, fences, hedges) - these are key!
        fence_count = 0
        for _, boundary in self.property_boundaries_wgs.iterrows():
            try:
                geom = boundary.geometry
                if geom.is_empty:
                    continue
                
                # Handle different geometry types
                geom_list = []
                if geom.geom_type == 'Polygon':
                    geom_list = [(list(geom.exterior.coords), True)]
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        geom_list.append((list(poly.exterior.coords), True))
                elif geom.geom_type == 'LineString':
                    geom_list = [(list(geom.coords), False)]
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        geom_list.append((list(line.coords), False))
                else:
                    continue
                
                for coords, is_closed in geom_list:
                    pixel_coords = []
                    for lon, lat in coords:
                        px = int((lon - self.geo_bounds["west"]) / 
                                (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
                        py = int((self.geo_bounds["north"] - lat) / 
                                (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
                        pixel_coords.append((px, py))
                    
                    if len(pixel_coords) >= 2:
                        pts = np.array(pixel_coords, dtype=np.int32)
                        cv2.polylines(boundary_mask, [pts], is_closed, 0, thickness=fence_width_px)
                        fence_count += 1
                    
            except Exception:
                pass
        
        # NOTE: Removed aggressive gap-closing as it was merging gardens across properties
        # Keep boundaries as-is to maintain property separation
        
        return boundary_mask
    
    def classify_regions(
        self,
        garden_mask: np.ndarray,
        min_region_area: int = 200,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Region-based classification - classifies entire connected garden regions.
        
        Key improvement: Uses buildings and roads as natural property separators
        to prevent gardens from merging across properties.
        
        Args:
            garden_mask: Binary mask where 255 = garden pixel
            min_region_area: Minimum region size to classify (pixels)
            show_progress: Show progress bar
            
        Returns:
            Classification mask with same shape as input
        """
        import cv2
        from tqdm import tqdm
        
        height, width = garden_mask.shape
        result = np.zeros((height, width), dtype=np.uint8)
        
        if self.buildings_m.empty or self.roads_m.empty:
            return result
        
        # Step 0: Apply exclusion zones
        if not self.exclusion_zones_m.empty:
            if show_progress:
                print("Excluding public spaces...")
            exclusion_mask = self._create_exclusion_mask(height, width)
            working_mask = garden_mask.copy()
            working_mask[exclusion_mask] = 0
        else:
            working_mask = garden_mask.copy()
        
        # Step 1: Create property boundaries from buildings and roads
        if show_progress:
            print("Creating property boundaries...")
        boundary_mask = self._create_property_boundaries(height, width)
        
        # Apply boundaries to split connected gardens
        working_mask = cv2.bitwise_and(working_mask, boundary_mask)
        
        # Step 2: Minimal morphological cleanup (avoid merging across properties!)
        if show_progress:
            print("Cleaning up vegetation regions...")
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # REDUCED: Only remove small noise, don't close gaps (that merges properties)
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Step 3: Find connected regions (now properly separated by properties)
        if show_progress:
            print("Finding connected garden regions...")
        contours, _ = cv2.findContours(working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by minimum area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_region_area]
        
        if show_progress:
            print(f"    Found {len(valid_contours)} garden regions (from {len(contours)} total)")
        
        if not valid_contours:
            return result
        
        # Step 4: Build KDTree of building EDGE points for better accuracy
        # Also store the building index and area for each edge point
        building_edge_points = []  # (x, y, building_idx_in_list)
        building_data = []  # (centroid_x, centroid_y, road_dir_x, road_dir_y, area)
        idx_to_list_idx = {}  # Map original idx to list index
        
        for list_idx, idx in enumerate(self.building_directions.keys()):
            info = self.building_directions[idx]
            if idx not in self.buildings_wgs.index:
                continue
            building_wgs = self.buildings_wgs.loc[idx]
            centroid_wgs = building_wgs.geometry.centroid
            
            cx = int((centroid_wgs.x - self.geo_bounds["west"]) / 
                    (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
            cy = int((self.geo_bounds["north"] - centroid_wgs.y) / 
                    (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
            
            # Get building area for multi-building preference
            area = building_wgs.geometry.area if hasattr(building_wgs.geometry, 'area') else 1
            
            road_dir = info["direction_to_road"]
            if np.linalg.norm(road_dir) > 0.1:
                idx_to_list_idx[idx] = len(building_data)
                building_data.append((cx, cy, road_dir[0], road_dir[1], area))
                
                # Also add edge points from the building polygon
                if idx in self.building_polygons_px:
                    for px, py in self.building_polygons_px[idx]:
                        building_edge_points.append((px, py, len(building_data) - 1))
        
        if not building_data:
            return result
        
        building_arr = np.array(building_data)
        building_centroids = building_arr[:, :2]
        building_tree = cKDTree(building_centroids)
        
        # Build KDTree of building edges for edge-based classification
        if building_edge_points:
            edge_arr = np.array(building_edge_points)
            edge_points_only = edge_arr[:, :2]
            edge_tree = cKDTree(edge_points_only)
        else:
            edge_tree = None
            edge_arr = None
        
        # Also build KDTree of road points for corner house handling
        road_points = []
        from src.osm import geometry_to_pixel_coords
        for _, road in self.roads_wgs.iterrows():
            try:
                coords = geometry_to_pixel_coords(road.geometry, self.geo_bounds, self.image_size)
                if coords:
                    road_points.extend(coords)
            except Exception:
                pass
        
        if road_points:
            road_tree = cKDTree(road_points)
        else:
            road_tree = None
        
        # Build KDTree of wall/fence points to check property boundaries
        fence_points = []
        for _, boundary in self.property_boundaries_wgs.iterrows():
            try:
                geom = boundary.geometry
                if geom.is_empty:
                    continue
                coords = []
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                elif geom.geom_type == 'Polygon':
                    coords = list(geom.exterior.coords)
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        coords.extend(list(line.coords))
                
                for lon, lat in coords:
                    px = int((lon - self.geo_bounds["west"]) / 
                            (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
                    py = int((self.geo_bounds["north"] - lat) / 
                            (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
                    fence_points.append((px, py))
            except Exception:
                pass
        
        fence_tree = cKDTree(fence_points) if fence_points else None
        
        # Step 5: Classify each region
        # CRITICAL: Only classify vegetation that's INSIDE a property boundary
        # - Vegetation must be closer to building than to any fence between them
        # - Vegetation separated by a fence from nearest building = UNKNOWN (common area)
        
        front_count = 0
        back_count = 0
        unknown_count = 0
        
        # IMPROVED: Per-pixel classification instead of per-region
        # This ensures each pixel uses its OWN nearest building, preventing
        # merged regions from being classified incorrectly
        
        if show_progress:
            print("Performing per-pixel classification...")
        
        # Get all vegetation pixel coordinates
        veg_ys, veg_xs = np.where(working_mask > 0)
        n_pixels = len(veg_xs)
        
        if n_pixels == 0:
            return result
        
        if show_progress:
            print(f"    Classifying {n_pixels:,} vegetation pixels...")
        
        # Batch process in chunks for memory efficiency
        chunk_size = 100000
        
        for chunk_start in tqdm(range(0, n_pixels, chunk_size), 
                                 desc="Classifying pixels", 
                                 disable=not show_progress):
            chunk_end = min(chunk_start + chunk_size, n_pixels)
            chunk_xs = veg_xs[chunk_start:chunk_end]
            chunk_ys = veg_ys[chunk_start:chunk_end]
            chunk_points = np.column_stack([chunk_xs, chunk_ys])
            
            # Find nearest building for each pixel
            if edge_tree is not None:
                dists_to_edge, edge_indices = edge_tree.query(chunk_points, k=1)
                building_indices = edge_arr[edge_indices, 2].astype(int)
            else:
                dists_to_edge, building_indices = building_tree.query(chunk_points, k=1)
            
            # Also get distance to building centroids
            dists_to_building, _ = building_tree.query(chunk_points, k=1)
            
            # Check fence boundaries if available
            if fence_tree is not None:
                dists_to_fence, fence_indices = fence_tree.query(chunk_points, k=1)
            
            # Process each pixel in the chunk
            for i in range(len(chunk_xs)):
                px, py = chunk_xs[i], chunk_ys[i]
                building_idx = building_indices[i]
                dist_edge = dists_to_edge[i]
                dist_building = dists_to_building[i]
                
                building_info = building_arr[building_idx]
                building_cx, building_cy = building_info[0], building_info[1]
                building_area = building_info[4] if len(building_info) > 4 else 100
                road_dir_x, road_dir_y = building_info[2], building_info[3]
                
                # PROPERTY BOUNDARY CHECK
                is_inside_property = True
                
                # Check if fence is between pixel and building
                if fence_tree is not None:
                    dist_fence = dists_to_fence[i]
                    if dist_fence < dist_edge * 0.4:
                        fence_pt = fence_points[fence_indices[i]]
                        vb_x = building_cx - px
                        vb_y = building_cy - py
                        vb_len = np.sqrt(vb_x**2 + vb_y**2)
                        vf_x = fence_pt[0] - px
                        vf_y = fence_pt[1] - py
                        vf_len = np.sqrt(vf_x**2 + vf_y**2)
                        
                        if vb_len > 0.1 and vf_len > 0.1:
                            dot = (vb_x * vf_x + vb_y * vf_y) / (vb_len * vf_len)
                            if dot > 0.6:
                                is_inside_property = False
                
                # Adaptive max distance based on building size
                base_max_distance = 250  # pixels
                area_factor = min(2.0, max(1.0, building_area / 0.00001))
                max_distance = int(base_max_distance * area_factor)
                max_distance = min(max_distance, 500)
                
                if dist_edge > max_distance:
                    is_inside_property = False
                
                if not is_inside_property:
                    result[py, px] = self.UNKNOWN
                    unknown_count += 1
                    continue
                
                # Classify based on building's road direction
                vec_x = px - building_cx
                vec_y = py - building_cy
                projection = vec_x * road_dir_x + vec_y * (-road_dir_y)
                
                if projection >= 0:
                    result[py, px] = self.FRONT_GARDEN
                    front_count += 1
                else:
                    result[py, px] = self.BACK_GARDEN
                    back_count += 1
        
        if show_progress:
            print(f"    Classified: {front_count} front, {back_count} back, {unknown_count} unknown (outside property)")
        
        # Final step: Ensure buildings are NOT colored as gardens
        # Apply building mask with buffer to ensure clean edges
        if show_progress:
            print("Removing building footprints from result...")
        building_mask = self._create_building_mask(height, width)
        
        # Dilate building mask slightly to ensure no garden color bleeds onto buildings
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        building_mask_dilated = cv2.dilate(building_mask, kernel, iterations=1)
        
        result[building_mask_dilated > 0] = self.NOT_GARDEN
        
        return result
    
    def classify_mask(
        self,
        garden_mask: np.ndarray,
        sample_step: int = 4,
        show_progress: bool = True,
        use_regions: bool = True
    ) -> np.ndarray:
        """
        Classify all garden pixels in a mask.
        
        Args:
            garden_mask: Binary mask where 255 = garden pixel
            sample_step: (Unused, kept for compatibility)
            show_progress: Show progress bar
            use_regions: If True, use region-based classification (recommended)
            
        Returns:
            Classification mask with same shape as input:
            - 0 = not garden
            - 1 = front garden
            - 2 = back garden
        """
        if use_regions:
            return self.classify_regions(garden_mask, show_progress=show_progress)
        else:
            return self.classify_mask_fast(garden_mask, show_progress)
    
    def get_classification_stats(self, classification_mask: np.ndarray) -> dict:
        """
        Get statistics about the classification.
        
        Args:
            classification_mask: Result from classify_mask()
            
        Returns:
            Dict with counts and percentages
        """
        front_pixels = np.sum(classification_mask == self.FRONT_GARDEN)
        back_pixels = np.sum(classification_mask == self.BACK_GARDEN)
        unknown_pixels = np.sum(classification_mask == self.UNKNOWN)
        total_garden = front_pixels + back_pixels + unknown_pixels
        
        return {
            "front_garden_pixels": front_pixels,
            "back_garden_pixels": back_pixels,
            "unknown_pixels": unknown_pixels,
            "total_garden_pixels": total_garden,
            "front_percentage": (front_pixels / total_garden * 100) if total_garden > 0 else 0,
            "back_percentage": (back_pixels / total_garden * 100) if total_garden > 0 else 0,
            "unknown_percentage": (unknown_pixels / total_garden * 100) if total_garden > 0 else 0,
        }


def create_classification_overlay(
    image: np.ndarray,
    classification_mask: np.ndarray,
    front_color: Tuple[int, int, int] = (0, 255, 0),  # Green
    back_color: Tuple[int, int, int] = (0, 0, 255),   # Blue (RGB, not BGR)
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create an overlay image showing front/back garden classification.
    
    Args:
        image: Original RGB image
        classification_mask: Classification result
        front_color: RGB color for front gardens
        back_color: RGB color for back gardens
        alpha: Transparency of overlay
        
    Returns:
        RGB image with colored overlay
    """
    overlay = image.copy()
    
    # Create colored overlay
    front_mask = classification_mask == GardenClassifier.FRONT_GARDEN
    back_mask = classification_mask == GardenClassifier.BACK_GARDEN
    
    # Apply colors
    overlay[front_mask] = (
        (1 - alpha) * overlay[front_mask] + alpha * np.array(front_color)
    ).astype(np.uint8)
    
    overlay[back_mask] = (
        (1 - alpha) * overlay[back_mask] + alpha * np.array(back_color)
    ).astype(np.uint8)
    
    return overlay
