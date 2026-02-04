"""
Garden detection module

Detects green vegetation areas (potential gardens) from aerial imagery
using color-based segmentation in HSV color space.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

import config


def detect_green_areas(
    image: np.ndarray,
    hsv_lower: Tuple[int, int, int] = None,
    hsv_upper: Tuple[int, int, int] = None,
    min_area: int = None,
    show_progress: bool = True,
    enhanced: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Detect green vegetation areas in an aerial image using HSV color thresholding.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        hsv_lower: Lower HSV threshold (default from config)
        hsv_upper: Upper HSV threshold (default from config)
        min_area: Minimum contour area in pixels (default from config)
        show_progress: Show progress bar
        enhanced: Use enhanced multi-range detection for better coverage
        
    Returns:
        Tuple of:
        - Binary mask where 255 = green area, 0 = not green
        - List of contours (potential garden boundaries)
    """
    from tqdm import tqdm
    
    if hsv_lower is None:
        hsv_lower = config.GREEN_HSV_LOWER
    if hsv_upper is None:
        hsv_upper = config.GREEN_HSV_UPPER
    if min_area is None:
        min_area = config.MIN_GARDEN_AREA_PX
    
    steps = ["Converting color space", "Creating mask", "Morphological ops", "Finding contours", "Filtering"]
    
    if show_progress:
        pbar = tqdm(total=len(steps), desc="Detecting vegetation")
    
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if show_progress:
        pbar.update(1)
    
    if enhanced:
        # Use multiple color ranges to catch ALL types of grass and vegetation
        # Range 1: Standard green (healthy grass)
        mask1 = cv2.inRange(hsv, np.array([35, 25, 25]), np.array([85, 255, 255]))
        
        # Range 2: Yellow-green (dry/autumn grass)
        mask2 = cv2.inRange(hsv, np.array([20, 15, 40]), np.array([45, 255, 255]))
        
        # Range 3: Dark green (shadowed grass, hedges, evergreens)
        mask3 = cv2.inRange(hsv, np.array([30, 10, 10]), np.array([95, 255, 200]))
        
        # Range 4: Light/bright green (well-lit lawns)
        mask4 = cv2.inRange(hsv, np.array([25, 20, 50]), np.array([100, 255, 255]))
        
        # Range 5: Very dark green (deep shadows, dense hedges)
        mask5 = cv2.inRange(hsv, np.array([30, 5, 5]), np.array([90, 200, 150]))
        
        # Range 6: Olive/brownish green (winter grass, dried patches)
        mask6 = cv2.inRange(hsv, np.array([15, 10, 30]), np.array([50, 200, 200]))
        
        # Range 7: Blue-green (some lawn types, moss)
        mask7 = cv2.inRange(hsv, np.array([75, 15, 20]), np.array([110, 255, 255]))
        
        # Combine all ranges
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
        mask = cv2.bitwise_or(mask, mask4)
        mask = cv2.bitwise_or(mask, mask5)
        mask = cv2.bitwise_or(mask, mask6)
        mask = cv2.bitwise_or(mask, mask7)
    else:
        # Simple single-range detection
        mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    
    if show_progress:
        pbar.update(1)
    
    # Morphological operations - conservative to avoid merging across properties
    # The classifier will use buildings/roads as separators, so we don't need aggressive merging here
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Close small gaps within grass patches
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    
    # Remove small noise specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Fill holes within regions (like patches where grass detection missed)
    mask = fill_holes(mask)
    
    if show_progress:
        pbar.update(1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if show_progress:
        pbar.update(1)
    
    # Filter by minimum area
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    # Create cleaned mask from filtered contours
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, filtered_contours, -1, 255, -1)
    if show_progress:
        pbar.update(1)
        pbar.close()
    
    return clean_mask, filtered_contours


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes within regions of a binary mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Mask with holes filled
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fill each contour (this fills internal holes)
    result = np.zeros_like(mask)
    cv2.drawContours(result, contours, -1, 255, -1)
    
    return result


def detect_vegetation_enhanced(
    image: np.ndarray,
    use_texture: bool = True
) -> np.ndarray:
    """
    Enhanced vegetation detection using multiple color spaces and optional texture analysis.
    
    This method combines:
    - HSV green detection
    - Excess Green Index (ExG) from RGB
    - Optional texture filtering to exclude non-grass surfaces
    
    Args:
        image: RGB image as numpy array
        use_texture: Whether to use texture analysis
        
    Returns:
        Binary mask where 255 = vegetation, 0 = not vegetation
    """
    # Method 1: HSV-based detection
    hsv_mask, _ = detect_green_areas(image)
    
    # Method 2: Excess Green Index
    # ExG = 2*G - R - B
    image_float = image.astype(np.float32) / 255.0
    r, g, b = image_float[:,:,0], image_float[:,:,1], image_float[:,:,2]
    
    exg = 2 * g - r - b
    exg_mask = (exg > 0.1).astype(np.uint8) * 255
    
    # Combine masks (both methods must agree)
    combined_mask = cv2.bitwise_and(hsv_mask, exg_mask)
    
    if use_texture:
        # Use texture to filter out smooth surfaces (like artificial turf or painted areas)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local standard deviation as texture measure
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        texture = np.abs(gray.astype(np.float32) - blur.astype(np.float32))
        
        # Natural grass has more texture variation than artificial surfaces
        texture_mask = (texture > 3).astype(np.uint8) * 255
        
        # Dilate texture mask to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        texture_mask = cv2.dilate(texture_mask, kernel, iterations=2)
        
        combined_mask = cv2.bitwise_and(combined_mask, texture_mask)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return combined_mask


def exclude_buildings_from_mask(
    vegetation_mask: np.ndarray,
    building_polygons_px: List[List[Tuple[int, int]]]
) -> np.ndarray:
    """
    Remove building footprints from vegetation mask.
    
    Args:
        vegetation_mask: Binary vegetation mask
        building_polygons_px: List of building polygons in pixel coordinates
        
    Returns:
        Vegetation mask with buildings excluded
    """
    result = vegetation_mask.copy()
    
    for poly_coords in building_polygons_px:
        if len(poly_coords) >= 3:
            pts = np.array(poly_coords, dtype=np.int32)
            cv2.fillPoly(result, [pts], 0)
    
    return result


def exclude_roads_from_mask(
    vegetation_mask: np.ndarray,
    road_lines_px: List[List[Tuple[int, int]]],
    road_width_px: int = 10
) -> np.ndarray:
    """
    Remove road areas from vegetation mask.
    
    Args:
        vegetation_mask: Binary vegetation mask
        road_lines_px: List of road lines in pixel coordinates
        road_width_px: Width of roads in pixels
        
    Returns:
        Vegetation mask with roads excluded
    """
    result = vegetation_mask.copy()
    
    for line_coords in road_lines_px:
        if len(line_coords) >= 2:
            pts = np.array(line_coords, dtype=np.int32)
            cv2.polylines(result, [pts], False, 0, thickness=road_width_px)
    
    return result


def get_garden_statistics(mask: np.ndarray, meters_per_pixel: float = 1.0) -> dict:
    """
    Calculate statistics about detected garden areas.
    
    Args:
        mask: Binary garden mask
        meters_per_pixel: Scale factor for area calculation
        
    Returns:
        Dict with statistics
    """
    total_pixels = mask.size
    garden_pixels = np.sum(mask > 0)
    
    # Find contours for individual garden count
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return {
        "total_pixels": total_pixels,
        "garden_pixels": garden_pixels,
        "garden_percentage": (garden_pixels / total_pixels) * 100,
        "garden_count": len(contours),
        "garden_area_m2": garden_pixels * (meters_per_pixel ** 2),
    }
