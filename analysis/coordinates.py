"""
Coordinate conversion utilities for GPS/IMU analysis.
Converts lat/lon to local ENU (East-North-Up) coordinates.
"""

import math
from typing import Tuple, List
import numpy as np


# Earth radius in meters (WGS84 mean radius)
EARTH_RADIUS = 6371000.0


def latlon_to_enu(lat: float, lon: float, lat_ref: float, lon_ref: float) -> Tuple[float, float]:
    """
    Convert latitude/longitude to local ENU (East-North-Up) coordinates.
    
    Uses a simple equirectangular projection which is accurate for small areas.
    Reference point becomes the origin (0, 0).
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        lat_ref: Reference latitude in degrees (origin)
        lon_ref: Reference longitude in degrees (origin)
        
    Returns:
        Tuple of (east, north) in meters
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat_ref_rad = math.radians(lat_ref)
    lon_ref_rad = math.radians(lon_ref)
    
    # East: longitude difference scaled by cosine of latitude
    east = EARTH_RADIUS * (lon_rad - lon_ref_rad) * math.cos(lat_ref_rad)
    
    # North: latitude difference
    north = EARTH_RADIUS * (lat_rad - lat_ref_rad)
    
    return east, north


def latlon_array_to_enu(
    lats: np.ndarray, 
    lons: np.ndarray, 
    lat_ref: float = None, 
    lon_ref: float = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Convert arrays of lat/lon to ENU coordinates.
    
    If reference point not provided, uses the first point.
    
    Returns:
        Tuple of (east_array, north_array, lat_ref, lon_ref)
    """
    if lat_ref is None:
        lat_ref = lats[0]
    if lon_ref is None:
        lon_ref = lons[0]
    
    lat_ref_rad = math.radians(lat_ref)
    
    east = EARTH_RADIUS * np.radians(lons - lon_ref) * math.cos(lat_ref_rad)
    north = EARTH_RADIUS * np.radians(lats - lat_ref)
    
    return east, north, lat_ref, lon_ref


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate 2D Euclidean distance."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def compute_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute distances between consecutive points.
    Returns array of length n-1.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx**2 + dy**2)


def compute_cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute cumulative distance traveled.
    Returns array of same length as input, starting with 0.
    """
    distances = compute_distances(x, y)
    return np.concatenate([[0], np.cumsum(distances)])


def compute_speed_from_position(
    x: np.ndarray, 
    y: np.ndarray, 
    time: np.ndarray,
    window: int = 1
) -> np.ndarray:
    """
    Compute speed from position and time.
    
    Args:
        x: East positions
        y: North positions
        time: Timestamps in seconds
        window: Number of samples to use for smoothing (1 = no smoothing)
        
    Returns:
        Array of speeds in m/s
    """
    n = len(x)
    speeds = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - window)
        
        dist = distance_2d(x[start_idx], y[start_idx], x[i], y[i])
        dt = time[i] - time[start_idx]
        
        if dt > 0:
            speeds[i] = dist / dt
        elif i > 0:
            speeds[i] = speeds[i-1]
    
    return speeds
