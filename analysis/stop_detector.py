"""
Stop detection algorithm for GPS/IMU data.
Identifies vehicle stops using speed threshold and minimum duration.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .coordinates import distance_2d, compute_speed_from_position


@dataclass
class Stop:
    """Represents a detected stop event."""
    index: int  # Stop number (1-indexed)
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration: float
    sample_count: int
    
    # Position (ENU meters)
    mean_x: float
    mean_y: float
    
    # GPS noise at stop
    pos_std: float  # Standard deviation from mean
    pos_max_radius: float  # Max distance from mean
    
    # Original lat/lon (center point)
    lat: float
    lon: float


@dataclass 
class StopAnalysis:
    """Complete stop analysis results."""
    stops: List[Stop]
    is_stopped: np.ndarray  # Boolean array per sample
    speed: np.ndarray  # Speed used for detection
    
    # Summary stats
    total_stop_time: float
    total_move_time: float
    stop_count: int
    shortest_stop: float
    longest_stop: float
    worst_gps_noise: float  # Max pos_std at any stop
    mean_gps_noise: float  # Mean pos_std across stops


def detect_stops(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    speed: np.ndarray = None,
    lat: np.ndarray = None,
    lon: np.ndarray = None,
    speed_threshold: float = 0.3,
    min_duration: float = 1.5,
    window_size: int = 50
) -> StopAnalysis:
    """
    Detect stops in trajectory.
    
    Args:
        x: East positions (meters)
        y: North positions (meters)
        time: Timestamps (seconds)
        speed: Speed array (m/s). If None, computed from position
        lat: Latitude array (for output metadata)
        lon: Longitude array (for output metadata)
        speed_threshold: Speed below which vehicle is stopped (m/s)
        min_duration: Minimum stop duration (seconds)
        window_size: Window for speed smoothing when computing from position
        
    Returns:
        StopAnalysis with detected stops and metrics
    """
    n = len(x)
    
    if n == 0:
        return StopAnalysis(
            stops=[], is_stopped=np.array([]), speed=np.array([]),
            total_stop_time=0, total_move_time=0, stop_count=0,
            shortest_stop=0, longest_stop=0, worst_gps_noise=0, mean_gps_noise=0
        )
    
    # Compute or use provided speed
    if speed is None:
        speed = compute_speed_from_position(x, y, time, window=window_size)
    
    # Determine stopped samples
    is_stopped = speed < speed_threshold
    
    # Group consecutive stopped samples into stop events
    stops = []
    current_stop_start = None
    stop_index = 0
    
    for i in range(n):
        if is_stopped[i]:
            if current_stop_start is None:
                current_stop_start = i
        else:
            if current_stop_start is not None:
                stop_end = i - 1
                duration = time[stop_end] - time[current_stop_start]
                
                if duration >= min_duration:
                    stop_index += 1
                    stop = _create_stop(
                        stop_index, current_stop_start, stop_end,
                        x, y, time, lat, lon
                    )
                    stops.append(stop)
                
                current_stop_start = None
    
    # Handle end of data
    if current_stop_start is not None:
        stop_end = n - 1
        duration = time[stop_end] - time[current_stop_start]
        
        if duration >= min_duration:
            stop_index += 1
            stop = _create_stop(
                stop_index, current_stop_start, stop_end,
                x, y, time, lat, lon
            )
            stops.append(stop)
    
    # Compute summary stats
    total_stop_time = sum(s.duration for s in stops)
    total_time = time[-1] - time[0] if n > 1 else 0
    total_move_time = total_time - total_stop_time
    
    if stops:
        shortest_stop = min(s.duration for s in stops)
        longest_stop = max(s.duration for s in stops)
        worst_gps_noise = max(s.pos_std for s in stops)
        mean_gps_noise = np.mean([s.pos_std for s in stops])
    else:
        shortest_stop = longest_stop = worst_gps_noise = mean_gps_noise = 0
    
    return StopAnalysis(
        stops=stops,
        is_stopped=is_stopped,
        speed=speed,
        total_stop_time=total_stop_time,
        total_move_time=total_move_time,
        stop_count=len(stops),
        shortest_stop=shortest_stop,
        longest_stop=longest_stop,
        worst_gps_noise=worst_gps_noise,
        mean_gps_noise=mean_gps_noise
    )


def _create_stop(
    index: int,
    start_idx: int,
    end_idx: int,
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray
) -> Stop:
    """Create a Stop object with computed metrics."""
    # Extract stop points
    stop_x = x[start_idx:end_idx+1]
    stop_y = y[start_idx:end_idx+1]
    
    # Mean position
    mean_x = np.mean(stop_x)
    mean_y = np.mean(stop_y)
    
    # Position scatter
    distances = np.array([distance_2d(sx, sy, mean_x, mean_y) 
                          for sx, sy in zip(stop_x, stop_y)])
    pos_std = np.std(distances) if len(distances) > 1 else 0
    pos_max_radius = np.max(distances) if len(distances) > 0 else 0
    
    # Center lat/lon
    mid_idx = (start_idx + end_idx) // 2
    center_lat = lat[mid_idx] if lat is not None else 0
    center_lon = lon[mid_idx] if lon is not None else 0
    
    return Stop(
        index=index,
        start_idx=start_idx,
        end_idx=end_idx,
        start_time=time[start_idx],
        end_time=time[end_idx],
        duration=time[end_idx] - time[start_idx],
        sample_count=end_idx - start_idx + 1,
        mean_x=mean_x,
        mean_y=mean_y,
        pos_std=pos_std,
        pos_max_radius=pos_max_radius,
        lat=center_lat,
        lon=center_lon
    )


def get_stop_summary_table(stops: List[Stop]) -> List[Dict]:
    """Convert stops to list of dicts for DataFrame display."""
    return [
        {
            'Stop #': s.index,
            'Duration (s)': round(s.duration, 2),
            'Samples': s.sample_count,
            'Position (E,N)': f"({s.mean_x:.1f}, {s.mean_y:.1f}) m",
            'GPS Noise (std)': f"{s.pos_std*100:.1f} cm",
            'GPS Noise (max)': f"{s.pos_max_radius*100:.1f} cm",
            'Start Time': f"{s.start_time:.1f} s",
            'Lat/Lon': f"{s.lat:.6f}, {s.lon:.6f}"
        }
        for s in stops
    ]
