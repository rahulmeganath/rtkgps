"""
IMU Analysis module.
Analyzes gyroscope data for turn detection and noise characterization.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Turn:
    """A detected turn event based on gyro data."""
    index: int
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration: float
    direction: str  # 'left' or 'right'
    peak_rate: float  # rad/s
    total_rotation: float  # radians


@dataclass
class IMUAnalysis:
    """IMU analysis results."""
    # Gyro Z data
    gyro_z: np.ndarray = None
    gyro_z_available: bool = False
    
    # Noise metrics (computed at stops)
    gyro_noise_std: Optional[float] = None
    gyro_noise_max: Optional[float] = None
    gyro_bias: Optional[float] = None
    noise_computed_at_stops: bool = False
    
    # Turn detection
    turns: List[Turn] = field(default_factory=list)
    turn_count: int = 0
    left_turns: int = 0
    right_turns: int = 0
    
    # Summary
    gyro_z_mean: float = 0.0
    gyro_z_std: float = 0.0
    gyro_z_max: float = 0.0


def analyze_imu(
    gyro_z: np.ndarray,
    time: np.ndarray,
    is_stopped: np.ndarray = None,
    turn_threshold: float = 0.2,  # rad/s
    min_turn_duration: float = 0.5  # seconds
) -> IMUAnalysis:
    """
    Analyze IMU gyroscope data.
    
    Args:
        gyro_z: Yaw rate in rad/s
        time: Timestamps in seconds
        is_stopped: Boolean array indicating stopped samples (for noise analysis)
        turn_threshold: Gyro rate above which to detect turn (rad/s)
        min_turn_duration: Minimum turn duration (seconds)
        
    Returns:
        IMUAnalysis with results
    """
    result = IMUAnalysis()
    
    if gyro_z is None or len(gyro_z) == 0:
        return result
    
    result.gyro_z = gyro_z
    result.gyro_z_available = True
    
    # Basic stats
    result.gyro_z_mean = float(np.mean(gyro_z))
    result.gyro_z_std = float(np.std(gyro_z))
    result.gyro_z_max = float(np.max(np.abs(gyro_z)))
    
    # Compute noise at stops
    if is_stopped is not None:
        stopped_gyro = gyro_z[is_stopped]
        if len(stopped_gyro) > 10:
            result.noise_computed_at_stops = True
            result.gyro_noise_std = float(np.std(stopped_gyro))
            result.gyro_noise_max = float(np.max(np.abs(stopped_gyro)))
            result.gyro_bias = float(np.mean(stopped_gyro))
    
    # Detect turns
    turns = _detect_turns(gyro_z, time, turn_threshold, min_turn_duration)
    result.turns = turns
    result.turn_count = len(turns)
    result.left_turns = sum(1 for t in turns if t.direction == 'left')
    result.right_turns = sum(1 for t in turns if t.direction == 'right')
    
    return result


def _detect_turns(
    gyro_z: np.ndarray,
    time: np.ndarray,
    threshold: float,
    min_duration: float
) -> List[Turn]:
    """Detect turn events from gyro data."""
    n = len(gyro_z)
    turns = []
    
    # Find regions where |gyro_z| > threshold
    above_threshold = np.abs(gyro_z) > threshold
    
    current_turn_start = None
    turn_index = 0
    
    for i in range(n):
        if above_threshold[i]:
            if current_turn_start is None:
                current_turn_start = i
        else:
            if current_turn_start is not None:
                turn_end = i - 1
                duration = time[turn_end] - time[current_turn_start]
                
                if duration >= min_duration:
                    turn_index += 1
                    turn = _create_turn(
                        turn_index, current_turn_start, turn_end,
                        gyro_z, time
                    )
                    turns.append(turn)
                
                current_turn_start = None
    
    # Handle end of data
    if current_turn_start is not None:
        turn_end = n - 1
        duration = time[turn_end] - time[current_turn_start]
        
        if duration >= min_duration:
            turn_index += 1
            turn = _create_turn(
                turn_index, current_turn_start, turn_end,
                gyro_z, time
            )
            turns.append(turn)
    
    return turns


def _create_turn(
    index: int,
    start_idx: int,
    end_idx: int,
    gyro_z: np.ndarray,
    time: np.ndarray
) -> Turn:
    """Create a Turn object with computed metrics."""
    turn_gyro = gyro_z[start_idx:end_idx+1]
    turn_time = time[start_idx:end_idx+1]
    
    # Find peak rate
    peak_idx = np.argmax(np.abs(turn_gyro))
    peak_rate = turn_gyro[peak_idx]
    
    # Direction based on sign of peak
    direction = 'left' if peak_rate > 0 else 'right'
    
    # Integrate rotation (trapezoidal)
    total_rotation = np.trapz(turn_gyro, turn_time)
    
    return Turn(
        index=index,
        start_idx=start_idx,
        end_idx=end_idx,
        start_time=time[start_idx],
        end_time=time[end_idx],
        duration=time[end_idx] - time[start_idx],
        direction=direction,
        peak_rate=peak_rate,
        total_rotation=total_rotation
    )


def get_turn_summary_table(turns: List[Turn]) -> List[Dict]:
    """Convert turns to list of dicts for DataFrame display."""
    return [
        {
            'Turn #': t.index,
            'Direction': t.direction.capitalize(),
            'Duration (s)': round(t.duration, 2),
            'Peak Rate (deg/s)': round(np.degrees(t.peak_rate), 1),
            'Total Rotation (deg)': round(np.degrees(t.total_rotation), 1),
            'Start Time (s)': round(t.start_time, 1),
        }
        for t in turns
    ]


def get_imu_interpretation() -> str:
    """Return explanation text about IMU analysis purpose."""
    return """
    **IMU Analysis Purpose**
    
    The IMU (Inertial Measurement Unit) provides gyroscope data that measures 
    rotational velocity. In this dashboard:
    
    - **Gyro Z (yaw rate)** measures rotation around the vertical axis
    - Used for **turn execution detection**, not for zone/junction detection
    - GPS is used for zone entry decisions; IMU confirms turn maneuvers
    
    **What to look for:**
    - Gyro noise at stops should be < 0.05 rad/s (3 deg/s)
    - Turn detection aligns with expected vehicle maneuvers
    - No spurious turn detections during straight driving
    """
