"""
CSV Parser with auto-column detection for GPS/IMU data.
Handles various CSV formats and gracefully degrades when columns are missing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re


@dataclass
class ColumnMapping:
    """Stores detected column mappings."""
    latitude: Optional[str] = None
    longitude: Optional[str] = None
    timestamp: Optional[str] = None
    timestamp_nsec: Optional[str] = None
    speed: Optional[str] = None
    pos_std_lat: Optional[str] = None
    pos_std_lon: Optional[str] = None
    gnss_status: Optional[str] = None
    gyro_z: Optional[str] = None
    gyro_x: Optional[str] = None
    gyro_y: Optional[str] = None
    accel_x: Optional[str] = None
    accel_y: Optional[str] = None
    accel_z: Optional[str] = None
    vel_east: Optional[str] = None
    vel_north: Optional[str] = None
    yaw: Optional[str] = None


# Column name patterns for auto-detection
COLUMN_PATTERNS = {
    'latitude': [r'^lat(?:itude)?$', r'^lat$', r'^y$'],
    'longitude': [r'^lon(?:gitude)?$', r'^lng$', r'^lon$', r'^x$'],
    'timestamp': [r'^(?:ros_)?secs?$', r'^time(?:stamp)?$', r'^t$', r'^epoch'],
    'timestamp_nsec': [r'^(?:ros_)?nsecs?$', r'^nanosec'],
    'speed': [r'^speed$', r'^velocity$', r'^vel$', r'^ground_speed'],
    'pos_std_lat': [r'^pos_std_lat$', r'^lat_std$', r'^position_std', r'^accuracy', r'^hdop'],
    'pos_std_lon': [r'^pos_std_lon$', r'^lon_std$'],
    'gnss_status': [r'^status(?:_name)?$', r'^fix_type$', r'^gnss_status'],
    'gyro_z': [r'^(?:ang_vel_z|gyro_z|yaw_rate)$'],
    'gyro_x': [r'^(?:ang_vel_x|gyro_x|roll_rate)$'],
    'gyro_y': [r'^(?:ang_vel_y|gyro_y|pitch_rate)$'],
    'accel_x': [r'^(?:lin_accel_x|accel_x|ax)$'],
    'accel_y': [r'^(?:lin_accel_y|accel_y|ay)$'],
    'accel_z': [r'^(?:lin_accel_z|accel_z|az)$'],
    'vel_east': [r'^(?:enu_vel_e|vel_e|velocity_e|ve)$'],
    'vel_north': [r'^(?:enu_vel_n|vel_n|velocity_n|vn)$'],
    'yaw': [r'^(?:yaw(?:_deg)?|heading)$'],
}


@dataclass
class ParsedData:
    """Container for parsed GPS/IMU data."""
    df: pd.DataFrame
    mapping: ColumnMapping
    warnings: List[str] = field(default_factory=list)
    
    # Derived data (filled after parsing)
    time: np.ndarray = None
    latitude: np.ndarray = None
    longitude: np.ndarray = None
    speed: np.ndarray = None
    pos_std_xy: np.ndarray = None
    gnss_status: np.ndarray = None
    gyro_z: np.ndarray = None
    
    # Metadata
    duration: float = 0.0
    sample_rate: float = 0.0
    sample_count: int = 0


def detect_columns(columns: List[str]) -> Tuple[ColumnMapping, List[str]]:
    """
    Auto-detect column mappings from column names.
    
    Returns:
        Tuple of (ColumnMapping, list of warnings)
    """
    mapping = ColumnMapping()
    warnings = []
    columns_lower = {c.lower(): c for c in columns}
    
    for field_name, patterns in COLUMN_PATTERNS.items():
        for pattern in patterns:
            for col_lower, col_orig in columns_lower.items():
                if re.match(pattern, col_lower, re.IGNORECASE):
                    setattr(mapping, field_name, col_orig)
                    break
            if getattr(mapping, field_name) is not None:
                break
    
    # Check required columns
    if mapping.latitude is None:
        warnings.append("⚠️ No latitude column detected")
    if mapping.longitude is None:
        warnings.append("⚠️ No longitude column detected")
    if mapping.timestamp is None:
        warnings.append("⚠️ No timestamp column detected")
    
    # Optional column warnings
    if mapping.speed is None and mapping.vel_east is None:
        warnings.append("ℹ️ No speed column - will compute from position")
    if mapping.pos_std_lat is None:
        warnings.append("ℹ️ No position std column - reported uncertainty unavailable")
    if mapping.gyro_z is None:
        warnings.append("ℹ️ No gyro Z column - IMU analysis limited")
    
    return mapping, warnings


def parse_csv(file_path_or_buffer, delimiter: str = None) -> ParsedData:
    """
    Parse a GPS/IMU CSV file with auto-column detection.
    
    Args:
        file_path_or_buffer: File path or file-like object
        delimiter: CSV delimiter (auto-detected if None)
        
    Returns:
        ParsedData object with parsed data and metadata
    """
    # Read CSV
    try:
        if delimiter:
            df = pd.read_csv(file_path_or_buffer, delimiter=delimiter)
        else:
            # Try to detect delimiter
            df = pd.read_csv(file_path_or_buffer)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")
    
    if len(df) == 0:
        raise ValueError("CSV file is empty")
    
    # Detect columns
    mapping, warnings = detect_columns(list(df.columns))
    
    # Create result object
    result = ParsedData(df=df, mapping=mapping, warnings=warnings)
    result.sample_count = len(df)
    
    # Extract time
    if mapping.timestamp:
        time_sec = pd.to_numeric(df[mapping.timestamp], errors='coerce').fillna(0).values
        if mapping.timestamp_nsec:
            time_nsec = pd.to_numeric(df[mapping.timestamp_nsec], errors='coerce').fillna(0).values
            result.time = time_sec + time_nsec * 1e-9
        else:
            result.time = time_sec
        
        # Normalize to start at 0
        result.time = result.time - result.time[0]
        result.duration = result.time[-1] if len(result.time) > 0 else 0
        result.sample_rate = (len(result.time) - 1) / result.duration if result.duration > 0 else 0
    else:
        # Assume 100Hz if no timestamp
        result.time = np.arange(len(df)) / 100.0
        result.duration = result.time[-1]
        result.sample_rate = 100.0
    
    # Extract latitude/longitude
    if mapping.latitude:
        result.latitude = pd.to_numeric(df[mapping.latitude], errors='coerce').values
    if mapping.longitude:
        result.longitude = pd.to_numeric(df[mapping.longitude], errors='coerce').values
    
    # Extract or compute speed
    if mapping.speed:
        result.speed = pd.to_numeric(df[mapping.speed], errors='coerce').fillna(0).values
    elif mapping.vel_east and mapping.vel_north:
        ve = pd.to_numeric(df[mapping.vel_east], errors='coerce').fillna(0).values
        vn = pd.to_numeric(df[mapping.vel_north], errors='coerce').fillna(0).values
        result.speed = np.sqrt(ve**2 + vn**2)
    else:
        result.speed = None  # Will be computed from position later
    
    # Extract position std
    if mapping.pos_std_lat:
        std_lat = pd.to_numeric(df[mapping.pos_std_lat], errors='coerce').fillna(0).values
        if mapping.pos_std_lon:
            std_lon = pd.to_numeric(df[mapping.pos_std_lon], errors='coerce').fillna(0).values
            result.pos_std_xy = np.sqrt(std_lat**2 + std_lon**2)
        else:
            result.pos_std_xy = std_lat
    
    # Extract GNSS status
    if mapping.gnss_status:
        result.gnss_status = df[mapping.gnss_status].values
    
    # Extract gyro Z
    if mapping.gyro_z:
        result.gyro_z = pd.to_numeric(df[mapping.gyro_z], errors='coerce').fillna(0).values
    
    return result


def get_data_summary(data: ParsedData) -> Dict:
    """Generate a summary of the parsed data."""
    summary = {
        'sample_count': data.sample_count,
        'duration_sec': data.duration,
        'duration_min': data.duration / 60,
        'sample_rate_hz': data.sample_rate,
        'has_speed': data.speed is not None,
        'has_pos_std': data.pos_std_xy is not None,
        'has_gnss_status': data.gnss_status is not None,
        'has_gyro': data.gyro_z is not None,
    }
    
    if data.pos_std_xy is not None:
        valid_std = data.pos_std_xy[~np.isnan(data.pos_std_xy)]
        if len(valid_std) > 0:
            summary['pos_std_mean'] = float(np.mean(valid_std))
            summary['pos_std_max'] = float(np.max(valid_std))
            summary['pos_std_min'] = float(np.min(valid_std))
    
    if data.gnss_status is not None:
        status_counts = pd.Series(data.gnss_status).value_counts().to_dict()
        summary['gnss_status_distribution'] = status_counts
    
    return summary
