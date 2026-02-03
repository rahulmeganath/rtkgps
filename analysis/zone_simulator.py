"""
Zone simulation engine for GPS-based detection.
Simulates zone entry/exit with hysteresis and persistence.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from .coordinates import distance_2d, latlon_to_enu


@dataclass
class Zone:
    """Definition of a detection zone."""
    id: int
    name: str
    center_x: float  # ENU meters
    center_y: float  # ENU meters
    radius: float  # meters
    hysteresis: float = 2.0  # meters (exit at radius + hysteresis)
    persistence: int = 30  # samples to confirm entry
    
    # Original lat/lon
    lat: float = 0.0
    lon: float = 0.0


@dataclass
class ZoneEvent:
    """A zone entry or exit event."""
    event_type: str  # 'entry' or 'exit'
    time: float
    index: int
    distance: float


@dataclass
class ZoneSimulationResult:
    """Results from simulating zone detection."""
    zone: Zone
    
    # Per-sample data
    distances: np.ndarray
    inside_simple: np.ndarray  # Without hysteresis
    inside_hysteresis: np.ndarray  # With hysteresis
    
    # Events
    entries: List[ZoneEvent] = field(default_factory=list)
    exits: List[ZoneEvent] = field(default_factory=list)
    
    # Metrics
    flip_count_simple: int = 0
    flip_count_hysteresis: int = 0
    time_inside: float = 0.0
    inside_ratio: float = 0.0
    min_distance: float = 0.0
    passed_through: bool = False
    
    # Verdict
    verdict: str = "UNKNOWN"
    verdict_color: str = "gray"
    verdict_explanation: str = ""


def simulate_zone(
    zone: Zone,
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray
) -> ZoneSimulationResult:
    """
    Simulate zone entry/exit detection.
    
    Args:
        zone: Zone definition
        x: East positions (meters)
        y: North positions (meters)
        time: Timestamps (seconds)
        
    Returns:
        ZoneSimulationResult with detection analysis
    """
    n = len(x)
    
    if n == 0:
        return ZoneSimulationResult(
            zone=zone,
            distances=np.array([]),
            inside_simple=np.array([]),
            inside_hysteresis=np.array([])
        )
    
    # Calculate distances to zone center
    distances = np.array([
        distance_2d(xi, yi, zone.center_x, zone.center_y)
        for xi, yi in zip(x, y)
    ])
    
    # Simple inside detection (no hysteresis)
    inside_simple = distances < zone.radius
    
    # Inside detection with hysteresis
    inside_hysteresis = np.zeros(n, dtype=bool)
    current_inside = distances[0] < zone.radius
    inside_hysteresis[0] = current_inside
    
    for i in range(1, n):
        if current_inside:
            # Need to exit beyond radius + hysteresis
            if distances[i] > zone.radius + zone.hysteresis:
                current_inside = False
        else:
            # Need to enter within radius
            if distances[i] < zone.radius:
                current_inside = True
        inside_hysteresis[i] = current_inside
    
    # Count state changes (flips)
    flip_count_simple = int(np.sum(np.diff(inside_simple.astype(int)) != 0))
    flip_count_hysteresis = int(np.sum(np.diff(inside_hysteresis.astype(int)) != 0))
    
    # Find entry/exit events
    entries = []
    exits = []
    
    for i in range(1, n):
        if inside_hysteresis[i] and not inside_hysteresis[i-1]:
            entries.append(ZoneEvent(
                event_type='entry',
                time=time[i],
                index=i,
                distance=distances[i]
            ))
        elif not inside_hysteresis[i] and inside_hysteresis[i-1]:
            exits.append(ZoneEvent(
                event_type='exit',
                time=time[i],
                index=i,
                distance=distances[i]
            ))
    
    # Calculate time inside zone
    time_inside = 0.0
    for i in range(n):
        if inside_hysteresis[i] and i > 0:
            time_inside += time[i] - time[i-1]
    
    total_time = time[-1] - time[0] if n > 1 else 1.0
    inside_ratio = time_inside / total_time if total_time > 0 else 0.0
    
    # Build result
    result = ZoneSimulationResult(
        zone=zone,
        distances=distances,
        inside_simple=inside_simple,
        inside_hysteresis=inside_hysteresis,
        entries=entries,
        exits=exits,
        flip_count_simple=flip_count_simple,
        flip_count_hysteresis=flip_count_hysteresis,
        time_inside=time_inside,
        inside_ratio=inside_ratio,
        min_distance=float(np.min(distances)),
        passed_through=len(entries) > 0 and len(exits) > 0
    )
    
    # Compute verdict
    _compute_zone_verdict(result)
    
    return result


def _compute_zone_verdict(result: ZoneSimulationResult):
    """Determine zone stability verdict."""
    zone = result.zone
    
    # For stationary at zone (inside most of time)
    if result.inside_ratio > 0.8:
        if result.flip_count_hysteresis <= 2:
            result.verdict = "STABLE"
            result.verdict_color = "green"
            result.verdict_explanation = (
                f"Zone is stable. {result.flip_count_hysteresis} state changes "
                f"with {result.inside_ratio*100:.0f}% time inside."
            )
        elif result.flip_count_hysteresis <= 5:
            result.verdict = "NEEDS LARGER RADIUS"
            result.verdict_color = "orange"
            result.verdict_explanation = (
                f"Zone shows {result.flip_count_hysteresis} flickers. "
                f"Consider increasing radius from {zone.radius:.1f}m to {zone.radius + 2:.1f}m."
            )
        else:
            result.verdict = "UNRELIABLE"
            result.verdict_color = "red"
            result.verdict_explanation = (
                f"Zone flickers {result.flip_count_hysteresis} times. "
                f"GPS noise too high for this zone size. Increase radius significantly."
            )
    
    # For pass-through zones
    elif result.passed_through:
        expected_flips = 2  # One entry, one exit
        if abs(result.flip_count_hysteresis - expected_flips) <= 2:
            result.verdict = "STABLE"
            result.verdict_color = "green"
            result.verdict_explanation = (
                f"Clean pass-through detected with {len(result.entries)} entries "
                f"and {len(result.exits)} exits."
            )
        else:
            result.verdict = "NEEDS TUNING"
            result.verdict_color = "orange"
            result.verdict_explanation = (
                f"Multiple entry/exit events ({result.flip_count_hysteresis} total). "
                f"Increase hysteresis from {zone.hysteresis:.1f}m."
            )
    
    # Zone not reached
    else:
        result.verdict = "NOT REACHED"
        result.verdict_color = "gray"
        result.verdict_explanation = (
            f"Vehicle did not enter zone. Minimum distance: {result.min_distance:.1f}m. "
            f"Zone radius: {zone.radius:.1f}m."
        )


def create_zone_from_stop(stop, lat_ref: float, lon_ref: float, zone_id: int = 1) -> Zone:
    """Create a zone from a detected stop."""
    return Zone(
        id=zone_id,
        name=f"Stop {stop.index}",
        center_x=stop.mean_x,
        center_y=stop.mean_y,
        radius=max(stop.pos_max_radius * 2 + 3.0, 5.0),  # Conservative radius
        hysteresis=max(stop.pos_std * 1.5, 2.0),
        lat=stop.lat,
        lon=stop.lon
    )


def create_zone_from_latlon(
    lat: float, 
    lon: float, 
    lat_ref: float, 
    lon_ref: float,
    radius: float = 10.0,
    hysteresis: float = 2.0,
    zone_id: int = 1,
    name: str = "Manual Zone"
) -> Zone:
    """Create a zone from lat/lon coordinates."""
    center_x, center_y = latlon_to_enu(lat, lon, lat_ref, lon_ref)
    
    return Zone(
        id=zone_id,
        name=name,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        hysteresis=hysteresis,
        lat=lat,
        lon=lon
    )


def sweep_hysteresis(
    zone: Zone,
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    hysteresis_values: List[float] = None
) -> List[Dict]:
    """
    Test different hysteresis values to find optimal setting.
    
    Returns list of results with hysteresis and flip count.
    """
    if hysteresis_values is None:
        hysteresis_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    results = []
    
    for h in hysteresis_values:
        test_zone = Zone(
            id=zone.id,
            name=zone.name,
            center_x=zone.center_x,
            center_y=zone.center_y,
            radius=zone.radius,
            hysteresis=h,
            lat=zone.lat,
            lon=zone.lon
        )
        
        sim = simulate_zone(test_zone, x, y, time)
        
        results.append({
            'hysteresis': h,
            'flip_count': sim.flip_count_hysteresis,
            'entries': len(sim.entries),
            'exits': len(sim.exits),
            'stable': sim.flip_count_hysteresis <= 2
        })
    
    return results
