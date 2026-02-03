"""
GPS Quality Analysis module.
Analyzes position noise, jumps, and uncertainty metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .coordinates import distance_2d


@dataclass
class GPSQualityMetrics:
    """GPS quality analysis results."""
    # Reported position uncertainty (from GPS receiver)
    reported_std_mean: Optional[float] = None
    reported_std_max: Optional[float] = None
    reported_std_min: Optional[float] = None
    reported_std_available: bool = False
    
    # Measured noise at stops
    measured_std_mean: Optional[float] = None
    measured_std_max: Optional[float] = None
    measured_radius_mean: Optional[float] = None
    measured_radius_max: Optional[float] = None
    measured_available: bool = False
    
    # Position jumps
    jump_max: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0
    large_jump_count: int = 0
    jumps: np.ndarray = None
    
    # Quality verdict
    verdict: str = "UNKNOWN"
    verdict_color: str = "gray"
    verdict_explanation: str = ""
    
    # Thresholds used
    jump_threshold: float = 5.0


def analyze_gps_quality(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    pos_std_xy: np.ndarray = None,
    stops: List = None,
    jump_threshold: float = 5.0
) -> GPSQualityMetrics:
    """
    Analyze GPS quality from trajectory data.
    
    Args:
        x: East positions (meters)
        y: North positions (meters)
        time: Timestamps (seconds)
        pos_std_xy: Reported position uncertainty (optional)
        stops: List of Stop objects from stop detector (optional)
        jump_threshold: Max allowed jump in meters
        
    Returns:
        GPSQualityMetrics with analysis results
    """
    metrics = GPSQualityMetrics(jump_threshold=jump_threshold)
    
    n = len(x)
    if n < 2:
        metrics.verdict = "INSUFFICIENT DATA"
        metrics.verdict_color = "gray"
        return metrics
    
    # 1. Analyze reported position uncertainty
    if pos_std_xy is not None:
        valid_std = pos_std_xy[~np.isnan(pos_std_xy) & (pos_std_xy > 0)]
        if len(valid_std) > 0:
            metrics.reported_std_available = True
            metrics.reported_std_mean = float(np.mean(valid_std))
            metrics.reported_std_max = float(np.max(valid_std))
            metrics.reported_std_min = float(np.min(valid_std))
    
    # 2. Analyze measured noise at stops
    if stops and len(stops) > 0:
        stop_stds = [s.pos_std for s in stops if s.pos_std is not None]
        stop_radii = [s.pos_max_radius for s in stops if s.pos_max_radius is not None]
        
        if stop_stds:
            metrics.measured_available = True
            metrics.measured_std_mean = float(np.mean(stop_stds))
            metrics.measured_std_max = float(np.max(stop_stds))
            metrics.measured_radius_mean = float(np.mean(stop_radii))
            metrics.measured_radius_max = float(np.max(stop_radii))
    
    # 3. Analyze position jumps
    jumps = np.array([
        distance_2d(x[i], y[i], x[i+1], y[i+1])
        for i in range(n-1)
    ])
    
    metrics.jumps = jumps
    metrics.jump_max = float(np.max(jumps)) if len(jumps) > 0 else 0
    metrics.jump_mean = float(np.mean(jumps)) if len(jumps) > 0 else 0
    metrics.jump_std = float(np.std(jumps)) if len(jumps) > 1 else 0
    metrics.large_jump_count = int(np.sum(jumps > jump_threshold))
    
    # 4. Compute verdict
    _compute_verdict(metrics)
    
    return metrics


def _compute_verdict(metrics: GPSQualityMetrics):
    """Compute GPS quality verdict based on metrics."""
    issues = []
    warnings = []
    
    # Check position jumps
    if metrics.jump_max > metrics.jump_threshold:
        issues.append(f"Large position jumps detected ({metrics.jump_max:.1f}m)")
    elif metrics.jump_max > metrics.jump_threshold / 2:
        warnings.append(f"Moderate position jumps ({metrics.jump_max:.1f}m)")
    
    # Check measured noise
    if metrics.measured_available:
        if metrics.measured_radius_max > 5.0:
            issues.append(f"High GPS noise at stops ({metrics.measured_radius_max:.2f}m)")
        elif metrics.measured_radius_max > 2.0:
            warnings.append(f"Moderate GPS noise at stops ({metrics.measured_radius_max:.2f}m)")
    
    # Check reported uncertainty
    if metrics.reported_std_available:
        if metrics.reported_std_max > 10.0:
            issues.append(f"High reported uncertainty ({metrics.reported_std_max:.1f}m)")
        elif metrics.reported_std_max > 5.0:
            warnings.append(f"Moderate reported uncertainty ({metrics.reported_std_max:.1f}m)")
    
    # Determine verdict
    if issues:
        metrics.verdict = "NOT USABLE"
        metrics.verdict_color = "red"
        metrics.verdict_explanation = "GPS quality issues detected: " + "; ".join(issues)
    elif warnings:
        metrics.verdict = "MARGINAL"
        metrics.verdict_color = "orange"
        metrics.verdict_explanation = "GPS quality is acceptable but has warnings: " + "; ".join(warnings)
    else:
        metrics.verdict = "GOOD FOR ZONE DECISION"
        metrics.verdict_color = "green"
        
        # Build positive explanation
        explanations = []
        if metrics.measured_available:
            explanations.append(f"GPS noise at stops is low ({metrics.measured_std_mean*100:.1f}cm std)")
        if metrics.reported_std_available:
            explanations.append(f"Position uncertainty stable ({metrics.reported_std_mean:.2f}m mean)")
        explanations.append(f"No large position jumps (max {metrics.jump_max:.2f}m)")
        
        metrics.verdict_explanation = "GPS is stable and suitable for zone-based decisions. " + "; ".join(explanations)


def get_jump_histogram_data(jumps: np.ndarray, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram data for position jumps."""
    if len(jumps) == 0:
        return np.array([]), np.array([])
    
    # Limit range to reasonable values
    max_val = min(np.max(jumps) * 1.1, 10.0)
    hist, bin_edges = np.histogram(jumps, bins=bins, range=(0, max_val))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, hist


def compare_reported_vs_measured(
    metrics: GPSQualityMetrics
) -> Dict:
    """
    Generate comparison between reported and measured GPS noise.
    Explains the difference to the user.
    """
    comparison = {
        'reported_available': metrics.reported_std_available,
        'measured_available': metrics.measured_available,
    }
    
    if metrics.reported_std_available and metrics.measured_available:
        ratio = metrics.measured_std_max / metrics.reported_std_mean if metrics.reported_std_mean > 0 else 0
        
        comparison['ratio'] = ratio
        
        if ratio < 0.5:
            comparison['interpretation'] = (
                "Measured noise is LOWER than reported uncertainty. "
                "The GPS receiver is being conservative in its uncertainty estimate. "
                "This is good - the actual position is more reliable than reported."
            )
        elif ratio > 2.0:
            comparison['interpretation'] = (
                "Measured noise is HIGHER than reported uncertainty. "
                "This could indicate multipath or environmental effects not captured "
                "by the receiver's uncertainty model. Use caution with tight zones."
            )
        else:
            comparison['interpretation'] = (
                "Measured noise matches reported uncertainty within expected range. "
                "The GPS uncertainty estimate is reliable for zone configuration."
            )
    elif metrics.measured_available:
        comparison['interpretation'] = (
            "Only measured noise available (no reported uncertainty in data). "
            f"Use measured noise ({metrics.measured_std_max:.2f}m) for zone sizing."
        )
    elif metrics.reported_std_available:
        comparison['interpretation'] = (
            "Only reported uncertainty available (no stops detected for measurement). "
            f"Use reported uncertainty ({metrics.reported_std_max:.2f}m) for zone sizing."
        )
    else:
        comparison['interpretation'] = (
            "Neither reported nor measured GPS noise available. "
            "Using default conservative values for zone configuration."
        )
    
    return comparison
