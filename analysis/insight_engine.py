"""
Insight Engine - Automatic analysis and recommendations.
Computes overall verdict and parameter recommendations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .gps_quality import GPSQualityMetrics
from .stop_detector import StopAnalysis
from .zone_simulator import ZoneSimulationResult


@dataclass
class Recommendations:
    """Recommended configuration parameters."""
    # Zone parameters
    approach_zone_radius: float = 12.0
    decision_zone_radius: float = 5.0
    hysteresis: float = 2.5
    persistence_moving_sec: float = 0.5
    persistence_stopped_sec: float = 0.3
    
    # Source of recommendations
    based_on: str = "default"
    noise_used: float = 2.0


@dataclass
class VerdictResult:
    """Overall analysis verdict."""
    verdict: str  # "SUITABLE", "MARGINAL", "NOT SUITABLE"
    verdict_color: str  # "green", "orange", "red"
    summary: str  # 1-2 sentence explanation
    
    # Component checks
    checks: Dict[str, bool] = None  # name -> passed
    
    # Metrics used
    gps_noise_at_stops: Optional[float] = None
    max_position_jump: Optional[float] = None
    zone_flicker_count: Optional[int] = None


def compute_recommendations(
    gps_metrics: GPSQualityMetrics = None,
    stop_analysis: StopAnalysis = None,
    sample_rate: float = 100.0
) -> Recommendations:
    """
    Compute recommended zone parameters based on analysis.
    
    Args:
        gps_metrics: GPS quality metrics
        stop_analysis: Stop analysis results
        sample_rate: Data sample rate in Hz
        
    Returns:
        Recommendations with explained parameters
    """
    rec = Recommendations()
    
    # Determine noise to use
    if stop_analysis and stop_analysis.worst_gps_noise > 0:
        noise = stop_analysis.worst_gps_noise
        rec.based_on = "measured_at_stops"
    elif gps_metrics and gps_metrics.reported_std_max:
        noise = gps_metrics.reported_std_max
        rec.based_on = "reported_uncertainty"
    else:
        noise = 2.0  # Conservative default
        rec.based_on = "default"
    
    rec.noise_used = noise
    
    # Zone radius calculations
    # Minimum radius should be at least 2x max noise + margin
    min_radius = 2 * noise + 3.0
    
    rec.decision_zone_radius = max(3.0, min_radius)
    rec.approach_zone_radius = rec.decision_zone_radius + 5.0
    
    # Hysteresis should prevent flicker from noise
    rec.hysteresis = max(2.0, noise * 1.5)
    
    # Persistence based on sample rate
    # Want enough samples to be confident, but not so many we miss events
    persistence_samples_moving = max(30, int(noise * 10))
    persistence_samples_stopped = max(15, int(noise * 5))
    
    rec.persistence_moving_sec = persistence_samples_moving / sample_rate
    rec.persistence_stopped_sec = persistence_samples_stopped / sample_rate
    
    return rec


def compute_verdict(
    gps_metrics: GPSQualityMetrics = None,
    stop_analysis: StopAnalysis = None,
    zone_results: List[ZoneSimulationResult] = None
) -> VerdictResult:
    """
    Compute overall verdict for dataset suitability.
    
    Args:
        gps_metrics: GPS quality analysis
        stop_analysis: Stop detection analysis
        zone_results: Zone simulation results (optional)
        
    Returns:
        VerdictResult with overall assessment
    """
    checks = {}
    
    # Check 1: GPS noise bounded
    if gps_metrics:
        noise_ok = gps_metrics.verdict != "NOT USABLE"
        checks["GPS noise bounded"] = noise_ok
    else:
        checks["GPS noise bounded"] = True  # Assume OK if no data
    
    # Check 2: No large position jumps
    if gps_metrics:
        jumps_ok = gps_metrics.large_jump_count == 0
        checks["No large position jumps"] = jumps_ok
    else:
        checks["No large position jumps"] = True
    
    # Check 3: Stable behavior at stops
    if stop_analysis and stop_analysis.stop_count > 0:
        stop_noise_ok = stop_analysis.worst_gps_noise < 2.0  # 2m threshold
        checks["Stable at stops"] = stop_noise_ok
    else:
        checks["Stable at stops"] = True
    
    # Check 4: Zone stability (if simulated)
    if zone_results:
        stable_zones = sum(1 for z in zone_results if z.verdict == "STABLE")
        zone_ok = stable_zones >= len(zone_results) * 0.8  # 80% stable
        checks["Zone detection stable"] = zone_ok
    
    # Determine overall verdict
    all_pass = all(checks.values())
    critical_fail = (
        not checks.get("GPS noise bounded", True) or
        not checks.get("No large position jumps", True)
    )
    
    if critical_fail:
        verdict = "NOT SUITABLE"
        verdict_color = "red"
    elif all_pass:
        verdict = "SUITABLE"
        verdict_color = "green"
    else:
        verdict = "MARGINAL"
        verdict_color = "orange"
    
    # Build summary
    summary = _build_verdict_summary(verdict, gps_metrics, stop_analysis, checks)
    
    result = VerdictResult(
        verdict=verdict,
        verdict_color=verdict_color,
        summary=summary,
        checks=checks
    )
    
    if gps_metrics:
        result.max_position_jump = gps_metrics.jump_max
    if stop_analysis:
        result.gps_noise_at_stops = stop_analysis.worst_gps_noise
    if zone_results:
        result.zone_flicker_count = sum(z.flip_count_hysteresis for z in zone_results)
    
    return result


def _build_verdict_summary(
    verdict: str,
    gps_metrics: GPSQualityMetrics,
    stop_analysis: StopAnalysis,
    checks: Dict[str, bool]
) -> str:
    """Build natural language summary of verdict."""
    
    if verdict == "SUITABLE":
        parts = ["GPS is stable and suitable for zone-based junction decisions."]
        
        if stop_analysis and stop_analysis.stop_count > 0:
            parts.append(
                f"Position noise at stops is {stop_analysis.mean_gps_noise*100:.0f}cm average."
            )
        
        return " ".join(parts)
    
    elif verdict == "MARGINAL":
        failed = [name for name, passed in checks.items() if not passed]
        return (
            f"Dataset has some concerns ({', '.join(failed)}). "
            "Use recommended parameters with caution."
        )
    
    else:  # NOT SUITABLE
        if gps_metrics and gps_metrics.verdict == "NOT USABLE":
            return gps_metrics.verdict_explanation
        
        failed = [name for name, passed in checks.items() if not passed]
        return f"Dataset not suitable for zone decisions. Issues: {', '.join(failed)}."


def get_recommendation_explanation(rec: Recommendations) -> str:
    """Generate explanation text for recommendations."""
    return f"""
**Recommended Zone Configuration**

Based on {rec.based_on.replace('_', ' ')} ({rec.noise_used:.2f}m noise):

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| Approach Zone Radius | {rec.approach_zone_radius:.1f} m | Early detection zone for preparation |
| Decision Zone Radius | {rec.decision_zone_radius:.1f} m | Trigger zone for junction decision |
| Hysteresis | {rec.hysteresis:.1f} m | Exit margin to prevent flicker |
| Persistence (Moving) | {rec.persistence_moving_sec:.2f} s | Time to confirm entry while moving |
| Persistence (Stopped) | {rec.persistence_stopped_sec:.2f} s | Time to confirm entry when stopped |

**Zone Logic Pseudocode:**
```python
# Entry detection
if not inside_zone:
    if distance < ZONE_RADIUS:
        consecutive_inside += 1
        if consecutive_inside >= PERSISTENCE:
            inside_zone = True
            trigger_decision()
    else:
        consecutive_inside = 0

# Exit detection (with hysteresis)
if inside_zone:
    if distance > ZONE_RADIUS + HYSTERESIS:
        inside_zone = False
```
"""


def generate_executive_summary(
    duration: float,
    sample_rate: float,
    total_distance: float,
    stop_analysis: StopAnalysis,
    verdict: VerdictResult
) -> Dict:
    """
    Generate executive summary data for dashboard header.
    
    Returns dict with all summary metrics.
    """
    total_time = stop_analysis.total_stop_time + stop_analysis.total_move_time
    stop_percent = (stop_analysis.total_stop_time / total_time * 100) if total_time > 0 else 0
    move_percent = 100 - stop_percent
    
    return {
        'duration_sec': duration,
        'duration_min': duration / 60,
        'sample_rate': sample_rate,
        'total_distance_m': total_distance,
        'total_distance_km': total_distance / 1000,
        'stop_percent': stop_percent,
        'move_percent': move_percent,
        'stop_count': stop_analysis.stop_count,
        'verdict': verdict.verdict,
        'verdict_color': verdict.verdict_color,
        'verdict_summary': verdict.summary,
    }
