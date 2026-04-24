"""Harmonized metrics v3.0 - Elevated with causal hypotheses.

Two tiers, each metric annotated with:
  - CAUSAL HYPOTHESIS: why it should predict the success metric
  - MATH: the formula
  - REF: academic/engineering reference

SUCCESS metrics measure final performance (collinearity-free, VIF < 5).
INTERMEDIARY metrics diagnose learning hypotheses.

Collinearity note:
  lap_time and avg_speed are collinear when progress==1.
  avg_speed_centerline is THE success proxy:
    v_eff = speed * (1 - |d_center| / (track_width/2))
  Ref: Leung 2026 (this project)
"""
import numpy as np
from typing import List
from config_loader import CFG


# ================================================================
#  SUCCESS metrics (what the grade depends on)
# ================================================================

def avg_speed_centerline(speeds: np.ndarray, d_center: np.ndarray,
                         track_width: float) -> float:
    """v_eff = speed * (1 - |d_center|/(w/2)).  Mean over episode.
    HYPOTHESIS: directly measures racing performance without lap_time collinearity.
    REF: custom proxy; analogous to F1 effective speed telemetry."""
    half_w = track_width / 2.0
    penalty = np.clip(np.abs(d_center) / half_w, 0.0, 1.0)
    v_eff = speeds * (1.0 - penalty)
    return float(np.mean(v_eff)) if len(v_eff) > 0 else 0.0


def track_progress(final_progress: float) -> float:
    """Fraction of track completed [0,1]. Orthogonal to speed."""
    return float(np.clip(final_progress, 0.0, 1.0))


def compute_success(steps: List[dict], final_progress: float,
                    track_width: float = 0.6) -> dict:
    if not steps:
        return dict(avg_speed_centerline=0.0, track_progress=0.0)
    speeds = np.array([s.get("speed", 0.0) for s in steps])
    d_center = np.array([s.get("distance_from_center", 0.0) for s in steps])
    return dict(
        avg_speed_centerline=avg_speed_centerline(speeds, d_center, track_width),
        track_progress=track_progress(final_progress),
    )


# ================================================================
#  INTERMEDIARY metrics (diagnose learning hypotheses)
# ================================================================

def race_line_adherence(steps: List[dict], track_width: float) -> float:
    """1 - mean(dist_to_raceline / half_w).  [0,1].
    H: closer to racing line -> faster through corners -> higher v_eff.
    REF: Heilmeier et al. min-curvature QP; Ferraresi reward (arXiv:2103.10098)."""
    dists = [s.get("dist_to_raceline", track_width/2) for s in steps]
    if not dists: return 0.0
    return float(np.clip(1.0 - np.mean(dists) / (track_width/2), 0.0, 1.0))


def brake_compliance(steps: List[dict]) -> float:
    """Fraction of braking events inside the brake field.
    H: braking at correct distance before corner -> no lockup, smooth entry -> higher v_eff.
    REF: d_brake = v^2/(2*mu*g), Wikipedia braking distance."""
    braking = [s for s in steps if s.get("is_braking", False)]
    if not braking: return 1.0
    in_field = sum(1 for s in braking if s.get("in_brake_field", False))
    return float(in_field / len(braking))


def corner_speed_error(steps: List[dict]) -> float:
    """1 - mean(|v - v_target|/v_target) in corner zones.
    H: matching optimal speed per corner -> minimal time loss.
    REF: Kapania 2015 two-step velocity profile."""
    cs = [s for s in steps if s.get("in_corner", False)]
    if not cs: return 0.0
    errors = [abs(s.get("speed",0)-s.get("corner_speed_target",1))/max(s.get("corner_speed_target",1),0.01) for s in cs]
    return float(1.0 - np.clip(np.mean(errors), 0.0, 1.0))


def heading_alignment_mean(steps: List[dict]) -> float:
    """(mean(cos(heading_error)) + 1) / 2.  [0,1].
    H: aligned heading -> less steering correction -> smoother -> faster.
    REF: Ferraresi arXiv:2103.10098 velocity-based reward."""
    if not steps: return 0.0
    he = np.array([s.get("heading_error", 0.0) for s in steps])
    return float((np.mean(np.cos(he)) + 1.0) / 2.0)


def smoothness_jerk_rms(steps: List[dict]) -> float:
    """1 - RMS(d(steer)/dt).  [0,1].
    H: smooth inputs -> less tire scrub -> faster + more stable.
    REF: F1 steering integral KPI (Performance Engineering ep.5)."""
    if len(steps) < 2: return 1.0
    steers = np.array([s.get("steering", 0.0) for s in steps])
    jerk = np.diff(steers)
    rms = float(np.sqrt(np.mean(jerk**2)))
    return float(np.clip(1.0 - rms, 0.0, 1.0))


def gg_ellipse_utilisation(steps: List[dict]) -> float:
    """Mean friction-ellipse utilisation [0,1].
    H: using more available grip -> carrying more speed through corners.
    REF: Brach SAE-2011-01-0094; fswiki.us/G-g_Diagram."""
    vals = [s.get("gg_utilisation", 0.0) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


def trail_braking_quality(steps: List[dict]) -> float:
    """Mean trail braking overlap quality [0,1].
    H: trail braking -> better weight transfer at corner entry -> faster rotation
       -> earlier throttle application -> higher exit speed -> higher v_eff.
    REF: Driver61 trail braking; Popometer.io telemetry analysis."""
    vals = [s.get("trail_brake_quality", 0.0) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


def velocity_profile_compliance_mean(steps: List[dict]) -> float:
    """Mean velocity profile compliance [0,1].
    H: matching curvature-optimal speed targets -> minimal time loss per sector.
    REF: Heilmeier min-curvature + Kapania speed profile; GoFynd DeepRacer blog."""
    vals = [s.get("vel_profile_compliance", 0.0) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


def curvature_anticipation_mean(steps: List[dict]) -> float:
    """Mean curvature anticipation score [0,1].
    H: early braking before high-curvature zones -> no panic braking
       -> smoother entry -> faster overall.
    REF: Dynamic Lookahead PPO (arXiv:2603.28625)."""
    vals = [s.get("curv_anticipation", 0.5) for s in steps]
    return float(np.mean(vals)) if vals else 0.5


def waypoint_coverage(steps: List[dict], n_waypoints: int) -> float:
    """Fraction of track waypoints visited.
    H: higher coverage -> more of track explored -> higher progress."""
    if n_waypoints <= 0: return 0.0
    visited = set()
    for s in steps:
        wp = s.get("closest_waypoint")
        if wp is not None: visited.add(wp)
    return float(len(visited) / n_waypoints)


def compute_intermediary(steps: List[dict], n_waypoints: int = 100,
                         track_width: float = 0.6) -> dict:
    return dict(
        race_line_adherence=race_line_adherence(steps, track_width),
        brake_compliance=brake_compliance(steps),
        corner_speed_error=corner_speed_error(steps),
        heading_alignment_mean=heading_alignment_mean(steps),
        smoothness_jerk_rms=smoothness_jerk_rms(steps),
        gg_ellipse_utilisation=gg_ellipse_utilisation(steps),
        trail_braking_quality=trail_braking_quality(steps),
        velocity_profile_compliance=velocity_profile_compliance_mean(steps),
        curvature_anticipation=curvature_anticipation_mean(steps),
        waypoint_coverage=waypoint_coverage(steps, n_waypoints),
    )


def compute_all(steps: List[dict], final_progress: float,
                n_waypoints: int = 100, track_width: float = 0.6) -> dict:
    out = {}
    out.update(compute_success(steps, final_progress, track_width))
    out.update(compute_intermediary(steps, n_waypoints, track_width))
    return out
