"""Harmonized metrics v4.0 — arc-length track_progress, waypoint_lookahead.

Two tiers, each metric annotated with:
  - CAUSAL HYPOTHESIS: why it should predict the success metric
  - MATH: the formula
  - REF: academic/engineering reference

SUCCESS metrics measure final performance (collinearity-free, VIF < 5).
INTERMEDIARY metrics diagnose learning hypotheses.

Key changes vs v3.0:
  1. track_progress: NOW arc-length distance on centreline / total track
     length.  Handles variable-length tracks (bowtie, reinvent, vegas)
     and is NOT collinear with waypoint count.
     REF: Leung (2026, this project); Heilmeier et al. (2020).

  2. waypoint_coverage REMOVED — was collinear with track_progress on
     forward episodes.  REPLACED by waypoint_lookahead:
     How far ahead (in waypoints) did the agent plan?
     = mean of (planning_wp - current_wp) across steps where the agent
       explicitly queried a future waypoint target.
     This is ORTHOGONAL to progress and measures PLANNING HORIZON.
     REF: Hettiarachchi et al. (2024) U-Transformer lookahead;
          Dynamic Lookahead PPO (arXiv:2603.28625).

  3. htm_composite added as intermediary metric — scores agent vs
     HTM oracle (speed, lateral, heading compliance).
     REF: Hawkins & Blakeslee (2004) On Intelligence.

Collinearity note:
  avg_speed_centerline is THE success proxy:
    v_eff = speed * (1 - |d_center| / (track_width/2))
  Ref: Leung 2026 (this project)
"""
import numpy as np
from typing import List, Optional
from config_loader import CFG

_DEFAULT_N_WP       = CFG.get("track", {}).get("n_waypoints", 120)
_DEFAULT_TRACK_LEN  = CFG.get("track", {}).get("track_length_m", None)  # None = compute from WPs


# ================================================================
#  Utility: arc-length track length from waypoints
# ================================================================

def compute_track_length(waypoints) -> float:
    """Sum of Euclidean distances between consecutive waypoints (metres).
    Used when config does not specify track_length_m.
    REF: standard arc-length approximation.
    """
    if waypoints is None or len(waypoints) < 2:
        return 1.0
    wpts = np.array([w[:2] for w in waypoints], dtype=float)
    diffs = np.diff(wpts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1))) + 1e-8


# ================================================================
#  SUCCESS metrics
# ================================================================

def avg_speed_centerline(speeds: np.ndarray, d_center: np.ndarray,
                         track_width: float) -> float:
    """v_eff = speed * (1 - |d_center|/(w/2)).  Mean over episode.
    HYPOTHESIS: directly measures racing performance without lap_time collinearity.
    REF: custom proxy; analogous to F1 effective speed telemetry."""
    half_w  = track_width / 2.0
    penalty = np.clip(np.abs(d_center) / half_w, 0.0, 1.0)
    v_eff   = speeds * (1.0 - penalty)
    return float(np.mean(v_eff)) if len(v_eff) > 0 else 0.0


def track_progress(steps: List[dict],
                   waypoints=None,
                   track_length_m: Optional[float] = None) -> float:
    """Arc-length distance covered along centreline / total track length.

    FORMULA:
        progress = sum_of_segment_lengths_up_to_furthest_wp / track_length

    Each step records 'closest_waypoint'.  We find the furthest WP index
    reached (max, not final, to handle backward drift) and sum the
    arc-length up to that WP from the starting WP.

    WHY NOT final_progress %?  The env's progress field is also
    waypoint-count-based and does NOT account for variable inter-WP spacing
    on tracks like bowtie (some segments 0.05 m, others 0.30 m).

    Falls back to final_progress/100 if no waypoints provided.
    REF: Leung (2026); Heilmeier et al. (2020) arc-length parameterisation.
    """
    if not steps:
        return 0.0
    if waypoints is None or len(waypoints) < 2:
        # fallback: use env progress field from last step
        return float(np.clip(
            steps[-1].get('progress', 0.0) / 100.0, 0.0, 1.0))

    wpts     = np.array([w[:2] for w in waypoints], dtype=float)
    n        = len(wpts)
    seg_lens = np.array([
        np.linalg.norm(wpts[(i+1) % n] - wpts[i]) for i in range(n)
    ])
    total_len = float(np.sum(seg_lens)) if track_length_m is None else track_length_m
    if total_len < 1e-6:
        return 0.0

    visited_wps = [s.get('closest_waypoint') for s in steps
                   if s.get('closest_waypoint') is not None]
    if not visited_wps:
        return float(np.clip(steps[-1].get('progress', 0.0) / 100.0, 0.0, 1.0))

    start_wp = visited_wps[0]
    max_wp   = max(visited_wps, key=lambda w: (w - start_wp) % n)
    n_segs   = (max_wp - start_wp) % n
    arc_dist = float(sum(seg_lens[(start_wp + j) % n] for j in range(n_segs)))
    return float(np.clip(arc_dist / total_len, 0.0, 1.0))


def compute_success(steps: List[dict], final_progress: float,
                    track_width: float = 0.6,
                    waypoints=None,
                    track_length_m: Optional[float] = None) -> dict:
    if not steps:
        return dict(avg_speed_centerline=0.0, track_progress=0.0)
    speeds   = np.array([s.get('speed', 0.0)               for s in steps])
    d_center = np.array([s.get('distance_from_center', 0.0) for s in steps])
    return dict(
        avg_speed_centerline=avg_speed_centerline(speeds, d_center, track_width),
        track_progress=track_progress(steps, waypoints, track_length_m),
    )


# ================================================================
#  INTERMEDIARY metrics
# ================================================================

def race_line_adherence(steps: List[dict], track_width: float) -> float:
    """1 - mean(dist_to_raceline / half_w).  [0,1].
    H: closer to racing line -> faster through corners -> higher v_eff.
    REF: Heilmeier et al. min-curvature QP; Ferraresi (arXiv:2103.10098)."""
    dists = [s.get('dist_to_raceline', track_width/2) for s in steps]
    if not dists:
        return 0.0
    return float(np.clip(1.0 - np.mean(dists) / (track_width/2), 0.0, 1.0))


def brake_compliance(steps: List[dict]) -> float:
    """Fraction of braking events inside the brake field.
    H: braking at correct distance before corner -> smooth entry -> higher v_eff.
    REF: d_brake = v^2/(2*mu*g), Wikipedia braking distance."""
    braking  = [s for s in steps if s.get('is_braking', False)]
    if not braking:
        return 1.0
    in_field = sum(1 for s in braking if s.get('in_brake_field', False))
    return float(in_field / len(braking))


def corner_speed_error(steps: List[dict]) -> float:
    """1 - mean(|v - v_target|/v_target) in corner zones.  [0,1].
    H: matching optimal speed per corner -> minimal time loss.
    REF: Kapania (2015) two-step velocity profile."""
    cs = [s for s in steps if s.get('in_corner', False)]
    if not cs:
        return 0.0
    errors = [
        abs(s.get('speed', 0) - s.get('corner_speed_target', 1))
        / max(s.get('corner_speed_target', 1), 0.01)
        for s in cs
    ]
    return float(1.0 - np.clip(np.mean(errors), 0.0, 1.0))


def heading_alignment_mean(steps: List[dict]) -> float:
    """(mean(cos(heading_error)) + 1) / 2.  [0,1].
    H: aligned heading -> less correction -> smoother -> faster.
    REF: Ferraresi arXiv:2103.10098."""
    if not steps:
        return 0.0
    he = np.array([s.get('heading_error', 0.0) for s in steps])
    return float((np.mean(np.cos(he)) + 1.0) / 2.0)


def smoothness_jerk_rms(steps: List[dict]) -> float:
    """1 - RMS(d(steer)/dt).  [0,1].
    H: smooth inputs -> less tire scrub -> faster + more stable.
    REF: F1 steering integral KPI."""
    if len(steps) < 2:
        return 1.0
    steers = np.array([s.get('steering', 0.0) for s in steps])
    rms    = float(np.sqrt(np.mean(np.diff(steers)**2)))
    return float(np.clip(1.0 - rms, 0.0, 1.0))


def gg_ellipse_utilisation(steps: List[dict]) -> float:
    """Mean friction-ellipse utilisation [0,1].
    H: using more available grip -> carrying more speed through corners.
    REF: Brach SAE-2011-01-0094."""
    vals = [s.get('gg_utilisation', 0.0) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


def trail_braking_quality(steps: List[dict]) -> float:
    """Mean trail braking overlap quality [0,1].
    H: trail braking -> better weight transfer -> faster exit speed.
    REF: Driver61 trail braking."""
    vals = [s.get('trail_brake_quality', 0.0) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


def velocity_profile_compliance_mean(steps: List[dict]) -> float:
    """Mean velocity profile compliance [0,1].
    H: matching curvature-optimal speed targets -> minimal time loss per sector.
    REF: Heilmeier min-curvature + Kapania speed profile."""
    vals = [s.get('vel_profile_compliance', 0.0) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


def curvature_anticipation_mean(steps: List[dict]) -> float:
    """Mean curvature anticipation score [0,1].
    H: early braking before high-curvature zones -> no panic braking -> faster.
    REF: Dynamic Lookahead PPO (arXiv:2603.28625)."""
    vals = [s.get('curv_anticipation', 0.5) for s in steps]
    return float(np.mean(vals)) if vals else 0.5


def waypoint_lookahead(steps: List[dict], n_waypoints: int) -> float:
    """Mean planning horizon: how many waypoints ahead the agent targets.

    FORMULA:
        lookahead_i = (target_wp_i - current_wp_i) mod n_waypoints
        result      = mean(lookahead_i) / n_waypoints   -> [0, 1]

    This is ORTHOGONAL to track_progress on forward episodes:
      - track_progress measures HOW FAR the car got
      - waypoint_lookahead measures HOW FAR AHEAD it was planning

    A value of 0.0 means the agent only looks at its current waypoint
    (reactive, no anticipation).  A value of 0.1+ means it plans
    ~10% of the track ahead (good for corner entry preparation).

    The step record must contain 'target_waypoint' (the WP the race-line
    engine or HTM oracle is steering toward) AND 'closest_waypoint'.
    If 'target_waypoint' is absent, falls back to 0.0.

    REF: Hettiarachchi et al. (2024) U-Transformer lookahead;
         Dynamic Lookahead PPO (arXiv:2603.28625).
    """
    if n_waypoints <= 0 or not steps:
        return 0.0
    horizons = []
    for s in steps:
        cur = s.get('closest_waypoint')
        tgt = s.get('target_waypoint')
        if cur is not None and tgt is not None:
            h = (int(tgt) - int(cur)) % n_waypoints
            horizons.append(h / n_waypoints)
    return float(np.mean(horizons)) if horizons else 0.0


def htm_composite_mean(steps: List[dict]) -> float:
    """Mean HTM oracle compliance score [0,1].
    H: closer to deterministic optimal policy -> better lap time.
    REF: Hawkins & Blakeslee (2004) On Intelligence; htm_reference.py."""
    vals = [s.get('htm_composite', 0.0) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


def icm_bonus_mean(steps: List[dict]) -> float:
    """Mean per-step ICM intrinsic reward.  Diagnostic only.
    High values in failure hotspots confirm curiosity is activating correctly.
    REF: Pathak et al. (2017) ICML."""
    vals = [s.get('icm_bonus', 0.0) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


def compute_intermediary(steps: List[dict],
                         n_waypoints: int = _DEFAULT_N_WP,
                         track_width: float = 0.6) -> dict:
    return dict(
        race_line_adherence         = race_line_adherence(steps, track_width),
        brake_compliance            = brake_compliance(steps),
        corner_speed_error          = corner_speed_error(steps),
        heading_alignment_mean      = heading_alignment_mean(steps),
        smoothness_jerk_rms         = smoothness_jerk_rms(steps),
        gg_ellipse_utilisation      = gg_ellipse_utilisation(steps),
        trail_braking_quality       = trail_braking_quality(steps),
        velocity_profile_compliance = velocity_profile_compliance_mean(steps),
        curvature_anticipation      = curvature_anticipation_mean(steps),
        waypoint_lookahead          = waypoint_lookahead(steps, n_waypoints),  # replaces waypoint_coverage
        htm_composite               = htm_composite_mean(steps),
        icm_bonus                   = icm_bonus_mean(steps),
    )


def compute_all(steps: List[dict],
                final_progress: float,
                n_waypoints: int = _DEFAULT_N_WP,
                track_width: float = 0.6,
                waypoints=None,
                track_length_m: Optional[float] = None) -> dict:
    out = {}
    out.update(compute_success(steps, final_progress, track_width,
                               waypoints, track_length_m))
    out.update(compute_intermediary(steps, n_waypoints, track_width))
    return out
