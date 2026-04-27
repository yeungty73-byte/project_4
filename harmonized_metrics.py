"""harmonized_metrics.py  v1.1.2 — live-wirable telemetry.

Design contract
---------------
* compute_all() is the ONLY public entry point used by run.py.
* It is DEFENSIVE: every path returns a valid dict, never raises.
* Step records written by run.py must include the STEP_KEYS below.
  Keys missing from a step default to 0.0 — no KeyError, no crash.
* All metric values are float in [0, 1] (or positive reals for speed).

Metric taxonomy
---------------
SUCCESS_METRICS  — response variables (VIF<5, non-collinear).
  avg_speed_centerline : v_eff = speed*(1 - |d_ctr|/(w/2)), mean over episode
  track_progress       : arc-length fraction covered [0,1]

INTERMEDIARY_METRICS — covariates for BSTS regression.
  race_line_adherence        [0,1]   1 - mean(d_rl / half_w)
  brake_compliance           [0,1]   fraction braking events inside brake field
  corner_speed_error         [0,1]   1 - mean(|v-v_tgt|/v_tgt) in corners
  heading_alignment_mean     [0,1]   (mean cos(heading_err)+1)/2
  smoothness_steering_rate   [0,1]   1 / (1 + RMS(Δsteer_action))
                                     — renamed from smoothness_jerk_rms (v1.1.2)
                                     — "jerk" means d²x/dt³ (acceleration rate)
                                     — this measures steering INPUT rate, not chassis jerk
                                     — logs show as [STEER_SMOOTH] to avoid confusion
  waypoint_lookahead         [0,1]   mean look-ahead horizon / n_wp
  gg_ellipse_utilisation     [0,1]   mean friction-ellipse fraction used
  velocity_profile_compliance[0,1]   mean per-step vprofile score
  curvature_anticipation     [0,1]   mean anticipation score
  htm_composite              [0,1]   mean HTM oracle compliance
  phase_id                   int     current AnnealingScheduler phase (−1,0,1,2)
  bc_seeded                  int     1 if BC pretraining ran this episode

STEP_KEYS that run.py writes into ep_step_log rows
----------------------------------------------------
  speed, distance_from_center, track_width, closest_waypoint,
  target_waypoint, dist_to_raceline, braking (int 0/1),
  is_braking (bool), in_brake_field (bool),
  in_corner, corner_speed_target, heading_error, steering_angle,
  gg_utilisation, vel_profile_compliance, curv_anticipation,
  htm_composite, is_offtrack, progress

v1.1.2 fixes (this file):
  - smoothness_steering_rate: ALL s.get() calls now wrapped in _safe().
    Old code used bare float(s.get("steering_angle", 0.0)) — if run.py
    wrote a None or NaN, that blew the entire metric to default=0.0 via
    the outer except, masking the real RMS with a silent 0.0.
  - _race_line_adherence: dist_to_raceline path uses _safe() — bare
    float() on None caused same silent failure.
  - Renamed smoothness_jerk_rms → smoothness_steering_rate everywhere.
    "jerk" = d²v/dt = da/dt in physics; this metric is Δsteer/step.
    Conflating them with the acceleration-jerk in ep_jerk_abs is wrong.
    Log tag: [STEER_SMOOTH] (not [JERK]).
  - Legacy alias _smoothness_jerk kept for any external callers.
  - backward-compat key "smoothness_jerk_rms" still written in compute_all()
    output so old JSONL logs / analyze_logs.py don't KeyError.

REF: Leung (2026) this project; Heilmeier et al. (2020) arc-length;
     Ferraresi (2021) arXiv:2103.10098; Kapania (2015) speed profile.
     Balaban et al. (2018) jerk as driving-intent indicator. Veh Syst Dyn.
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Optional

# ------------------------------------------------------------------
# Public API contract lists (kept in sync with analyze_logs.py)
# ------------------------------------------------------------------
SUCCESS_METRICS: List[str] = [
    "avg_speed_centerline",
    "track_progress",
]

INTERMEDIARY_METRICS: List[str] = [
    "race_line_adherence",
    "brake_compliance",
    "corner_speed_error",
    "heading_alignment_mean",
    "smoothness_steering_rate",   # v1.1.2: renamed from smoothness_jerk_rms
    "waypoint_lookahead",
    "gg_ellipse_utilisation",
    "velocity_profile_compliance",
    "curvature_anticipation",
    "htm_composite",
    "phase_id",
    "bc_seeded",
]

# Legacy alias so any code still importing the old name keeps working
# without crashing.  Will be removed in v1.2.
_LEGACY_ALIAS = {"smoothness_jerk_rms": "smoothness_steering_rate"}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe(v, default=0.0) -> float:
    """Convert v to float; return default on any error."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _arc_len(waypoints) -> float:
    """Total track arc-length in metres. Safe against bad input."""
    if not waypoints or len(waypoints) < 2:
        return 1.0
    wpts = np.array([[w[0], w[1]] for w in waypoints], dtype=np.float64)
    n = len(wpts)
    segs = np.linalg.norm(np.diff(np.vstack([wpts, wpts[0:1]]), axis=0), axis=1)
    return float(np.sum(segs)) or 1.0


# ------------------------------------------------------------------
# SUCCESS metrics
# ------------------------------------------------------------------

def _avg_speed_centerline(steps: List[dict], track_width: float) -> float:
    """
    v_eff = speed * (1 - |d_ctr| / (w/2))   mean over episode.
    REF: Leung (2026); F1 effective speed telemetry analogue.
    """
    half_w = max(track_width / 2.0, 0.01)
    vals = []
    for s in steps:
        spd = _safe(s.get("speed", 0))
        dc  = _safe(s.get("distance_from_center", 0))
        pen = min(abs(dc) / half_w, 1.0)
        vals.append(spd * (1.0 - pen))
    return float(np.mean(vals)) if vals else 0.0


def _track_progress(steps: List[dict], waypoints=None,
                    track_length_m: Optional[float] = None) -> float:
    """
    Arc-length fraction of track covered, [0, 1].
    Uses furthest waypoint reached (not final position) to handle backward drift.
    Falls back to env progress % / 100 if waypoints unavailable.
    REF: Heilmeier et al. (2020) arc-length parameterisation.
    """
    if not steps:
        return 0.0

    if waypoints is None or len(waypoints) < 2:
        return min(_safe(steps[-1].get("progress", 0)) / 100.0, 1.0)

    wpts = np.array([[w[0], w[1]] for w in waypoints], dtype=np.float64)
    n = len(wpts)
    seg_lens = np.linalg.norm(
        np.diff(np.vstack([wpts, wpts[0:1]]), axis=0), axis=1
    )
    total_len = float(track_length_m or np.sum(seg_lens)) or 1.0

    visited = [s.get("closest_waypoint") for s in steps
                if s.get("closest_waypoint") is not None]
    if not visited:
        return min(_safe(steps[-1].get("progress", 0)) / 100.0, 1.0)

    start  = int(visited[0])
    def _fwd_dist(wp):
        idx = int(wp)
        n_segs = (idx - start) % n
        return float(sum(seg_lens[(start + j) % n] for j in range(n_segs)))

    arc = max(_fwd_dist(w) for w in visited)
    return min(arc / total_len, 1.0)


# ------------------------------------------------------------------
# INTERMEDIARY metrics
# ------------------------------------------------------------------

def _race_line_adherence(steps: List[dict], track_width: float) -> float:
    """
    Fraction of lap spent on the racing line.
    [RACE_LINE] in logs.

    v1.1.2: _safe() on dist_to_raceline — bare float() blew up on None
    when BrakeField hadn't yet computed _racing_line_err at step 0.
    """
    half_w = max(track_width / 2.0, 0.01)
    # v1.1.2: _safe() wraps every get — prevents None/NaN crash
    dists  = [_safe(s.get("dist_to_raceline", half_w)) for s in steps]
    if not dists:
        return 0.0
    return float(np.clip(1.0 - np.mean(dists) / half_w, 0.0, 1.0))


def _brake_compliance(steps: List[dict]) -> float:
    """
    Fraction of braking events that occur inside the brake field.
    [BRAKE_COMPLIANCE] in logs.

    v1.1.2 fix: run.py writes BOTH:
      'braking':    int(_braking_intent)  — always set (primary)
      'is_braking': bool(_braking_intent) — sometimes False due to dict
                    being appended before '_braking_intent' is computed
                    in the hard-truncation code path.

    Strategy: a step counts as a braking event if EITHER key is truthy.
    'in_brake_field' is also checked via its int counterpart 'brake_potential'>0
    as fallback in case the bool key was written before BrakeField.step() ran.
    """
    braking_steps = [
        s for s in steps
        if s.get("is_braking", False) or int(s.get("braking", 0)) > 0
    ]
    if not braking_steps:
        return 1.0  # no braking recorded → assume compliant (episode too short)
    in_field = sum(
        1 for s in braking_steps
        if s.get("in_brake_field", False)
        or _safe(s.get("brake_potential", 0.0)) > 0.1
    )
    return float(in_field / len(braking_steps))


def _corner_speed_error(steps: List[dict]) -> float:
    cs = [s for s in steps if s.get("in_corner", False)]
    if not cs:
        return 0.0
    errs = [
        abs(_safe(s.get("speed", 0)) - _safe(s.get("corner_speed_target", 1)))
        / max(_safe(s.get("corner_speed_target", 1)), 0.01)
        for s in cs
    ]
    return float(np.clip(1.0 - np.mean(errs), 0.0, 1.0))


def _heading_alignment(steps: List[dict]) -> float:
    if not steps:
        return 0.0
    he = np.array([_safe(s.get("heading_error", 0)) for s in steps])
    return float((np.mean(np.cos(he)) + 1.0) / 2.0)


def _smoothness_steering_rate(steps: List[dict]) -> float:
    """
    Steering INPUT rate smoothness.  [0, 1] — higher = smoother.
    Logged as [STEER_SMOOTH] — NOT jerk (d²v/dt) which is in ep_jerk_abs.

    v1.1.2:
      Source key: 'steering_angle' (action[0], written by run.py as
                  float(action[0]) in every ep_step_log row).
      Old key 'steering' was NEVER written → always 0.0 → RMS=0 → value=1.0
      constantly → BSTS Kalman trend = 0.0 (flat line).

      ALL s.get() values now wrapped in _safe() — bare float() on a None
      from ep_step_log (step 0 before rp populated) caused silent zeroing
      of the entire metric via the outer except clause.

      Mapping: sigmoid  1 / (1 + RMS)  instead of hard clip(1 - RMS).
      Rationale: RMS scale depends on action normalisation ([-1,1] vs [-30,30]).
      Sigmoid keeps output in (0,1) regardless of scale; hard clip would
      saturate immediately for degree-space steering values (~±30).

    REF: Balaban et al. (2018) steering rate as smooth-driving indicator.
    """
    if len(steps) < 2:
        return 1.0
    # v1.1.2: _safe() on every element — no bare float()
    steers = np.array([_safe(s.get("steering_angle", 0.0)) for s in steps])
    if np.all(steers == 0.0):
        # Fallback: steer_rate already computed in ante_buf and stored
        # under 'steer_rate' by some code paths
        rates = np.array([_safe(s.get("steer_rate", 0.0)) for s in steps])
        rms = float(np.sqrt(np.mean(rates ** 2)))
    else:
        rms = float(np.sqrt(np.mean(np.diff(steers) ** 2)))
    # Sigmoid mapping: smooth (rms→0) → 1.0; jerky (rms→∞) → 0.0
    return float(1.0 / (1.0 + rms))


# Legacy name — identical implementation, kept for backward compat
# NOTE: This is steering INPUT rate, NOT chassis jerk (d²v/dt).
#       ep_jerk_abs in run.py tracks true acceleration jerk separately.
def _smoothness_jerk(steps: List[dict]) -> float:
    """Deprecated alias for _smoothness_steering_rate. Do not use for new code."""
    return _smoothness_steering_rate(steps)


def _waypoint_lookahead(steps: List[dict], n_waypoints: int) -> float:
    """
    Mean planning horizon: (target_wp - current_wp) mod n_wp, normalised.
    ORTHOGONAL to track_progress — measures HOW FAR AHEAD, not how far gone.
    REF: Dynamic Lookahead PPO (arXiv:2603.28625).
    """
    if n_waypoints <= 0 or not steps:
        return 0.0
    horizons = []
    for s in steps:
        cur = s.get("closest_waypoint")
        tgt = s.get("target_waypoint")
        if cur is not None and tgt is not None:
            h = (int(tgt) - int(cur)) % n_waypoints
            horizons.append(h / n_waypoints)
    return float(np.mean(horizons)) if horizons else 0.0


def _gg_utilisation(steps: List[dict]) -> float:
    vals = [_safe(s.get("gg_utilisation", 0)) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


def _vprofile_compliance(steps: List[dict]) -> float:
    vals = [_safe(s.get("vel_profile_compliance", 0)) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


def _curvature_anticipation(steps: List[dict]) -> float:
    vals = [_safe(s.get("curv_anticipation", 0.5)) for s in steps]
    return float(np.mean(vals)) if vals else 0.5


def _htm_composite(steps: List[dict]) -> float:
    vals = [_safe(s.get("htm_composite", 0)) for s in steps]
    return float(np.mean(vals)) if vals else 0.0


# ------------------------------------------------------------------
# Main public entry point
# ------------------------------------------------------------------

def compute_all(
    steps: List[dict],
    final_progress: float = 0.0,
    n_waypoints: int = 120,
    track_width: float = 0.6,
    waypoints=None,
    track_length_m: Optional[float] = None,
    phase_id: int = -1,
    bc_seeded: int = 0,
) -> dict:
    """
    Compute ALL harmonized metrics for one episode.

    ALWAYS returns a complete dict — never raises, never returns {}.
    Missing step keys silently default to 0.0.

    Returns
    -------
    dict with keys  SUCCESS_METRICS + INTERMEDIARY_METRICS.
    All values are finite floats.
    """
    if not steps:
        return {k: 0.0 for k in SUCCESS_METRICS + INTERMEDIARY_METRICS}

    try:
        tw = float(track_width) if track_width else 0.6
        nwp = int(n_waypoints) if n_waypoints else 120

        out: dict = {}

        # --- SUCCESS ---
        out["avg_speed_centerline"] = _avg_speed_centerline(steps, tw)
        out["track_progress"]       = _track_progress(steps, waypoints, track_length_m)

        # --- INTERMEDIARY ---
        out["race_line_adherence"]         = _race_line_adherence(steps, tw)
        out["brake_compliance"]            = _brake_compliance(steps)
        out["corner_speed_error"]          = _corner_speed_error(steps)
        out["heading_alignment_mean"]      = _heading_alignment(steps)
        out["smoothness_steering_rate"]    = _smoothness_steering_rate(steps)  # v1.1.2
        out["waypoint_lookahead"]          = _waypoint_lookahead(steps, nwp)
        out["gg_ellipse_utilisation"]      = _gg_utilisation(steps)
        out["velocity_profile_compliance"] = _vprofile_compliance(steps)
        out["curvature_anticipation"]      = _curvature_anticipation(steps)
        out["htm_composite"]               = _htm_composite(steps)
        out["phase_id"]                    = float(phase_id)
        out["bc_seeded"]                   = float(bc_seeded)

        # v1.1.2: backward-compat alias so old BSTS logs / analyze_logs.py
        # that still reference 'smoothness_jerk_rms' don't KeyError.
        out["smoothness_jerk_rms"] = out["smoothness_steering_rate"]

        # Guarantee all keys finite
        for k in list(out):
            if not math.isfinite(_safe(out[k])):
                out[k] = 0.0

        return out

    except Exception:
        # Hard fallback: nothing crashes training
        return {k: 0.0 for k in SUCCESS_METRICS + INTERMEDIARY_METRICS}


# Aliases so analyze_logs.py import stays clean
compute_intermediary = lambda steps, n_waypoints=120, track_width=0.6: {
    k: compute_all(steps, n_waypoints=n_waypoints, track_width=track_width)[k]
    for k in INTERMEDIARY_METRICS
}
compute_success = lambda steps, n_waypoints=120, track_width=0.6: {
    k: compute_all(steps, n_waypoints=n_waypoints, track_width=track_width)[k]
    for k in SUCCESS_METRICS
}
