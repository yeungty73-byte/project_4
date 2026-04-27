"""harmonized_metrics.py — v1.4.2
Compliance metrics are CONTINUOUS gradients [0,1], not binary flags.
brake_field_compliance_gradient and race_line_compliance_gradient added to
SUCCESS_METRICS so BSTS Kalman shapes them directly.
REF: Heilmeier et al. (2020); Brayshaw & Harrison (2005); Scott & Varian (2014).

v1.4.2 fixes:
  - smoothness_steering_rate clamped to [0,1]; was going negative (RMS > 1.0)
  - race_line_adherence returns 0.5 when all dist_to_raceline == 0.0
    (race_engine not initialized on 13-29 step episodes -- 0.0 != bad compliance)
  - brake_compliance returns 1.0 (not 0.0) when no brake events exist
    -- the neutral=1.0 was only applied in the exception path before; now
    it is also applied in the happy-path when the active list is empty.
"""
import math
import logging
import numpy as np
from typing import List, Optional

SUCCESS_METRICS: List[str] = [
    "avg_speed_centerline",
    "track_progress",
    "brake_field_compliance_gradient",
    "race_line_compliance_gradient",
]

INTERMEDIARY_METRICS: List[str] = [
    "race_line_adherence",
    "brake_compliance",
    "corner_speed_error",
    "heading_alignment_mean",
    "smoothness_steering_rate",
    "waypoint_lookahead",
    "gg_ellipse_utilisation",
    "velocity_profile_compliance",
    "curvature_anticipation",
    "htm_composite",
    "phase_id",
    "bc_seeded",
]

_HM_LOG = logging.getLogger(__name__)
_LEGACY_ALIAS = {"smoothness_jerk_rms": "smoothness_steering_rate"}


def _safe(v, default=0.0):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _arc_len(waypoints):
    if not waypoints or len(waypoints) < 2:
        return 1.0
    wpts = np.array([[w[0], w[1]] for w in waypoints], dtype=np.float64)
    segs = np.linalg.norm(np.diff(np.vstack([wpts, wpts[0:1]]), axis=0), axis=1)
    return float(np.sum(segs)) or 1.0


# -- SUCCESS ------------------------------------------------------------------

def _avg_speed_centerline(steps, track_width):
    half_w = max(track_width / 2.0, 0.01)
    vals = []
    for s in steps:
        spd = _safe(s.get("speed", 0))
        dc  = _safe(s.get("distance_from_center", 0))
        vals.append(spd * (1.0 - min(abs(dc) / half_w, 1.0)))
    return float(np.mean(vals)) if vals else 0.0


def _track_progress(steps, waypoints=None, track_length_m=None):
    if not steps:
        return 0.0
    if waypoints is None or len(waypoints) < 2:
        return min(_safe(steps[-1].get("progress", 0)) / 100.0, 1.0)
    wpts     = np.array([[w[0], w[1]] for w in waypoints], dtype=np.float64)
    n        = len(wpts)
    seg_lens = np.linalg.norm(np.diff(np.vstack([wpts, wpts[0:1]]), axis=0), axis=1)
    total    = float(track_length_m) if (track_length_m and float(track_length_m) > 1.0 and float(track_length_m) < 99.0) else float(np.sum(seg_lens))
    total    = max(total, 1.0)
    visited  = [s.get("closest_waypoint") for s in steps if s.get("closest_waypoint") is not None]
    if not visited:
        return min(_safe(steps[-1].get("progress", 0)) / 100.0, 1.0)
    start = int(visited[0])
    def _fwd(wp):
        idx    = int(wp)
        n_segs = (idx - start) % n
        return float(sum(seg_lens[(start + j) % n] for j in range(n_segs)))
    return min(max(_fwd(w) for w in visited) / total, 1.0)


def _brake_field_compliance_gradient(steps):
    """Mean continuous compliance_gradient from BrakeField.step() v1.2.0.
    v1.4.2: returns 1.0 (vacuously compliant) when no brake-field steps exist.
    """
    active = []
    for s in steps:
        cg = s.get("compliance_gradient")
        if cg is not None:
            active.append(_safe(cg, 1.0))
            continue
        if bool(s.get("in_brake_field", False)):
            is_b = bool(s.get("braking", 0)) or bool(s.get("is_braking", False))
            active.append(1.0 if is_b else 0.0)
    return float(np.clip(np.mean(active), 0.0, 1.0)) if active else 1.0


def _race_line_compliance_gradient(steps, track_width):
    """Mean race_line_compliance_gradient from race_engine.get_combined_reward() v1.4.3.

    Priority resolution per step:
      1. If 'race_line_compliance_gradient' key present (engine running): use it directly.
      2. If dist_to_raceline == 0.0 AND engine NOT running: neutral 0.5
         (race_engine not initialized on 13-29 step episodes — 0 ≠ perfect compliance).
      3. If dist_to_raceline == 0.0 AND engine IS running: 1.0 (car is on the line).
      4. If dist_to_raceline > 0.0: Gaussian penalty exp(-0.5*(d/sigma)^2).
      5. If dist_to_raceline < 0.0 (sentinel/missing): append 0.5 neutral.
         PREVIOUS BUG: `continue` silently DROPPED these steps from grads[],
         shrinking the denominator and inflating the mean toward the remaining steps.
         FIX v1.4.3: append 0.5 neutral so every step contributes to the mean.

    REF: Heilmeier et al. (2020) -- race-line compliance as continuous gradient.
    REF: AWS DeepRacer docs -- dist_to_raceline not available until race_engine init.
    """
    grads = []
    has_engine_grad = False
    for s in steps:
        rlcg = s.get("race_line_compliance_gradient")
        if rlcg is not None:
            grads.append(_safe(rlcg, 0.5))
            has_engine_grad = True
            continue
        half_w = max(track_width / 2.0, 0.01)
        d      = _safe(s.get("dist_to_raceline", -1.0))
        if d < 0.0:
            # v1.4.3 FIX: missing/sentinel dist → neutral 0.5, NOT silently dropped.
            # Dropping shrank the denominator and inflated mean toward remaining steps.
            grads.append(0.5)
        elif d == 0.0:
            # v1.4.3: d==0 with engine running = car is exactly on line → 1.0.
            # d==0 without engine (all zeros from uninit) → neutral 0.5.
            grads.append(1.0 if has_engine_grad else 0.5)
        else:
            grads.append(float(math.exp(-0.5 * (d / (half_w * 0.5)) ** 2)))
    if not grads:
        return 0.5
    return float(np.clip(np.mean(grads), 0.0, 1.0))


# -- INTERMEDIARY -------------------------------------------------------------

def _race_line_adherence(steps, track_width):
    return _race_line_compliance_gradient(steps, track_width)


def _brake_compliance(steps):
    return _brake_field_compliance_gradient(steps)


def _corner_speed_error(steps):
    errs = []
    for s in steps:
        spd = _safe(s.get("speed", 0))
        tgt = _safe(s.get("corner_speed_target", spd))
        in_c = s.get("in_corner", False) or s.get("is_turn", False)
        if tgt > 0 and in_c:
            errs.append(abs(spd - tgt) / max(tgt, 0.1))
    return float(np.clip(1.0 - np.mean(errs), 0.0, 1.0)) if errs else 0.5


def _heading_alignment(steps):
    vals = [min(abs(_safe(s.get("heading_diff", math.pi))), math.pi) for s in steps]
    return float(np.clip(1.0 - np.mean(vals) / math.pi, 0.0, 1.0)) if vals else 0.0


def _smoothness_steering_rate(steps):
    """v1.4.2 FIX: clamped to [0,1] via sigmoid normalized by 30 degrees.
    Was: 1.0 - rms which goes negative when rms > 1.0.
    Steering angles in rp['steering_angle'] are in degrees; step-to-step
    diffs of 5-15 deg are normal and produce rms > 1.0.

    New formula: 1.0 / (1.0 + rms/30) -- sigmoid, always in [0,1]:
      rms=0   -> 1.0 (perfectly smooth)
      rms=30  -> 0.5 (one full max-steer change per step)
      rms=60  -> 0.33 (extreme oscillation)
    """
    steers = []
    for s in steps:
        val = s.get("steering_angle")
        if val is None:
            val = s.get("steering")
        steers.append(_safe(val, 0.0))
    if len(steers) < 2:
        return 1.0
    diffs = [abs(steers[i] - steers[i - 1]) for i in range(1, len(steers))]
    rms = float(np.sqrt(np.mean(np.array(diffs) ** 2)))
    return float(np.clip(1.0 / (1.0 + rms / 30.0), 0.0, 1.0))


def _waypoint_lookahead(steps, n_waypoints):
    indices = [s.get("closest_waypoint") for s in steps if s.get("closest_waypoint") is not None]
    if len(indices) < 2:
        return 0.5
    jumps = [(indices[i + 1] - indices[i]) % n_waypoints for i in range(len(indices) - 1)]
    return float(np.clip(np.mean([min(j, 5) / 5.0 for j in jumps]), 0.0, 1.0))


def _gg_utilisation(steps):
    utils = []
    for s in steps:
        spd   = _safe(s.get("speed", 0))
        decel = _safe(s.get("accel", 0))
        steer_v = s.get("steering_angle")
        if steer_v is None: steer_v = s.get("steering", 0)
        steer = _safe(steer_v, 0)
        lat   = (spd ** 2) * math.sin(math.radians(abs(steer))) / max(spd, 0.1)
        g_use = math.sqrt(lat ** 2 + abs(decel) ** 2) / 9.81
        utils.append(min(g_use, 1.5))
    return float(np.clip(np.mean(utils), 0.0, 1.0)) if utils else 0.0


def _vprofile_compliance(steps):
    ok = total = 0
    for s in steps:
        tgt = _safe(s.get("corner_speed_target", 0))
        spd = _safe(s.get("speed", 0))
        if tgt > 0 and tgt != spd:
            total += 1
            if abs(spd - tgt) / max(tgt, 0.01) < 0.20:
                ok += 1
    return float(ok / max(total, 1))


def _curvature_anticipation(steps):
    slowing = corners = 0
    for i in range(1, len(steps)):
        cur  = steps[i].get("in_corner", steps[i].get("is_turn", False))
        prev = steps[i - 1].get("in_corner", steps[i - 1].get("is_turn", False))
        if cur and not prev:
            corners += 1
            if _safe(steps[i].get("speed", 0)) < _safe(steps[i - 1].get("speed", 1)):
                slowing += 1
    return float(slowing / max(corners, 1))


def _htm_composite(steps):
    headings = [_safe(s.get("heading_diff", 0)) for s in steps]
    if not headings:
        return 0.5
    return float(np.clip(1.0 - float(np.std(headings)) / math.pi, 0.0, 1.0))


# -- Public API ---------------------------------------------------------------

def compute_all(steps, final_progress=0.0, n_waypoints=120, track_width=0.6,
                waypoints=None, track_length_m=None, phase_id=-1, bc_seeded=0):
    """Always returns complete dict with SUCCESS_METRICS + INTERMEDIARY_METRICS.
    Never raises. All values finite floats.

    v1.4.2: neutral defaults in HAPPY PATH (not only on exception).
    race_line_adherence=0.5, brake_compliance=1.0, smoothness=[0,1] sigmoid.
    """
    _NEUTRALS = {
        "race_line_adherence":             0.5,
        "brake_compliance":                1.0,
        "brake_field_compliance_gradient": 1.0,
        "race_line_compliance_gradient":   0.5,
        # v1.1.4c: combined neutral = 0.5*1.0 + 0.5*0.5 = 0.75
        # (vacuously compliant brake field + neutral race line)
        "compliance_gradient":             0.75,
        "smoothness_steering_rate":        1.0,
        "corner_speed_error":              0.5,
    }
    if not steps:
        _neutral = {k: 0.0 for k in SUCCESS_METRICS + INTERMEDIARY_METRICS}
        _neutral.update(_NEUTRALS)
        return _neutral
    try:
        tw  = float(track_width) if track_width else 0.6
        nwp = int(n_waypoints)   if n_waypoints else 120
        out = {}
        out["avg_speed_centerline"]            = _avg_speed_centerline(steps, tw)
        out["track_progress"]                  = _track_progress(steps, waypoints, track_length_m)
        out["brake_field_compliance_gradient"] = _brake_field_compliance_gradient(steps)
        out["race_line_compliance_gradient"]   = _race_line_compliance_gradient(steps, tw)
        out["race_line_adherence"]             = _race_line_adherence(steps, tw)
        out["brake_compliance"]                = _brake_compliance(steps)
        # v1.1.4c FIX-5: compute combined compliance_gradient here so
        # harmonized_metrics itself produces the key that bsts_metrics.update() expects.
        # Formula must match per-step formula in run.py (L2627) and episode-end (L3082).
        out["compliance_gradient"]             = float(
            0.5 * out["brake_field_compliance_gradient"]
            + 0.5 * out["race_line_compliance_gradient"]
        )
        out["corner_speed_error"]              = _corner_speed_error(steps)
        out["heading_alignment_mean"]          = _heading_alignment(steps)
        out["smoothness_steering_rate"]        = _smoothness_steering_rate(steps)
        out["waypoint_lookahead"]              = _waypoint_lookahead(steps, nwp)
        out["gg_ellipse_utilisation"]          = _gg_utilisation(steps)
        out["velocity_profile_compliance"]     = _vprofile_compliance(steps)
        out["curvature_anticipation"]          = _curvature_anticipation(steps)
        out["htm_composite"]                   = _htm_composite(steps)
        out["phase_id"]                        = float(phase_id)
        out["bc_seeded"]                       = float(bc_seeded)
        out["smoothness_jerk_rms"]             = out["smoothness_steering_rate"]  # compat alias
        for k in list(out):
            if not math.isfinite(_safe(out[k])):
                out[k] = _NEUTRALS.get(k, 0.0)
        return out
    except Exception as _hm_exc:
        _HM_LOG.exception("[HM] compute_all exception -- returning neutral defaults")
        _neutral = {k: 0.0 for k in SUCCESS_METRICS + INTERMEDIARY_METRICS}
        _neutral.update(_NEUTRALS)
        return _neutral


compute_intermediary = lambda steps, n_waypoints=120, track_width=0.6: {
    k: compute_all(steps, n_waypoints=n_waypoints, track_width=track_width)[k]
    for k in INTERMEDIARY_METRICS
}
compute_success = lambda steps, n_waypoints=120, track_width=0.6: {
    k: compute_all(steps, n_waypoints=n_waypoints, track_width=track_width)[k]
    for k in SUCCESS_METRICS
}
