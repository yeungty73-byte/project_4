"""analyze_logs.py  v5.0 — BSTS telemetry, fully wired.

Public symbols consumed by run.py
----------------------------------
  BSTSKalmanFilter         — online Kalman filter; one instance per success metric
  extract_intermediary_metrics(ep_dict)           -> dict[str, float]
  episode_summary_metrics(ep_dict, intermediary)  -> dict[str, float]
  bsts_compliance_report(matrix)                  -> dict
  compute_anneal_recommendations(bsts_rpt, matrix)-> dict
  INTERMEDIARY_METRICS                            — list[str]
  SUCCESS_METRICS                                 — list[str]
  compute_optimal_raceline(waypoints)             -> dict
  score_raceline_compliance(episodes, race_line)  -> dict

What changed in v5.0
---------------------
  1. INTERMEDIARY_METRICS is now imported from harmonized_metrics —
     single source of truth; no more list-skew between files.
  2. extract_intermediary_metrics() calls harmonized_metrics.compute_all()
     with the step keys that run.py ACTUALLY writes (speed, distance_from_center,
     heading, steering, progress, closest_waypoints, etc.)  — and translates
     them so every metric gets real signal instead of zeroes.
  3. episode_summary_metrics() output is now the EXACT shape that
     BSTSFeedback.update() and bsts_row consumes (keys match).  run.py
     used to call these and discard the result — v5 fixes the call site too.
  4. bsts_compliance_report() now returns per-metric EMA trends and betas
     in a flat dict that BSTSFeedback can consume directly.
  5. compute_anneal_recommendations() maps degrading-metric drivers to
     *actual reward weight keys* used in AnnealingScheduler, not generic strings.

Step-key translation table (run.py log key -> harmonized_metrics key)
----------------------------------------------------------------------
  speed                   -> speed
  distance_from_center    -> distance_from_center
  heading_diff            -> heading_error       (radians)
  steering                -> steering
  progress                -> progress
  closest_waypoints[1]    -> closest_waypoint
  racing_line_offset*tw/2 -> dist_to_raceline    (metres)
  braking (bool)          -> is_braking
  in_brake_field (bool)   -> in_brake_field
  in_corner (bool)        -> in_corner
  corner_speed_target     -> corner_speed_target
  safe_speed_ratio        -> vel_profile_compliance
  curv_anticipation       -> curv_anticipation
  htm_composite           -> htm_composite

REF:
  Scott & Varian (2014) BSTS for causal inference
  Durbin & Koopman (2012) Time Series Analysis by State Space Methods
  Kalman (1960) J. Basic Eng. 82(1):35-45
"""
from __future__ import annotations
import os, sys, json, csv, glob, math
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from harmonized_metrics import (
    compute_all as _hm_compute_all,
    INTERMEDIARY_METRICS,
    SUCCESS_METRICS,
)

# ================================================================
# 0.  Step-record translation
#     run.py writes ep_step_log rows with ITS OWN key names.
#     We translate before passing to harmonized_metrics.
# ================================================================

def _translate_step(s: dict, track_width: float = 0.6) -> dict:
    """
    Translate a run.py ep_step_log row into the harmonized_metrics step format.
    Safe — all missing keys default to 0 / False.
    """
    tw = float(track_width) or 0.6
    half_w = tw / 2.0

    # closest_waypoint: run.py stores list [prev, next]
    cw = s.get("closest_waypoints") or s.get("closest_waypoint")
    if isinstance(cw, (list, tuple)):
        cw = cw[1] if len(cw) > 1 else cw[0]

    # dist_to_raceline: run.py stores racing_line_offset in [-1,1] x half_w
    rl_off = float(s.get("racing_line_offset", 0) or 0)
    dist_rl = abs(rl_off) * half_w   # metres

    # heading_error: run.py stores heading_diff in degrees or heading_error in rad
    he_raw = s.get("heading_error") or s.get("heading_diff", 0)
    try:
        he = float(he_raw)
    except Exception:
        he = 0.0
    # If magnitude > 2*pi assume degrees, convert
    if abs(he) > 2 * math.pi:
        he = math.radians(he)

    # vel_profile_compliance from safe_speed_ratio (1.0 = perfect)
    ssr = float(s.get("safe_speed_ratio", 1.0) or 1.0)
    vpc = float(np.clip(1.0 - abs(ssr - 1.0), 0.0, 1.0))

    return dict(
        speed                    = float(s.get("speed", 0) or 0),
        distance_from_center     = float(s.get("distance_from_center", 0) or 0),
        closest_waypoint         = int(cw) if cw is not None else 0,
        target_waypoint          = s.get("target_waypoint"),
        dist_to_raceline         = dist_rl,
        is_braking               = bool(s.get("braking") or s.get("is_braking", False)),
        in_brake_field           = bool(s.get("in_brake_field", False)),
        in_corner                = bool(s.get("in_corner", False)),
        corner_speed_target      = float(s.get("corner_speed_target", 1.5) or 1.5),
        heading_error            = he,
        steering                 = float(s.get("steering", 0) or 0),
        gg_utilisation           = float(s.get("gg_utilisation", 0) or 0),
        vel_profile_compliance   = vpc,
        curv_anticipation        = float(s.get("curv_anticipation", 0.5) or 0.5),
        htm_composite            = float(s.get("htm_composite", 0) or 0),
        progress                 = float(s.get("progress", 0) or 0),
        is_offtrack              = bool(s.get("is_offtrack") or s.get("all_wheels_on_track") is False),
    )


# ================================================================
# I.  Data Loading
# ================================================================

def load_jsonl_episodes(log_dir: str) -> List[dict]:
    """Load all JSONL episode logs from a results directory."""
    eps, files = [], []
    for p in [os.path.join(log_dir, '*.jsonl'),
               os.path.join(log_dir, 'episodes', '*.jsonl')]:
        files.extend(sorted(glob.glob(p)))
    for f in sorted(set(files)):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        eps.append(json.loads(line))
                    except Exception:
                        pass
    return eps


def load_bsts_csv(log_dir: str) -> List[dict]:
    """Load bsts_metrics.csv rows."""
    p = os.path.join(log_dir, 'bsts_metrics.csv')
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except (ValueError, TypeError):
                    out[k] = v
            rows.append(out)
    return rows


# ================================================================
# II.  Intermediary extraction  (WIRED to run.py step format)
# ================================================================

def extract_intermediary_metrics(ep: dict) -> dict:
    """
    Compute INTERMEDIARY_METRICS for one episode dict.

    Accepts the JSONL episode format written by run.py:
      ep['steps']       = ep_step_log rows  (run.py format)
      ep['track_width'] = float metres
      ep['n_waypoints'] = int
      ep['waypoints']   = list of [x, y] (optional)

    Returns dict[metric_name -> float], all keys from INTERMEDIARY_METRICS.
    """
    raw_steps  = ep.get('steps') or ep.get('trajectory') or []
    tw         = float(ep.get('track_width', 0.6) or 0.6)
    n_wp       = int(ep.get('n_waypoints', 120) or 120)
    waypoints  = ep.get('waypoints')
    phase_id   = int(ep.get('phase_id', -1))
    bc_seeded  = int(ep.get('bc_seeded', 0))

    if len(raw_steps) < 2:
        return {m: 0.0 for m in INTERMEDIARY_METRICS}

    # Translate every step to harmonized_metrics format
    steps = [_translate_step(s, tw) for s in raw_steps]

    all_metrics = _hm_compute_all(
        steps,
        n_waypoints    = n_wp,
        track_width    = tw,
        waypoints      = waypoints,
        phase_id       = phase_id,
        bc_seeded      = bc_seeded,
    )
    return {m: float(all_metrics.get(m, 0.0)) for m in INTERMEDIARY_METRICS}


def episode_summary_metrics(ep: dict, intermediary: dict) -> dict:
    """
    Merge success metrics + intermediary metrics into one flat dict.

    Keys returned match what run.py's bsts_row and BSTSFeedback.update() expect.
    """
    raw_steps = ep.get('steps') or ep.get('trajectory') or []
    tw        = float(ep.get('track_width', 0.6) or 0.6)
    n_wp      = int(ep.get('n_waypoints', 120) or 120)
    waypoints = ep.get('waypoints')

    steps = [_translate_step(s, tw) for s in raw_steps]

    all_metrics = _hm_compute_all(
        steps,
        n_waypoints = n_wp,
        track_width = tw,
        waypoints   = waypoints,
    )

    summary: dict = {k: float(all_metrics.get(k, 0.0)) for k in SUCCESS_METRICS}
    for k, v in intermediary.items():
        summary[k] = float(v) if np.isscalar(v) else 0.0

    # Legacy scalars for dashboard back-compat
    if raw_steps:
        speeds = [float(s.get('speed', 0)) for s in raw_steps]
        summary['_legacy_avg_speed']     = float(np.mean(speeds)) if speeds else 0.0
        on_track = [bool(s.get('all_wheels_on_track', True)) for s in raw_steps]
        summary['_legacy_off_track_rate']= 1.0 - float(np.mean(on_track))
    summary['_legacy_crash_flag'] = float(
        1 if ep.get('crashed') or ep.get('termination_reason') == 'crashed' else 0
    )
    summary['_legacy_completion_pct'] = float(
        ep.get('completion_pct') or ep.get('progress') or 0
    )
    return summary


# ================================================================
# III.  Kalman-Filter BSTS  (unchanged structure, tuned defaults)
# ================================================================

class BSTSKalmanFilter:
    """Online Bayesian Structural Time Series via Kalman Filter.

    State: [level, trend, s_1..s_{S-1}, beta_1..beta_p]
    Observation: y_t = level + season + X_t @ beta + eps

    REF: Kalman (1960) J. Basic Eng. 82:35-45;
         Durbin & Koopman (2012) Time Series Analysis.
    """

    def __init__(self, seasonal_period: int = 8, n_regressors: int = 0,
                 sigma_obs: float = 1.0, sigma_level: float = 0.1,
                 sigma_trend: float = 0.01, sigma_season: float = 0.05,
                 sigma_beta: float = 0.01):
        self.S  = max(int(seasonal_period), 2)
        self.p  = max(int(n_regressors), 0)
        self.d  = 2 + (self.S - 1) + self.p
        self.state = np.zeros(self.d)
        self.P     = np.eye(self.d) * 10.0
        self.sigma_obs    = sigma_obs
        self.sigma_level  = sigma_level
        self.sigma_trend  = sigma_trend
        self.sigma_season = sigma_season
        self.sigma_beta   = sigma_beta
        self._build()
        self._t   = 0         # step counter
        self.last = {}        # last update output

    def _build(self):
        d, S, p = self.d, self.S, self.p
        T = np.zeros((d, d))
        T[0, 0] = 1.0;  T[0, 1] = 1.0   # level
        T[1, 1] = 1.0                     # trend
        if S > 1:
            T[2, 2:2+S-1] = -1.0
            for i in range(3, 2 + S - 1):
                T[i, i-1] = 1.0
        for i in range(p):
            idx = 2 + S - 1 + i
            T[idx, idx] = 1.0
        Q = np.zeros((d, d))
        Q[0, 0] = self.sigma_level  ** 2
        Q[1, 1] = self.sigma_trend  ** 2
        if S > 1:
            Q[2, 2] = self.sigma_season ** 2
        for i in range(p):
            Q[2 + S - 1 + i, 2 + S - 1 + i] = self.sigma_beta ** 2
        self.T = T
        self.Q = Q
        self.R = self.sigma_obs ** 2

    def _Z(self, X_t=None):
        Z = np.zeros(self.d)
        Z[0] = 1.0
        if self.S > 1:
            Z[2] = 1.0
        if X_t is not None and self.p > 0:
            Z[2 + self.S - 1: 2 + self.S - 1 + self.p] = \
                np.asarray(X_t, dtype=float)[:self.p]
        return Z

    def update_online(self, y: float, X_t=None) -> dict:
        """One-step predict+update.  Call once per episode."""
        self._t += 1
        # Predict
        self.state = self.T @ self.state
        self.P     = self.T @ self.P @ self.T.T + self.Q
        # Update
        Z = self._Z(X_t)
        y_pred   = float(Z @ self.state)
        residual = float(y) - y_pred
        S        = float(Z @ self.P @ Z) + self.R
        K        = self.P @ Z / (S + 1e-12)
        self.state = self.state + K * residual
        self.P     = self.P - np.outer(K, Z) @ self.P
        level  = float(self.state[0])
        trend  = float(self.state[1])
        season = float(self.state[2]) if self.S > 1 else 0.0
        betas  = self.state[2 + self.S - 1:].tolist() if self.p > 0 else []
        self.last = dict(
            t       = self._t,
            y       = float(y),
            y_pred  = y_pred,
            residual= residual,
            level   = level,
            trend   = trend,
            season  = season,
            betas   = betas,
        )
        return self.last

    def filter_series(self, y: np.ndarray,
                      X: Optional[np.ndarray] = None) -> dict:
        """Batch Kalman filter over a complete series."""
        T_len = len(y)
        levels, trends, seasonals = (np.zeros(T_len) for _ in range(3))
        residuals, predictions    = (np.zeros(T_len) for _ in range(2))
        betas = np.zeros((T_len, self.p)) if self.p > 0 else None
        for t in range(T_len):
            out = self.update_online(float(y[t]),
                                     X[t] if X is not None else None)
            levels[t]     = out['level']
            trends[t]     = out['trend']
            seasonals[t]  = out['season']
            residuals[t]  = out['residual']
            predictions[t]= out['y_pred']
            if betas is not None:
                betas[t] = out['betas']
        result = dict(levels=levels, trends=trends, seasonals=seasonals,
                      residuals=residuals, predictions=predictions)
        if betas is not None:
            result['betas'] = betas
        return result


# ================================================================
# IV.  Race Line Geometry
# ================================================================

def compute_optimal_raceline(waypoints: List) -> dict:
    """Curvature + speed profile from waypoint geometry.
    REF: Heilmeier et al. (2020); Kapania (2015).
    """
    if not waypoints or len(waypoints) < 4:
        return dict(race_line=waypoints or [], curvatures=[], max_speeds=[],
                    brake_points=[], brake_zone_integral=0.0, normals=[])
    pts = np.array([[w[0], w[1]] for w in waypoints], dtype=float)
    n   = len(pts)
    dx  = np.gradient(pts[:, 0])
    dy  = np.gradient(pts[:, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    denom      = (dx**2 + dy**2)**1.5 + 1e-8
    curvatures = np.abs(dx * d2y - dy * d2x) / denom
    a_lat      = 4.0
    max_speeds = np.clip(np.sqrt(a_lat / (curvatures + 1e-6)), 0, 4.0)
    seg_ds     = np.append(
        np.linalg.norm(np.diff(pts, axis=0), axis=1), 0.01)
    a_brake    = 3.0
    bl         = max_speeds.copy()
    for i in range(n - 2, -1, -1):
        bl[i] = min(bl[i], math.sqrt(bl[i+1]**2 + 2*a_brake*seg_ds[i]))
    brake_mask  = bl < max_speeds * 0.95
    brake_pts   = np.where(brake_mask)[0].tolist()
    mag         = np.sqrt(dx**2 + dy**2) + 1e-8
    nx, ny      = -dy / mag, dx / mag
    return dict(
        race_line          = pts.tolist(),
        curvatures         = curvatures.tolist(),
        max_speeds         = bl.tolist(),
        brake_points       = brake_pts,
        brake_zone_integral= float(np.sum(seg_ds[brake_mask])),
        normals            = list(zip(nx.tolist(), ny.tolist())),
    )


def score_raceline_compliance(episodes: List[dict], race_line: dict) -> dict:
    """Score v_perp at barrier and lateral deviation.  REF: Leung 2026."""
    if not episodes or not race_line.get('race_line'):
        return dict(avg_perp_v=0, avg_deviation=0, brake_efficiency=0,
                    per_ep_perp=[])
    rl  = np.array(race_line['race_line'])
    nrm = np.array(race_line.get('normals', []))
    all_perp, all_dev, all_beff, per_ep = [], [], [], []
    for ep in episodes:
        steps = ep.get('steps') or ep.get('trajectory') or []
        if len(steps) < 3:
            continue
        xs  = np.array([s.get('x', 0) for s in steps])
        ys  = np.array([s.get('y', 0) for s in steps])
        spd = np.array([s.get('speed', 0) for s in steps])
        hdg = np.deg2rad([s.get('heading', 0) for s in steps])
        vx, vy = spd * np.cos(hdg), spd * np.sin(hdg)
        ep_perp, ep_dev = [], []
        for i in range(len(steps)):
            d   = np.sqrt((rl[:, 0]-xs[i])**2 + (rl[:, 1]-ys[i])**2)
            ci  = int(np.argmin(d))
            ep_dev.append(float(d[ci]))
            if ci < len(nrm):
                ep_perp.append(abs(float(vx[i]*nrm[ci][0] + vy[i]*nrm[ci][1])))
        if ep_perp:
            all_perp.extend(ep_perp)
            per_ep.append(float(np.mean(ep_perp)))
        all_dev.extend(ep_dev)
        th  = [s.get('throttle', 1) for s in steps]
        act = sum(1 for t in th if float(t) < 0.5)
        thr = len(race_line.get('brake_points', []))
        if act > 0:
            all_beff.append(min(1.0, thr / act))
    return dict(
        avg_perp_v     = float(np.mean(all_perp)) if all_perp else 0.0,
        avg_deviation  = float(np.mean(all_dev))  if all_dev  else 0.0,
        brake_efficiency= float(np.mean(all_beff)) if all_beff else 0.0,
        per_ep_perp    = per_ep,
    )


# ================================================================
# V.  BSTS Compliance Report
# ================================================================

def bsts_compliance_report(matrix: List[dict]) -> dict:
    """
    Run BSTS Kalman decomposition on each success metric.

    Returns a flat dict that BSTSFeedback.kftrends and .kfbetas
    can consume directly, plus a 'recommendations' list.

    Output keys
    -----------
    trends           : dict[metric -> float]   final Kalman trend slope
    betas_final      : dict[metric -> dict[intermediary -> float]]
    per_metric_trends: dict[metric -> 'improving'|'degrading'|'flat']
    recommendations  : list[dict]
    """
    if len(matrix) < 5:
        return dict(trends={}, betas_final={}, per_metric_trends={},
                    recommendations=[], seasonal_period=2,
                    decompositions={}, intermediary_drivers={})

    n  = len(matrix)
    S  = min(8, max(2, n // 3))
    p  = len(INTERMEDIARY_METRICS)
    X  = np.zeros((n, p))
    for t, row in enumerate(matrix):
        for j, m in enumerate(INTERMEDIARY_METRICS):
            X[t, j] = float(row.get(m, 0.0))
    Xmean, Xstd = X.mean(0), X.std(0) + 1e-8
    Xn = (X - Xmean) / Xstd

    decompositions, trends, betas_final = {}, {}, {}
    intermediary_drivers = {}
    per_metric_trends = {}

    for sm in SUCCESS_METRICS:
        y = np.array([float(row.get(sm, 0.0)) for row in matrix])
        kf = BSTSKalmanFilter(
            seasonal_period = S, n_regressors=p,
            sigma_obs    = float(np.std(y) + 1e-6),
            sigma_level  = 0.1  * float(np.std(y) + 1e-6),
            sigma_trend  = 0.01 * float(np.std(y) + 1e-6),
            sigma_season = 0.05 * float(np.std(y) + 1e-6),
            sigma_beta   = 0.01,
        )
        res = kf.filter_series(y, Xn)
        recent = res['trends'][2*n//3:]
        slope  = float(np.mean(np.diff(recent))) if len(recent) > 1 else 0.0
        trends[sm] = slope
        # Direction: all SUCCESS_METRICS are 'up-good'
        if   slope >  0.001:  per_metric_trends[sm] = 'improving'
        elif slope < -0.001:  per_metric_trends[sm] = 'degrading'
        else:                 per_metric_trends[sm] = 'flat'

        final_betas = res['betas'][-1].tolist() if 'betas' in res else [0.0] * p
        betas_final[sm] = {m: float(b)
                           for m, b in zip(INTERMEDIARY_METRICS, final_betas)}
        driver_pairs = sorted(
            betas_final[sm].items(), key=lambda x: abs(x[1]), reverse=True)
        intermediary_drivers[sm] = driver_pairs[:5]
        decompositions[sm] = dict(
            levels      = res['levels'].tolist(),
            trends      = res['trends'].tolist(),
            seasonals   = res['seasonals'].tolist(),
            residuals   = res['residuals'].tolist(),
            predictions = res['predictions'].tolist(),
            betas       = res.get('betas', np.zeros((n, p))).tolist(),
        )

    improving = sum(1 for v in per_metric_trends.values() if v == 'improving')
    degrading  = sum(1 for v in per_metric_trends.values() if v == 'degrading')
    overall   = 'improving' if improving > degrading else (
                'degrading' if degrading > improving else 'flat')

    recs = []
    for sm, drv in intermediary_drivers.items():
        if per_metric_trends.get(sm) == 'degrading' and drv:
            top_name, top_coeff = drv[0]
            recs.append(dict(metric=sm, status='degrading',
                             top_driver=top_name, driver_coeff=top_coeff,
                             action=f'Address {top_name} (b={top_coeff:.4f}) → {sm}'))

    return dict(
        trends               = trends,
        betas_final          = betas_final,
        per_metric_trends    = per_metric_trends,
        recommendations      = recs,
        seasonal_period      = S,
        decompositions       = decompositions,
        intermediary_drivers = intermediary_drivers,
        trend                = overall,
    )


# ================================================================
# VI.  Anneal Recommendations  (maps to AnnealingScheduler keys)
# ================================================================

# Maps degrading intermediary driver -> (reward_weight_key, delta)
_DRIVER_TO_WEIGHT: Dict[str, Tuple[str, float]] = {
    "race_line_adherence"        : ("racingline",     +0.04),
    "brake_compliance"           : ("braking",        +0.05),
    "corner_speed_error"         : ("corner",         +0.04),
    "heading_alignment_mean"     : ("heading",        +0.03),
    "smoothness_jerk_rms"        : ("speedsteering",  +0.03),
    "waypoint_lookahead"         : ("progress",       +0.04),
    "gg_ellipse_utilisation"     : ("curvspeed",      +0.03),
    "velocity_profile_compliance": ("minspeed",       +0.03),
    "curvature_anticipation"     : ("corner",         +0.03),
    "htm_composite"              : ("racingline",     +0.03),
}

def compute_anneal_recommendations(bsts_rpt: dict,
                                   matrix: List[dict]) -> dict:
    """
    Map degrading BSTS drivers to reward weight adjustment proposals.

    Returns a dict directly usable by AnnealingScheduler / BSTSFeedback:
      weight_deltas   : dict[reward_key -> delta_float]  (add to current weight)
      lr_advice       : str
      residual_std    : float
      overall_trend   : str
    """
    weight_deltas: Dict[str, float] = {}
    drivers = bsts_rpt.get('intermediary_drivers', {})
    per_trends = bsts_rpt.get('per_metric_trends', {})

    for sm, drv_list in drivers.items():
        if per_trends.get(sm) != 'degrading':
            continue
        for name, coeff in drv_list[:3]:
            if name in _DRIVER_TO_WEIGHT:
                wkey, base_delta = _DRIVER_TO_WEIGHT[name]
                # Scale delta by |coeff| (capped at 2x base)
                delta = float(np.clip(base_delta * (1 + abs(coeff)), 0, base_delta * 2))
                weight_deltas[wkey] = weight_deltas.get(wkey, 0.0) + delta

    # Cap total adjustment per key at 0.15
    weight_deltas = {k: float(np.clip(v, 0, 0.15))
                     for k, v in weight_deltas.items()}

    # LR advice from residual variance
    decomps = bsts_rpt.get('decompositions', {})
    avg_res_std = float(np.mean([
        np.std(d.get('residuals', [0])) for d in decomps.values()
    ])) if decomps else 0.0

    if avg_res_std > 0.3:
        lr_advice = 'decrease_lr'       # high noise, unstable
    elif avg_res_std < 0.05:
        lr_advice = 'increase_lr'       # too little exploration
    else:
        lr_advice = 'hold'

    return dict(
        weight_deltas  = weight_deltas,
        lr_advice      = lr_advice,
        residual_std   = avg_res_std,
        overall_trend  = bsts_rpt.get('trend', 'flat'),
        seasonal_period= bsts_rpt.get('seasonal_period', 8),
    )


# ================================================================
# VII.  CLI orchestrator
# ================================================================

def run_full_analysis(log_dir: str = 'results'):
    eps      = load_jsonl_episodes(log_dir)
    bsts_rows= load_bsts_csv(log_dir)
    if not eps:
        print(f'[analyze_logs] No episodes found in {log_dir}')
        return
    from harmonized_metrics import compute_all as _ca
    matrix = []
    for i, ep in enumerate(eps):
        interm = extract_intermediary_metrics(ep)
        summ   = episode_summary_metrics(ep, interm)
        summ['episode'] = i
        matrix.append(summ)
    rpt  = bsts_compliance_report(matrix)
    recs = compute_anneal_recommendations(rpt, matrix)
    print(json.dumps(recs, indent=2))
    out_dir = os.path.join(log_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'bsts_report.json'), 'w') as f:
        json.dump(rpt, f, indent=2, default=str)
    with open(os.path.join(out_dir, 'anneal_recs.json'), 'w') as f:
        json.dump(recs, f, indent=2)
    print(f'[analyze_logs] saved to {out_dir}/')


if __name__ == '__main__':
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    run_full_analysis(log_dir)
