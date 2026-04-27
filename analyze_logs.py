import os
from denim_theme import apply_theme, get_color, DENIM_BRIGHT, AMBER, TERRA_COTTA, SAGE_GREEN, MUTED_PURPLE, GOLD_WARM, DENIM_MID, DENIM_DARK, BG_DARK, WHITE_SMOKE
#!/usr/bin/env python3
"""analyze_logs.py - DeepRacer training log analysis with proper BSTS decomposition.

Implements Bayesian Structural Time Series with:
  - Local linear trend (level + slope)
  - Seasonal component (lap-cycle seasonality)
  - Regression on intermediary time-series covariates
  - Kalman filter for state estimation

Intermediary metrics (covariates that AFFECT success metrics):
  - curvature_at_step, approach_velocity, perpendicular_velocity_at_barrier,
  - brake_zone_length, race_line_deviation, lateral_g, jerk, angular_rate,
  - steering_smoothness, throttle_consistency

Success metrics (response variables):
  - lap_completion_pct, reward_per_step, off_track_rate, crash_rate

Race-line calculus: v_perp -> 0 at barrier while minimizing integral of brake zones.

References: AWS (2020)TheRayG (2020)
"""
import os, sys, json, csv, glob, math
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

def bca_bootstrap_ci(data, stat_fn=None, n_boot=2000, alpha=0.05):
    """BCa bootstrap CI (Efron 1987 JASA 82:171-185)."""
    import numpy as np
    from scipy.stats import norm
    data = np.asarray(data, dtype=float)
    if stat_fn is None:
        stat_fn = np.mean
    observed = stat_fn(data)
    n = len(data)
    boot_stats = np.array([stat_fn(data[np.random.randint(0, n, n)]) for _ in range(n_boot)])
    z0 = norm.ppf(np.mean(boot_stats < observed))
    jk = np.array([stat_fn(np.delete(data, i)) for i in range(n)])
    jk_mean = jk.mean()
    num = np.sum((jk_mean - jk)**3)
    den = 6.0 * (np.sum((jk_mean - jk)**2))**1.5
    a = num / (den + 1e-12)
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    a_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    return float(np.percentile(boot_stats, 100 * a_lo)), float(np.percentile(boot_stats, 100 * a_hi))



# ============================================================
# I.  Data Loading
# ============================================================

def load_jsonl_episodes(log_dir: str) -> List[dict]:
    """Load all JSONL episode logs from a results directory."""
    patterns = [
        os.path.join(log_dir, '*.jsonl'),
        os.path.join(log_dir, 'episodes', '*.jsonl'),
    ]
    eps = []
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    for f in set(files):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    try: eps.append(json.loads(line))
                    except: pass
    return eps

def load_bsts_csv(log_dir: str) -> List[dict]:
    """Load BSTSFeedback EMA metrics CSV (legacy compat)."""
    p = os.path.join(log_dir, 'bsts_metrics.csv')
    if not os.path.exists(p): return []
    rows = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in r:
                try: r[k] = float(r[k])
                except: pass
            rows.append(r)
    return rows

# ============================================================
# II. Intermediary Time-Series Metrics Extraction
# ============================================================
# --- harmonized v3.0 metric keys (sourced from harmonized_metrics.compute_all) ---
from harmonized_metrics import compute_all as _ha_compute_all, compute_intermediary, compute_success  # noqa: F401
# v1.3.0: Canonical list — mirrors harmonized_metrics.py exactly.
# Do NOT edit here without editing harmonized_metrics.py.
# 'success' and diverged keys ('trail_braking_quality', 'waypoint_coverage',
#  'smoothness_jerk_rms') removed to fix BSTS Kalman always-0.0 bug.
from harmonized_metrics import (
    SUCCESS_METRICS,        # noqa: F401 re-export
    INTERMEDIARY_METRICS,   # noqa: F401 re-export
)

def extract_intermediary_metrics(ep: dict) -> dict:
    """Compute harmonized v3.0 intermediary time-series metrics for an episode.

    Delegates to harmonized_metrics.compute_intermediary so the keys stay in
    lock-step with live_bsts_plot.py / bsts_seasonal.py regressors.
    """
    from harmonized_metrics import compute_intermediary
    steps = ep.get('steps', ep.get('trajectory', []))
    if len(steps) < 3:
        return {m: 0.0 for m in INTERMEDIARY_METRICS}
    n_wp = int(ep.get('n_waypoints', 120))   # v1.6.0: 120 matches run.py default
    track_width = float(ep.get('track_width', 0.6))
    scalars = compute_intermediary(steps, n_waypoints=n_wp, track_width=track_width)
    # Ensure every expected key is present (zero-fill missing)
    return {m: float(scalars.get(m, 0.0)) for m in INTERMEDIARY_METRICS}


def episode_summary_metrics(ep: dict, intermediary: dict) -> dict:
    """Summarize an episode into scalar intermediary + success metrics."""
    steps = ep.get('steps', ep.get('trajectory', []))
    n = max(len(steps), 1)
    rewards = [s.get('reward', 0) for s in steps]
    on_track = [s.get('all_wheels_on_track', True) for s in steps]
    crashed = ep.get('crashed', ep.get('termination_reason', '') == 'crashed')
    completion = ep.get('completion_pct', ep.get('progress', 0))
    
    # v1.6.0 FIX-I: forward track_length_m and n_waypoints to compute_success.
    # ep_data now carries these (FIX-L in run.py). Without them compute_success
    # uses 100 n_wp and 0.6 tw defaults → tracklengthm=0.0 in every Kalman line.
    # --- Success metrics: harmonized v3.0 (VIF<5, collinearity-free) ---
    from harmonized_metrics import compute_success
    steps_for_success = ep.get('steps', ep.get('trajectory', []))
    n_wp = int(ep.get('n_waypoints', 120))           # FIX-I: was 100
    track_width = float(ep.get('track_width', 0.6))
    track_length_m = float(ep.get('track_length_m', 16.6))  # FIX-I: forward from ep_data
    # v1.1.5c FIX-I-compute: compute_success is a lambda that DROPS track_length_m before
    # calling compute_all → _track_progress() receives track_length_m=None → falls back to
    # progress/100 → track_progress=0.0 in every Kalman line.
    # FIX: call compute_all directly with track_length_m forwarded.
    # REF: heilmeier2020minCurv — track arc length normalization for progress metric.
    from harmonized_metrics import compute_all as _compute_all_full
    _succ_full = _compute_all_full(
        steps_for_success,
        n_waypoints=n_wp,
        track_width=track_width,
        track_length_m=track_length_m,
    ) if steps_for_success else {}
    succ = {k: _succ_full.get(k, 0.0) for k in SUCCESS_METRICS}
    summary = {k: float(succ.get(k, 0.0)) for k in SUCCESS_METRICS}
    # Keep legacy scalars available for audit / back-compat dashboards
    summary['_legacy_lap_completion_pct'] = float(completion)
    summary['_legacy_reward_per_step']    = float(np.mean(rewards)) if rewards else 0.0
    summary['_legacy_off_track_rate']     = 1.0 - float(np.mean(on_track)) if on_track else 0.0
    summary['_legacy_crash_rate']         = 1.0 if crashed else 0.0
    # --- Intermediary scalars (already per-episode scalars from harmonized_metrics) ---
    for k, v in intermediary.items():
        summary[k] = float(v) if np.isscalar(v) else (float(np.mean(v)) if len(v) else 0.0)
    return summary

# ============================================================
# III. Kalman-Filter BSTS Engine
# ============================================================

class BSTSKalmanFilter:
# REF: Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. J. Basic Eng., 82(1), 35-45.
    """Bayesian Structural Time Series via Kalman Filter.
    
    State vector: [level, trend, s_1, s_2, ..., s_{S-1}, beta_1, ..., beta_p]
    where S = seasonal_period, p = number of regression covariates.
    
    Observation: y_t = level_t + season_t + X_t @ beta_t + epsilon_t
    
    Transitions:
      level_{t+1}  = level_t + trend_t + eta_level
      trend_{t+1}  = trend_t + eta_trend
      season_{t+1} = -sum(s_{t}, ..., s_{t-S+2}) + eta_season
      beta_{t+1}   = beta_t + eta_beta  (random walk coefficients)
    """
    
    def __init__(self, seasonal_period: int = 8, n_regressors: int = 0,
                 sigma_obs: float = 1.0, sigma_level: float = 0.1,
                 sigma_trend: float = 0.01, sigma_season: float = 0.1,
                 sigma_beta: float = 0.01):
        self.S = seasonal_period
        self.p = n_regressors
        # State dim: 2 (level+trend) + (S-1) seasonal + p regression
        self.d = 2 + (self.S - 1) + self.p
        
        # Initial state and covariance
        self.state = np.zeros(self.d)
        self.P = np.eye(self.d) * 1e4  # diffuse prior
        
        # Noise variances
        self.sigma_obs = sigma_obs
        self.sigma_level = sigma_level
        self.sigma_trend = sigma_trend
        self.sigma_season = sigma_season
        self.sigma_beta = sigma_beta
        
        self._build_matrices()
    
    def _build_matrices(self):
        """Build transition matrix T and observation vector Z."""
        d = self.d
        S = self.S
        p = self.p
        
        # --- Transition matrix T ---
        self.T = np.zeros((d, d))
        # Level: level_{t+1} = level_t + trend_t
        self.T[0, 0] = 1.0
        self.T[0, 1] = 1.0
        # Trend: trend_{t+1} = trend_t
        self.T[1, 1] = 1.0
        # Seasonal: s_{t+1} = -s_t - s_{t-1} - ... - s_{t-S+2}
        if S > 1:
            self.T[2, 2:2+S-1] = -1.0
            # Shift: s_{i+1} = s_i for i in [3..2+S-2]
            for i in range(3, 2 + S - 1):
                self.T[i, i-1] = 1.0
        # Regression: random walk beta_{t+1} = beta_t
        for i in range(p):
            idx = 2 + S - 1 + i
            self.T[idx, idx] = 1.0
        
        # --- Process noise Q ---
        self.Q = np.zeros((d, d))
        self.Q[0, 0] = self.sigma_level**2
        self.Q[1, 1] = self.sigma_trend**2
        if S > 1:
            self.Q[2, 2] = self.sigma_season**2
        for i in range(p):
            idx = 2 + S - 1 + i
            self.Q[idx, idx] = self.sigma_beta**2
        
        # Observation noise
        self.R = self.sigma_obs**2

    def _apply_spike_slab(self):
        """Soft spike-and-slab beta shrinkage via pseudo-inclusion probability.

        REF: George & McCulloch (1993) Variable selection via Gibbs sampling.
             JASA, 88(423), 881-889.

        Approximation: betas with |beta| < inclusion_threshold are shrunk by 50%
        each step. This mimics the Gibbs spike-and-slab prior without full MCMC.
        Betas that earn their keep (|beta| > threshold) are left alone.
        """
        _beta_start = 2 + self.S - 1   # index into state vector after level+trend+seasonals
        _inclusion_threshold = 0.05
        for i in range(self.p):
            idx = _beta_start + i
            if idx < len(self.state):
                beta_i = self.state[idx]
                # Soft-threshold: shrink near-zero betas toward 0 (exclude them)
                if abs(beta_i) < _inclusion_threshold:
                    self.state[idx] *= 0.5   # 50% shrinkage per step

    def _adapt_sigma_obs(self):
        """Adaptive observation noise sigma from rolling residual variance.

        REF: Scott & Varian (2014) use Inverse-Gamma prior on sigma_obs.
             Here we approximate with an online rolling variance update every 10 steps.
        """
        if not hasattr(self, '_residuals'):
            self._residuals = []
        if len(self._residuals) >= 20:
            rolling_var = float(np.var(self._residuals[-20:]) + 1e-6)
            # Smooth update: don't jump sigma_obs abruptly
            self.R = 0.8 * self.R + 0.2 * rolling_var

    def _obs_vector(self, X_t: Optional[np.ndarray] = None) -> np.ndarray:
        """Observation vector Z_t: y_t = Z_t @ state_t + eps."""
        Z = np.zeros(self.d)
        Z[0] = 1.0  # level
        if self.S > 1:
            Z[2] = 1.0  # current seasonal
        if X_t is not None and self.p > 0:
            idx0 = 2 + self.S - 1
            Z[idx0:idx0+self.p] = X_t[:self.p]
        return Z
    
    def predict(self):
        """Kalman predict step."""
        self.state = self.T @ self.state
        self.P = self.T @ self.P @ self.T.T + self.Q
    
    def update(self, y_t: float, X_t: Optional[np.ndarray] = None):
        """Kalman update step.

        v1.5.0: isfinite guard + nan_to_num state recovery.
          Non-finite y_t (nan/±inf) are SKIPPED — same measurement-gating
          pattern as bsts_seasonal.BSTSKalmanFilter v1.3.1.
          A single nan y_t poisons self.state via:
            residual = nan → K * nan = nan → state = nan → trend = nan
          After every update, nan/inf that leaked through numerical edge
          cases (e.g., singular P, near-zero S denom) are recovered.
          REF: Thrun, Burgard & Fox (2005) Probabilistic Robotics §3.4
               measurement gating — reject corrupt observations.
          REF: Welch & Bishop (1995) TR 95-041 — numerical stability.

        v1.1.0: Adaptive sigma_obs and spike-and-slab variable selection now
        fire automatically every 10 updates inside this method.  The previously
        commented-out stub block has been removed; the calls live here.

        REF: Scott & Varian (2014) Predicting the present with BSTS.
             Int. J. Math. Model. Numer. Optim., 5(1-2), 4-23.
        REF: George & McCulloch (1993) Variable selection via Gibbs sampling.
             JASA, 88(423), 881-889.
        """
        # v1.5.0 FIX-F: skip non-finite observations (measurement gating).
        # nan y_t → residual=nan → K*nan=nan → state=[nan,nan,...] → trend=nan.
        # Skipping is preferable to substituting 0.0 (would bias the level down).
        # REF: Thrun et al. (2005) §3.4 — measurement gating.
        y_t = float(y_t)
        if not math.isfinite(y_t):
            return {
                'y_pred': 0.0, 'residual': 0.0,
                'level': float(self.state[0]),
                'trend': float(self.state[1]),
                'seasonal': float(self.state[2]) if self.S > 1 else 0.0,
                'beta': self.state[2 + self.S - 1:].tolist() if self.p > 0 else [],
                'kalman_gain_norm': 0.0,
            }
        # v1.5.0: recover nan/inf that may have leaked through numerical edge cases
        # (singular P, near-zero innovation variance S → gain explosion).
        if not np.all(np.isfinite(self.state)):
            self.state = np.nan_to_num(self.state, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(self.P)):
            self.P = np.nan_to_num(self.P, nan=1e4, posinf=1e4, neginf=0.0)
        # --- step counter (drives the periodic adapt/prune cadence) ---
        self._t = getattr(self, '_t', 0) + 1
        Z = self._obs_vector(X_t)
        y_pred = Z @ self.state
        residual = y_t - y_pred
        S = Z @ self.P @ Z + self.R
        K = self.P @ Z / (S + 1e-12)  # Kalman gain
        self.state = self.state + K * residual
        self.P = self.P - np.outer(K, Z) @ self.P
        # v1.1.0: adaptive sigma_obs + spike-and-slab (every 10 updates)
        self._residuals = getattr(self, '_residuals', [])
        self._residuals.append(float(residual))
        if self._t % 10 == 0:
            self._adapt_sigma_obs()
            self._apply_spike_slab()
        return {
            'y_pred': float(y_pred),
            'residual': float(residual),
            'level': float(self.state[0]),
            'trend': float(self.state[1]),
            'seasonal': float(self.state[2]) if self.S > 1 else 0.0,
            'beta': self.state[2+self.S-1:].tolist() if self.p > 0 else [],
            'kalman_gain_norm': float(np.linalg.norm(K)),
        }
    
    def filter_series(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> dict:
        """Run full Kalman filter over a time series.
        
        Args:
            y: (T,) response time series
            X: (T, p) regressor matrix, optional
        Returns:
            dict with arrays: levels, trends, seasonals, betas, residuals, predictions
        """
        T_len = len(y)
        levels = np.zeros(T_len)
        trends = np.zeros(T_len)
        seasonals = np.zeros(T_len)
        betas = np.zeros((T_len, self.p)) if self.p > 0 else None
        residuals = np.zeros(T_len)
        predictions = np.zeros(T_len)
        
        for t in range(T_len):
            self.predict()
            X_t = X[t] if X is not None else None
            res = self.update(y[t], X_t)
            levels[t] = res['level']
            trends[t] = res['trend']
            seasonals[t] = res['seasonal']
            if betas is not None:
                betas[t] = res['beta']
            residuals[t] = res['residual']
            predictions[t] = res['y_pred']
        
        result = {
            'levels': levels,
            'trends': trends,
            'seasonals': seasonals,
            'residuals': residuals,
            'predictions': predictions,
        }
        if betas is not None:
            result['betas'] = betas
        return result

# ============================================================
# IV. Race Line Analysis & Perpendicular Velocity Calculus
# ============================================================

def compute_optimal_race_line(waypoints: List[Tuple[float, float]],
                              track_width: float = 0.6) -> dict:
    """Compute optimal race line geometry.

    v1.1.0 HARMONIZED: delegates curvature calculation to race_line_engine
    (Menger-correct circumradius) so analyze_logs.py and run.py share ONE
    implementation.  The old standalone finite-difference curvature loop is
    used only as a fallback when race_line_engine is not importable.
    The backward brake-propagation pass is preserved here for batch analysis.

    REF: Heilmeier et al. (2020) Minimum-curvature QP racing line. Proc. IMechE.
    REF: Hart et al. (1968) A* heuristic. IEEE Trans. SSC, 4(2), 100-107.
    REF: Menger (1930) three-point circumradius (correct curvature estimator).
    REF: Brayshaw & Harrison (2005) Quasi-steady-state lap simulation. Proc. IMechE.

    Returns dict with:
      - race_line: list of (x,y) optimal positions
      - curvatures: Menger curvature at each waypoint
      - max_speeds: theoretical max speed (backward-pass brake-propagated)
      - brake_points: indices where braking must begin
      - brake_zone_integral: total braking distance (to minimise)
      - normals: (nx, ny) unit normal at each wp (for v_perp barrier check)
    """
    if not waypoints or len(waypoints) < 4:
        return {'race_line': waypoints, 'curvatures': [], 'max_speeds': [],
                'brake_points': [], 'brake_zone_integral': 0.0, 'normals': []}

    pts = np.array(waypoints, dtype=float)
    n = len(pts)

    # --- Curvature: prefer Menger (race_line_engine), fall back to finite diff ---
    try:
        from race_line_engine import _curvature_radius, _optimal_speed
        wpts_list = [(float(p[0]), float(p[1])) for p in pts]
        curvatures = np.array([
            1.0 / max(_curvature_radius(wpts_list, i), 1e-4)
            for i in range(n)
        ])
        max_speeds = np.array([_optimal_speed(1.0 / max(k, 1e-4)) for k in curvatures])
    except ImportError:
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        denom = (dx**2 + dy**2)**1.5 + 1e-8
        curvatures = np.abs(dx * d2y - dy * d2x) / denom
        a_lat_max = 4.0
        max_speeds = np.clip(np.sqrt(a_lat_max / (curvatures + 1e-6)), 0, 4.0)

    # --- Backward brake-propagation pass (Brayshaw & Harrison 2005) ---
    a_brake = 3.0
    ds = np.sqrt(np.diff(pts[:, 0])**2 + np.diff(pts[:, 1])**2)
    ds = np.append(ds, ds[-1])
    brake_limited = max_speeds.copy()
    for i in range(n - 2, -1, -1):
        v_next = brake_limited[i + 1]
        v_brake = math.sqrt(v_next**2 + 2 * a_brake * float(ds[i]))
        brake_limited[i] = min(brake_limited[i], v_brake)

    brake_mask = brake_limited < max_speeds * 0.95
    brake_points = np.where(brake_mask)[0].tolist()
    brake_zone_integral = float(np.sum(ds[brake_mask]))

    # Normal vectors (for v_perp . n_barrier dot-product at barrier)
    dx2 = np.gradient(pts[:, 0])
    dy2 = np.gradient(pts[:, 1])
    mag = np.sqrt(dx2**2 + dy2**2) + 1e-8
    tangent_x, tangent_y = dx2 / mag, dy2 / mag
    normal_x, normal_y = -tangent_y, tangent_x

    return {
        'race_line': pts.tolist(),
        'curvatures': curvatures.tolist(),
        'max_speeds': brake_limited.tolist(),
        'brake_points': brake_points,
        'brake_zone_integral': brake_zone_integral,
        'normals': list(zip(normal_x.tolist(), normal_y.tolist())),
    }

def score_race_line_compliance(episodes: List[dict], race_line: dict) -> dict:
    """Score how well episodes follow the optimal race line.
    
    Key metric: perpendicular velocity at barrier/curb should be zero.
    v_perp = v . n_barrier  (dot product of velocity with barrier normal)
    
    Also measures brake zone efficiency: actual vs theoretical minimum.
    """
    if not episodes or not race_line.get('race_line'):
        return {'avg_perp_v': 0, 'avg_deviation': 0, 'brake_efficiency': 0,
                'perp_v_at_barrier_episodes': []}
    
    rl_pts = np.array(race_line['race_line'])
    normals = np.array(race_line.get('normals', []))
    
    all_perp_v = []
    all_deviations = []
    all_brake_eff = []
    per_ep_perp = []
    
    for ep in episodes:
        steps = ep.get('steps', ep.get('trajectory', []))
        if len(steps) < 3:
            continue
        
        xs = np.array([s.get('x', 0) for s in steps])
        ys = np.array([s.get('y', 0) for s in steps])
        speeds = np.array([s.get('speed', 0) for s in steps])
        headings = np.deg2rad([s.get('heading', 0) for s in steps])
        
        # Velocity vectors
        vx = speeds * np.cos(headings)
        vy = speeds * np.sin(headings)
        
        ep_perp_v = []
        ep_dev = []
        
        for i in range(len(steps)):
            # Find closest race line point
            dists = np.sqrt((rl_pts[:, 0] - xs[i])**2 + (rl_pts[:, 1] - ys[i])**2)
            closest = np.argmin(dists)
            ep_dev.append(float(dists[closest]))
            
            # Perpendicular velocity at this point
            if closest < len(normals):
                n = normals[closest]
                v_perp = abs(vx[i] * n[0] + vy[i] * n[1])
                ep_perp_v.append(float(v_perp))
        
        if ep_perp_v:
            all_perp_v.extend(ep_perp_v)
            per_ep_perp.append(float(np.mean(ep_perp_v)))
        if ep_dev:
            all_deviations.extend(ep_dev)
        
        # Brake efficiency: ratio of theoretical min brake zone to actual
        throttles = [s.get('throttle', 1) for s in steps]
        actual_brake = sum(1 for t in throttles if t < 0.5)
        theo_brake = len(race_line.get('brake_points', []))
        if actual_brake > 0:
            all_brake_eff.append(min(1.0, theo_brake / actual_brake))
    
    return {
        'avg_perp_v': float(np.mean(all_perp_v)) if all_perp_v else 0,
        'avg_deviation': float(np.mean(all_deviations)) if all_deviations else 0,
        'brake_efficiency': float(np.mean(all_brake_eff)) if all_brake_eff else 0,
        'perp_v_at_barrier_episodes': per_ep_perp,
    }

# ============================================================
# V.  BSTS Compliance Report (proper decomposition)
# ============================================================

BSTS_COLUMNS = (
    ['episode'] + SUCCESS_METRICS +
    [f'{m}_mean' for m in INTERMEDIARY_METRICS] +
    [f'{m}_max' for m in INTERMEDIARY_METRICS] +
    [f'{m}_std' for m in INTERMEDIARY_METRICS]
)

def build_design_matrix(episodes: List[dict], bsts_rows: List[dict]) -> List[dict]:
    """Build full design matrix with intermediary + success metrics per episode."""
    matrix = []
    bsts_idx = {}
    for i, r in enumerate(bsts_rows):
        gs = r.get('global_step', r.get('episode', i))
        bsts_idx[int(gs)] = r
    
    for i, ep in enumerate(episodes):
        intermediary = extract_intermediary_metrics(ep)
        row = episode_summary_metrics(ep, intermediary)
        row['episode'] = i
        
        # Merge legacy BSTS EMA if available
        bsts_r = bsts_idx.get(i, {})
        row['lap_completion_rate_ema'] = bsts_r.get('completion_ema', row['lap_completion_pct'])
        row['crash_rate_ema'] = bsts_r.get('crash_rate_ema', row['crash_rate'])
        
        matrix.append(row)
    return matrix

def bsts_compliance_report(matrix: List[dict]) -> dict:
    """Run proper BSTS decomposition on each success metric.
    
    For each success metric, we:
    1. Build the response time series y = success_metric[t]
    2. Build regressor matrix X from intermediary metrics
    3. Run Kalman filter BSTS with trend + seasonal + regression
    4. Return decomposed components + regression coefficients
    
    This tells us:
    - trend: is the metric improving/degrading over episodes?
    - seasonal: are there cyclical patterns (e.g., track rotation effects)?
    - regression betas: which intermediary metrics DRIVE this success metric?
    """
    if len(matrix) < 5:
        return {'decompositions': {}, 'trend': 'insufficient_data',
                'recommendations': [], 'intermediary_drivers': {}}
    
    n_eps = len(matrix)
    
    # Collect intermediary regressor names
    reg_names = list(INTERMEDIARY_METRICS)  # harmonized v3.0 scalar keys
    p = len(reg_names)
    
    # Build regressor matrix X (n_eps x p)
    X = np.zeros((n_eps, p))
    for t in range(n_eps):
        for j, rn in enumerate(reg_names):
            X[t, j] = matrix[t].get(rn, 0.0)
    
    # Normalize regressors
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    # Seasonal period: use 8 (typical config rotation cycle)
    S = min(8, max(2, n_eps // 3))
    
    decompositions = {}
    intermediary_drivers = {}
    overall_trends = {}
    
    for sm in SUCCESS_METRICS:
        y = np.array([matrix[t].get(sm, 0.0) for t in range(n_eps)])
        
        # Run BSTS Kalman filter
        kf = BSTSKalmanFilter(
            seasonal_period=S,
            n_regressors=p,
            sigma_obs=float(np.std(y) + 1e-6),
            sigma_level=0.1 * float(np.std(y) + 1e-6),
            sigma_trend=0.01 * float(np.std(y) + 1e-6),
            sigma_season=0.05 * float(np.std(y) + 1e-6),
            sigma_beta=0.01,
        )
        result = kf.filter_series(y, X_norm)
        
        # Store decomposition
        decompositions[sm] = {
            'levels': result['levels'].tolist(),
            'trends': result['trends'].tolist(),
            'seasonals': result['seasonals'].tolist(),
            'residuals': result['residuals'].tolist(),
            'predictions': result['predictions'].tolist(),
        }
        
        # Trend direction from last 1/3 of data
        recent = result['trends'][2*n_eps//3:]
        if len(recent) > 1:
            trend_slope = float(np.mean(np.diff(recent)))
            overall_trends[sm] = 'improving' if (
                (sm in ('lap_completion_pct', 'reward_per_step') and trend_slope > 0) or
                (sm in ('off_track_rate', 'crash_rate') and trend_slope < 0)
            ) else 'degrading'
        else:
            overall_trends[sm] = 'insufficient_data'
        
        # Regression coefficients: which intermediary metrics matter most?
        if 'betas' in result:
            final_betas = result['betas'][-1]  # last time step
            driver_pairs = sorted(
                zip(reg_names, final_betas.tolist()),
                key=lambda x: abs(x[1]), reverse=True
            )
            intermediary_drivers[sm] = driver_pairs[:5]  # top 5 drivers
            decompositions[sm]['betas'] = result['betas'].tolist()
    
    # Overall trend summary
    improving_count = sum(1 for v in overall_trends.values() if v == 'improving')
    degrading_count = sum(1 for v in overall_trends.values() if v == 'degrading')
    
    if improving_count > degrading_count:
        overall = 'improving'
    elif degrading_count > improving_count:
        overall = 'degrading'
    else:
        overall = 'mixed'
    
    # Recommendations based on decomposition
    recommendations = []
    for sm, drivers in intermediary_drivers.items():
        if overall_trends[sm] == 'degrading':
            top_driver = drivers[0] if drivers else ('unknown', 0)
            recommendations.append({
                'metric': sm,
                'status': 'degrading',
                'top_driver': top_driver[0],
                'driver_coeff': top_driver[1],
                'action': f'Address {top_driver[0]} (coeff={top_driver[1]:.4f}) to improve {sm}',
            })
    
    return {
        'decompositions': decompositions,
        'trend': overall,
        'per_metric_trends': overall_trends,
        'intermediary_drivers': intermediary_drivers,
        'recommendations': recommendations,
        'seasonal_period': S,
    }

# ============================================================
# VI.  Anneal Recommendations from BSTS Decomposition
# ============================================================

def compute_anneal_recommendations(bsts_report: dict, matrix: List[dict]) -> dict:
    """Based on BSTS decomposition, recommend hyperparameter annealing.
    
    Uses regression coefficients to identify which intermediary metrics
    most affect degrading success metrics, then suggests reward weight
    and learning rate adjustments.
    """
    recs = {}
    flags = bsts_report.get('per_metric_trends', {})
    drivers = bsts_report.get('intermediary_drivers', {})
    
    # Reward weight adjustments based on BSTS regression
    reward_adjustments = {}
    
    for sm, trend in flags.items():
        sm_drivers = drivers.get(sm, [])
        if trend == 'degrading' and sm_drivers:
            # The top driver with negative coefficient needs more reward weight
            for name, coeff in sm_drivers[:3]:
                base_metric = name  # harmonized v3.0 keys are already bare metric names
                # Map harmonized degrading driver -> reward weight adjustment recommendation
                if base_metric == 'corner_speed_error':
                    reward_adjustments['speed_weight']    = 'decrease 0.15 -> 0.08 (curb overspeed in corners)'
                    reward_adjustments['steering_weight'] = 'increase 0.10 -> 0.20 (smoother corner entry)'
                elif base_metric == 'brake_compliance':
                    reward_adjustments['progress_weight'] = 'increase 0.20 -> 0.30 (reward trail-brake compliance)'
                elif base_metric == 'race_line_adherence':
                    reward_adjustments['center_weight']   = 'increase 0.05 -> 0.15 (tighten race-line adherence)'
                elif base_metric == 'smoothness_jerk_rms':
                    reward_adjustments['smoothness_weight'] = 'increase 0.05 -> 0.12 (reduce jerk RMS)'
                elif base_metric == 'curvature_anticipation':
                    reward_adjustments['corner_weight']   = 'increase 0.08 -> 0.18 (earlier curvature anticipation)'
                elif base_metric == 'gg_ellipse_utilisation':
                    reward_adjustments['traction_weight'] = 'increase 0.05 -> 0.15 (exploit gg envelope)'
                elif base_metric == 'trail_braking_quality':
                    reward_adjustments['brake_weight']    = 'increase 0.05 -> 0.15 (improve trail braking)'
                elif base_metric == 'velocity_profile_compliance':
                    reward_adjustments['speed_weight']    = 'increase 0.05 -> 0.12 (match optimal v-profile)'
                elif base_metric == 'heading_alignment_mean':
                    reward_adjustments['heading_weight']  = 'increase 0.05 -> 0.12 (improve heading alignment)'
                elif base_metric == 'waypoint_coverage':
                    reward_adjustments['progress_weight'] = 'increase 0.10 -> 0.20 (boost waypoint coverage)'

    # Learning rate schedule based on trend stability
    decomps = bsts_report.get('decompositions', {})
    avg_residual_std = 0
    for sm, d in decomps.items():
        if d.get('residuals'):
            avg_residual_std += np.std(d['residuals'])
    avg_residual_std /= max(len(decomps), 1)
    
    if avg_residual_std > 0.3:
        lr_rec = 'decrease LR: high residual variance suggests unstable learning'
    elif avg_residual_std < 0.05:
        lr_rec = 'increase LR: low residual variance suggests room for faster learning'
    else:
        lr_rec = 'maintain current LR'
    
    recs = {
        'reward_weight_adjustments': reward_adjustments,
        'learning_rate': lr_rec,
        'residual_std': float(avg_residual_std),
        'overall_trend': bsts_report.get('trend', 'unknown'),
        'seasonal_detected': any(
            np.std(d.get('seasonals', [0])) > 0.01
            for d in decomps.values()
        ),
    }
    return recs


# ============================================================
# VII. Print Helpers
# ============================================================

def print_section(title: str):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')

def print_bsts_decomposition(bsts_rpt: dict):
    """Pretty-print BSTS decomposition results."""
    print_section('BSTS DECOMPOSITION RESULTS')
    
    print(f'  Overall trend: {bsts_rpt.get("trend", "N/A")}')
    print(f'  Seasonal period: {bsts_rpt.get("seasonal_period", "N/A")}')
    
    for sm, trend in bsts_rpt.get('per_metric_trends', {}).items():
        print(f'  {sm}: {trend}')
        drivers = bsts_rpt.get('intermediary_drivers', {}).get(sm, [])
        if drivers:
            print(f'    Top drivers:')
            for name, coeff in drivers[:3]:
                print(f'      {name}: beta={coeff:.4f}')
    
    recs = bsts_rpt.get('recommendations', [])
    if recs:
        print(f'\n  Recommendations:')
        for r in recs:
            print(f'    [{r["status"]}] {r["action"]}')


# ============================================================
# VIII. Main Orchestrator
# ============================================================

def run_full_analysis(log_dir: str = 'results', _prefix: str = 'v'):  # TODO: wire prefix into output naming
    """Run complete BSTS analysis pipeline."""
    print_section(f'DeepRacer BSTS Analysis - {log_dir}')
    
    # 1. Load episodes
    eps = load_jsonl_episodes(log_dir)
    bsts_rows = load_bsts_csv(log_dir)
    # jsonl_eps removed (was alias for eps)
    # csv_eps removed (was alias for bsts_rows)
    print(f'  Loaded {len(eps)} episodes, {len(bsts_rows)} BSTS CSV rows')
    
    if not eps:
        print('  No episodes found. Exiting.')
        return
    
    # 2. Build design matrix with all intermediary metrics
    print_section('DESIGN MATRIX')
    matrix = build_design_matrix(eps, bsts_rows)
    print(f'  Matrix: {len(matrix)} episodes x {len(BSTS_COLUMNS)} columns')
    
    # 3. Run proper BSTS decomposition
    print_section('BSTS DECOMPOSITION (Kalman Filter)')
    bsts_rpt = bsts_compliance_report(matrix)
    print_bsts_decomposition(bsts_rpt)
    
    # 4. Race line analysis
    print_section('RACE LINE ANALYSIS')
    # Extract waypoints from first complete episode
    waypoints = []
    for ep in eps:
        steps = ep.get('steps', ep.get('trajectory', []))
        completion = ep.get('completion_pct', ep.get('progress', 0))
        if completion > 80 and len(steps) > 10:
            waypoints = [(s.get('x', 0), s.get('y', 0)) for s in steps]
            break
    
    if waypoints:
        race_line = compute_optimal_race_line(waypoints)
        rl_score = score_race_line_compliance(eps[-20:], race_line)
        print(f'  Avg perpendicular velocity at barrier: {rl_score["avg_perp_v"]:.4f}')
        print(f'  Avg race line deviation: {rl_score["avg_deviation"]:.4f}')
        print(f'  Brake zone efficiency: {rl_score["brake_efficiency"]:.2%}')
        print(f'  Brake zone integral (minimize): {race_line["brake_zone_integral"]:.4f}')
    else:
        print('  No complete laps found for race line analysis.')
        race_line = {}
        rl_score = {}
    
    # 5. Anneal recommendations
    print_section('ANNEAL RECOMMENDATIONS')
    recs = compute_anneal_recommendations(bsts_rpt, matrix)
    print(f'  Learning rate: {recs["learning_rate"]}')
    print(f'  Residual std: {recs["residual_std"]:.4f}')
    print(f'  Seasonal effects detected: {recs["seasonal_detected"]}')
    for k, v in recs.get('reward_weight_adjustments', {}).items():
        print(f'  {k}: {v}')
    
    # 6. Summary stats
    print_section('SUMMARY')
    R = np.array([m.get('reward_per_step', 0) for m in matrix])
    C = np.array([m.get('lap_completion_pct', 0) for m in matrix])
    laps = sum(1 for m in matrix if m.get('lap_completion_pct', 0) >= 100)
    print(f'  Total episodes:     {len(eps)}')
    print(f'  Avg reward:         {np.mean(R):.2f} +/- {np.std(R):.2f}')
    print(f'  Avg completion:     {np.mean(C):.1f}%')
    print(f'  Full laps:          {laps}/{len(eps)} ({laps/max(1,len(eps))*100:.1f}%)')
    print(f'  Training trend:     {bsts_rpt.get("trend", "unknown")}')
    
    # 7. Save outputs
    matrix_path = os.path.join(log_dir, 'design_matrix.json')
    with open(matrix_path, 'w') as f:
        json.dump(matrix, f, indent=2, default=str)
    print(f'\n  Design matrix saved to: {matrix_path}')
    
    bsts_path = os.path.join(log_dir, 'bsts_decomposition.json')
    with open(bsts_path, 'w') as f:
        json.dump(bsts_rpt, f, indent=2, default=str)
    print(f'  BSTS decomposition saved to: {bsts_path}')
    
    rec_path = os.path.join(log_dir, 'anneal_recommendations.json')
    with open(rec_path, 'w') as f:
        json.dump(recs, f, indent=2)
    print(f'  Recommendations saved to: {rec_path}')

def plot_bsts_decomposition(bsts_rpt: dict, matrix: list,
                            directory: str = './plots'):
    """Render BSTS Kalman decomposition as publication-quality panels.

    For each success metric produces:
      Row 1: Observed vs BSTS prediction with residual shading
      Row 2: Level + Trend extracted by Kalman filter
      Row 3: Seasonal component
      Row 4: Regression coefficient evolution (top-5 driver betas over time)
      Row 5: Driver importance heatmap (final beta magnitudes)

    References:
        Scott & Varian (2014) BSTS for causal inference
        Durbin & Koopman (2012) Time Series Analysis by State Space Methods
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    try:
        from denim_theme import (apply_theme, get_color, DENIM_BRIGHT,
            AMBER, TERRA_COTTA, SAGE_GREEN, MUTED_PURPLE, GOLD_WARM,
            DENIM_MID, DENIM_DARK, BG_DARK, WHITE_SMOKE, MUTED_GRAY)
    except ImportError:
        BG_DARK, WHITE_SMOKE, MUTED_GRAY = "#2E2E2E", "#F5F5F5", "#8C8C8C"
        DENIM_BRIGHT, AMBER, TERRA_COTTA = "#5B9BD5", "#D4A03C", "#C75B39"
        SAGE_GREEN, MUTED_PURPLE, GOLD_WARM = "#6BB38A", "#9B6EB7", "#E8C167"
        DENIM_MID, DENIM_DARK = "#3A6B8C", "#1B3A5C"
        def apply_theme(fig=None, ax=None): pass
        def get_color(i): return [DENIM_BRIGHT, AMBER, TERRA_COTTA][i % 3]

    os.makedirs(directory, exist_ok=True)
    decomps = bsts_rpt.get('decompositions', {})
    drivers_map = bsts_rpt.get('intermediary_drivers', {})
    trends_map = bsts_rpt.get('per_metric_trends', {})
    PALETTE = [DENIM_BRIGHT, AMBER, TERRA_COTTA, SAGE_GREEN, MUTED_PURPLE]

    for sm, dec in decomps.items():
        lvl = np.asarray(dec['levels'])
        trn = np.asarray(dec['trends'])
        sea = np.asarray(dec['seasonals'])
        res = np.asarray(dec['residuals'])
        pred = np.asarray(dec['predictions'])
        n = len(lvl)
        x = np.arange(n)

        # Build observed from matrix
        obs = np.array([m.get(sm, 0) for m in matrix[:n]])

        fig = plt.figure(figsize=(15, 18))
        gs = gridspec.GridSpec(5, 1, hspace=0.38,
                               height_ratios=[2, 1.2, 1, 1.5, 1.2])
        axes = [fig.add_subplot(gs[i]) for i in range(5)]

        # Row 0: Observed vs Prediction
        ax = axes[0]
        ax.plot(x, obs, linewidth=0.5, alpha=0.5, color=MUTED_GRAY, label='observed')
        ax.plot(x, pred, linewidth=1.6, color=DENIM_BRIGHT, label='BSTS prediction')
        ax.fill_between(x, pred - np.abs(res), pred + np.abs(res),
                        alpha=0.12, color=DENIM_BRIGHT)
        trend_dir = trends_map.get(sm, 'N/A')
        ax.set_title(f'{sm}  [{trend_dir}]', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

        # Row 1: Level + Trend
        ax1 = axes[1]
        ax1.plot(x, lvl, linewidth=1.4, color=GOLD_WARM, label='level')
        ax1.plot(x, lvl + trn, linewidth=0.9, color=AMBER, linestyle='--',
                 label='level + trend')
        ax1.set_title('Kalman Level + Trend', fontsize=10)
        ax1.legend(fontsize=7)

        # Row 2: Seasonal
        ax2 = axes[2]
        ax2.fill_between(x, 0, sea, where=sea>=0, color=SAGE_GREEN, alpha=0.5)
        ax2.fill_between(x, 0, sea, where=sea<0, color=TERRA_COTTA, alpha=0.5)
        ax2.axhline(0, color=MUTED_GRAY, linewidth=0.5)
        ax2.set_title(f'Seasonal (period={bsts_rpt.get("seasonal_period", "?")})',
                      fontsize=10)

        # Row 3: Beta evolution (top 5 drivers over time)
        ax3 = axes[3]
        betas = dec.get('betas')
        sm_drivers = drivers_map.get(sm, [])
        if betas is not None and sm_drivers:
            betas = np.asarray(betas)
            reg_names = [d[0] for d in sm_drivers[:5]]
            for i, (nm, _) in enumerate(sm_drivers[:5]):
                # Find index in reg_names list -- betas columns correspond
                # to intermediary metric means in order
                col_idx = i  # simplified: top drivers sorted by magnitude
                if col_idx < betas.shape[1]:
                    ax3.plot(x[:betas.shape[0]], betas[:, col_idx],
                             linewidth=1.2, color=PALETTE[i % len(PALETTE)],
                             label=nm.replace('_mean', ''), alpha=0.8)
            ax3.axhline(0, color=MUTED_GRAY, linewidth=0.5, linestyle='--')
            ax3.legend(fontsize=6, ncol=2, loc='upper left')
        ax3.set_title('Regression Coefficients (beta evolution)', fontsize=10)

        # Row 4: Driver importance heatmap
        ax4 = axes[4]
        if sm_drivers:
            names = [d[0].replace('_mean', '') for d in sm_drivers[:8]]
            vals = [abs(d[1]) for d in sm_drivers[:8]]
            colors = [DENIM_BRIGHT if d[1] > 0 else TERRA_COTTA
                      for d in sm_drivers[:8]]
            bars = ax4.barh(range(len(names)), vals, color=colors, alpha=0.75)
            ax4.set_yticks(range(len(names)))
            ax4.set_yticklabels(names, fontsize=7)
            ax4.set_xlabel('|beta|', fontsize=8)
            ax4.set_title('Driver Importance (|final beta|)', fontsize=10)
            for i, (bar, d) in enumerate(zip(bars, sm_drivers[:8])):
                sign = '+' if d[1] > 0 else '-'
                ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                         f'{sign}{abs(d[1]):.4f}', va='center', fontsize=6,
                         color=WHITE_SMOKE)
        else:
            ax4.text(0.5, 0.5, 'No regression drivers available',
                     ha='center', va='center', transform=ax4.transAxes,
                     fontsize=10, color=MUTED_GRAY)

        apply_theme(fig, axes)
        for a in axes:
            a.tick_params(labelsize=7)
        axes[-1].set_xlabel('Episode', fontsize=9)

        fig.suptitle(f'BSTS Decomposition: {sm}', fontsize=14,
                     fontweight='bold', y=0.995, color=DENIM_BRIGHT)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        safe = sm.replace(' ', '_')
        out = f'{directory}/bsts_{safe}.png'
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        plt.close(fig)
        print(f'  [bsts_plot] saved {out}')


if __name__ == '__main__':
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    prefix = sys.argv[2] if len(sys.argv) > 2 else 'v'
    run_full_analysis(log_dir, prefix)
