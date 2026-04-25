"""BSTS module: canonical home for BSTSFeedback + BSTSSeasonal.

Classes
-------
BSTSFeedback   -- EMA-smoothed, Kalman-informed, race_type-aware reward-weight adjuster.
                  Transplanted from run.py (v1.0.13).  run.py now imports it from here.
BSTSSeasonal   -- Batch BSTS decomposition + live per-step buffer for run.py integration.

BSTSFeedback race_type support (v1.0.13)
-----------------------------------------
The constructor now accepts ``race_type: str`` (default ``"TIME_TRIAL"``).
Supported values (case-insensitive): TIME_TRIAL, OBJECT_AVOIDANCE, HEAD_TO_BOT.

Adjustments by race_type:
  TIME_TRIAL      – baseline behaviour (unchanged from run.py original)
  OBJECT_AVOIDANCE – obstacle weight floor raised; braking weight multiplier +50%;
                     avg_safe_speed_ratio threshold tightened (1.1 → 1.05)
  HEAD_TO_BOT     – obstacle weight floor raised even higher; steering weight
                     boosted; speed_steering coupled earlier; corner weight up

adjust_weights() now normalises all returned weights so they always sum to 1.0.

BSTSSeasonal v1.0.13 changes (unchanged from prior version)
------------------------------------------------------------
  - record_step() buffers per-step data in a rolling deque (no longer a no-op)
  - get_trend()    returns real phase/slope computed from last fit_from_jsonl decomposition
  - get_season()   returns real worst_segments + per-segment crash/speed map
  - get_seasonal() alias added (run.py uses this name in one call-site)
  - fit_from_jsonl() caches decomposition results into self._last_results
  - _fit_from_step_buffer() added: periodic fit from record_step() buffer without jsonl

REF: Scott, S. L. & Varian, H. R. (2014). Predicting the present with Bayesian
     structural time series. Int. J. Math. Model. Numer. Optim., 5(1-2), 4-23.
REF: Brodersen, K. H. et al. (2015). Inferring causal impact using Bayesian
     structural time-series models. Ann. Appl. Stat., 9(1), 247-274.
"""
import json, os, math, collections
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Denim theme palette (shared by BSTSSeasonal plots)
# ---------------------------------------------------------------------------
DENIM_DARK   = '#1B3A5C'
DENIM_MID    = '#3A6B8C'
DENIM_BRIGHT = '#5B9BD5'
AMBER        = '#D4A03C'
TERRA_COTTA  = '#C75B39'
GOLD_WARM    = '#E8C167'
SAGE_GREEN   = '#6BB38A'
MUTED_PURPLE = '#9B6EB7'
BG_DARK      = '#2E2E2E'
BG_LIGHT     = '#F0EDE6'
MUTED_GRAY   = '#8C8C8C'
WHITE_SMOKE  = '#F5F5F5'

COMPLEMENTARY_PAIRS = [
    (DENIM_BRIGHT, AMBER),
    (SAGE_GREEN, TERRA_COTTA),
    (MUTED_PURPLE, GOLD_WARM),
    (DENIM_MID, TERRA_COTTA),
]

# Harmonized with analyze_logs.py (single source of truth for run.py + live_bsts_plot.py)
from analyze_logs import SUCCESS_METRICS as SUCCESS_KEYS, INTERMEDIARY_METRICS as REGRESSOR_KEYS

# ---------------------------------------------------------------------------
# Normalised race_type token
# ---------------------------------------------------------------------------
_RACE_TYPE_MAP = {
    "time_trial":        "TIME_TRIAL",
    "timetrial":         "TIME_TRIAL",
    "tt":                "TIME_TRIAL",
    "object_avoidance":  "OBJECT_AVOIDANCE",
    "objectavoidance":   "OBJECT_AVOIDANCE",
    "oa":                "OBJECT_AVOIDANCE",
    "head_to_bot":       "HEAD_TO_BOT",
    "headtobot":         "HEAD_TO_BOT",
    "h2b":               "HEAD_TO_BOT",
    "h2h":               "HEAD_TO_BOT",
}

def _norm_race_type(raw: str) -> str:
    """Return canonical race_type token, defaulting to TIME_TRIAL."""
    return _RACE_TYPE_MAP.get(str(raw).lower().replace(" ", "_"), "TIME_TRIAL")


# ===========================================================================
# BSTSFeedback  (transplanted from run.py v1.0.13, now race_type-aware)
# ===========================================================================
class BSTSFeedback:
    """EMA-smoothed, Kalman-informed reward-weight adjuster.

    Parameters
    ----------
    ema_alpha         : float   EMA decay for metric smoothing (default 0.05)
    feedback_strength : float   Scale of weight adjustments (default 0.15)
    race_type         : str     One of TIME_TRIAL | OBJECT_AVOIDANCE | HEAD_TO_BOT
                                Case-insensitive; aliases like 'oa', 'h2h' accepted.

    Public API
    ----------
    update(metrics_dict)            -> None   Feed new scalar metrics into EMA.
    adjust_weights(base_weights)    -> dict   Return re-normalised adjusted weights.
    set_race_type(race_type: str)   -> None   Hot-swap race_type mid-training.

    Attributes written by caller (run.py / analyze_logs integration)
    ---------------------------------------------------------------
    feedback.kf_trends : dict[str, float]   Kalman trend slopes per metric
    feedback.kf_betas  : dict[str, float]   Kalman regression coefficients
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        feedback_strength: float = 0.15,
        race_type: str = "TIME_TRIAL",
    ):
        self.ema      = {}
        self.alpha    = float(ema_alpha)
        self.strength = float(feedback_strength)
        self.race_type = _norm_race_type(race_type)

        # Populated externally by analyze_logs / BSTSKalmanFilter integration
        self.kf_trends: dict = {}
        self.kf_betas:  dict = {}

    def set_race_type(self, race_type: str) -> None:
        """Hot-swap race_type (e.g. when the 9-phase orchestrator changes phase)."""
        self.race_type = _norm_race_type(race_type)

    # ------------------------------------------------------------------
    def update(self, metrics_dict: dict) -> None:
        """Feed new scalar metrics into EMA state."""
        for k, v in metrics_dict.items():
            if not isinstance(v, (int, float)):
                continue                    # skip non-scalar BSTS metrics
            v = float(v)
            if k not in self.ema:
                self.ema[k] = v
            else:
                self.ema[k] = self.alpha * v + (1.0 - self.alpha) * self.ema[k]

    # ------------------------------------------------------------------
    def adjust_weights(self, base_weights) -> dict:
        """Return re-normalised reward weights adjusted by BSTS feedback.

        Applies three layers of adjustment in order:
        1. Kalman trend / regression signals (if kf_trends / kf_betas set)
        2. EMA metric thresholds (crash_rate, offtrack_rate, speed, etc.)
        3. Race-type-specific multipliers (OBJECT_AVOIDANCE / HEAD_TO_BOT)

        Always returns a dict that sums to 1.0.
        """
        # --- coerce base weights to float dict ---
        if isinstance(base_weights, dict):
            w = {k: float(v) if isinstance(v, (int, float)) else 0.1
                 for k, v in base_weights.items()}
        else:
            w = {}

        s  = self.strength
        cr  = self.ema.get("crash_rate",        0.0)
        otr = self.ema.get("offtrack_rate",      0.0)
        spd = self.ema.get("avg_speed",          2.0)
        ccr = self.ema.get("corner_crash_rate",  0.0)

        # ---- Layer 1: Kalman trend signals ----
        for metric, trend_val in self.kf_trends.items():
            if metric == "crash_rate" and trend_val > 0.01:
                w["obstacle"] = w.get("obstacle", 0.1) + s * 0.3
                w["center"]   = w.get("center",   0.1) + s * 0.2
            elif metric == "off_track_rate" and trend_val > 0.01:
                w["center"]  = w.get("center",  0.1) + s * 0.4
                w["heading"] = w.get("heading", 0.1) + s * 0.3

        # ---- Layer 1b: Kalman regression coefficients ----
        for metric, beta in self.kf_betas.items():
            if "perp_velocity" in metric and abs(beta) > 0.1:
                w["speed_steering"] = w.get("speed_steering", 0.08) + s * abs(beta) * 0.2
            if "brake_zone" in metric and abs(beta) > 0.1:
                w["progress"] = w.get("progress", 0.12) + s * abs(beta) * 0.15

        # ---- Layer 2: EMA threshold rules ----
        if cr > 0.3:
            b = s * min(cr, 1.0)
            w["obstacle"] = w.get("obstacle", 0.1) + b * 0.5
            w["center"]   = w.get("center",   0.1) + b * 0.3
        if otr > 0.2:
            b = s * min(otr, 1.0)
            w["center"]  = w.get("center",  0.1) + b * 0.4
            w["heading"] = w.get("heading", 0.1) + b * 0.3
        if spd < 1.5:
            w["curv_speed"] = w.get("curv_speed", 0.15) + s * 0.3
        if ccr > 0.2:
            w["corner"] = w.get("corner", 0.1) + s * min(ccr, 1.0) * 0.5

        ssr = self.ema.get("avg_safe_speed_ratio", 1.0)
        ter = self.ema.get("avg_turn_entry_ratio", 1.0)
        rle = self.ema.get("avg_racing_line_err",  0.0)

        _ssr_thresh = {
            "TIME_TRIAL":       1.2,
            "OBJECT_AVOIDANCE": 1.05,   # tighter — obstacles demand precise speed
            "HEAD_TO_BOT":      1.1,
        }.get(self.race_type, 1.2)

        if ssr > _ssr_thresh:
            w["safe_speed"] = w.get("safe_speed", 0.06) + s * min(ssr - 1.0, 0.5) * 0.3
        if ter > 1.1:
            w["safe_speed"] = w.get("safe_speed", 0.06) + s * 0.2
        if rle > 0.4:
            w["racing_line"] = w.get("racing_line", 0.04) + s * min(rle, 1.0) * 0.2

        # ---- Layer 2b: harmonised v27 metric taxonomy ----
        def _ema(k, d=0.0):
            return float(self.ema.get(k, d))

        def _slope(k):
            try:
                return float(self.kf_trends.get(k, 0.0))
            except Exception:
                return 0.0

        if _ema("race_line_gradient_compliance", 0.5) < 0.5 or _slope("race_line_gradient_compliance") < -0.005:
            w["racing_line"] = w.get("racing_line", 0.04) + s * 0.25
        if _ema("avg_speed_centerline", 0.0) > 0 and _slope("avg_speed_centerline") < -0.01:
            w["progress"] = w.get("progress", 0.12) + s * 0.20
        if _slope("jerk_rms") < -0.01:
            w["jerk"] = w.get("jerk", 0.03) + s * 0.15
        if _slope("late_corner_entry") < -0.01 or _slope("early_corner_exit") < -0.01:
            w["corner"] = w.get("corner", 0.10) + s * 0.30
        if _ema("brake_field_compliance", 0.5) < 0.4:
            w["braking"] = w.get("braking", 0.08) + s * 0.25
        if _slope("steer_speed_coordination") < -0.01:
            w["speed_steering"] = w.get("speed_steering", 0.08) + s * 0.15
        if _slope("waypoint_coverage") < -0.005:
            w["progress"] = w.get("progress", 0.12) + s * 0.10

        # ---- Layer 3: race_type multipliers ----
        if self.race_type == "OBJECT_AVOIDANCE":
            # Obstacle awareness and braking are critical; raise their floor
            w["obstacle"] = max(w.get("obstacle", 0.0), 0.12) * 1.3
            w["braking"]  = w.get("braking",  0.08) * 1.5
            w["center"]   = w.get("center",   0.10) * 1.1

        elif self.race_type == "HEAD_TO_BOT":
            # Need both obstacle avoidance AND speed — steering precision matters more
            w["obstacle"]       = max(w.get("obstacle", 0.0), 0.15) * 1.5
            w["steering"]       = w.get("steering",       0.02) * 2.0
            w["speed_steering"] = w.get("speed_steering", 0.08) * 1.3
            w["corner"]         = w.get("corner",         0.10) * 1.2

        # ---- Final normalisation — always sums to 1.0 ----
        total = sum(w.values())
        if total > 0:
            return {k: v / total for k, v in w.items()}
        return w


# ===========================================================================
# Helper functions (shared by BSTSSeasonal internals)
# ===========================================================================
def _apply_denim():
    rcParams.update({
        'figure.facecolor': BG_LIGHT,
        'axes.facecolor':   WHITE_SMOKE,
        'axes.edgecolor':   DENIM_DARK,
        'axes.labelcolor':  DENIM_DARK,
        'xtick.color':      DENIM_DARK,
        'ytick.color':      DENIM_DARK,
        'text.color':       DENIM_DARK,
        'grid.color':       MUTED_GRAY,
        'grid.alpha':       0.3,
        'axes.grid':        True,
        'font.family':      'sans-serif',
    })


def _load_jsonl(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _ewma(x, alpha=0.02):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _ols(X, y):
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta, X @ beta
    except Exception:
        return np.zeros(X.shape[1]), np.zeros_like(y)


# ===========================================================================
# BSTSSeasonal  (batch decomposer — unchanged from v1.0.13 prior version)
# ===========================================================================
class BSTSSeasonal:
    """Batch BSTS decomposition + live per-step buffer for run.py integration.

    Constructor:
        BSTSSeasonal(n_segments=12, save_dir='results', alpha=0.02)
    """

    STEP_BUFFER_MAXLEN    = 5000
    STEP_BUFFER_MIN_EPISODES = 20
    TREND_WINDOW          = 30

    def __init__(self, n_segments=12, save_dir='results', alpha=0.02):
        self.n_segments = int(n_segments)
        self.save_dir   = str(save_dir)
        self.alpha      = float(alpha)
        os.makedirs(self.save_dir, exist_ok=True)

        self._step_buf:    collections.deque = collections.deque(maxlen=self.STEP_BUFFER_MAXLEN)
        self._ep_buf:      list  = []
        self._current_ep:  dict  = {}
        self._ep_count:    int   = 0
        self._last_results: dict = {}
        self._cached_trend: dict = {}
        self._cached_season: dict = {'worst_segments': [], 'segments': {}}

    # ------------------------------------------------------------------
    # Per-step live buffer
    # ------------------------------------------------------------------
    def record_step(
        self,
        progress:     float = 0.0,
        speed:        float = 0.0,
        steering:     float = 0.0,
        heading_err:  float = 0.0,
        raceline_err: float = 0.0,
        reward:       float = 0.0,
        context=None,
        action=None,
        lidar_min:    float = 0.0,
        wp_idx:       int   = 0,
        **kwargs,
    ):
        """Buffer one step of telemetry for periodic BSTS fitting."""
        self._step_buf.append({
            'progress':     float(progress),
            'speed':        float(speed),
            'steering':     float(steering),
            'heading_err':  float(heading_err),
            'raceline_err': float(raceline_err),
            'reward':       float(reward),
            'lidar_min':    float(lidar_min),
            'wp_idx':       int(wp_idx),
        })
        ep = self._current_ep
        ep.setdefault('rewards', []).append(float(reward))
        ep.setdefault('speeds',  []).append(float(speed))
        ep.setdefault('rl_errs', []).append(float(raceline_err))
        ep.setdefault('max_prog', 0.0)
        if float(progress) > ep['max_prog']:
            ep['max_prog'] = float(progress)

    def _flush_episode(self, lap_completed: bool = False):
        """Commit current episode summary to _ep_buf."""
        ep = self._current_ep
        if not ep.get('rewards'):
            self._current_ep = {}
            return
        self._ep_count += 1
        rewards = ep['rewards']
        speeds  = ep['speeds']
        rl_errs = ep['rl_errs']
        self._ep_buf.append({
            'episode':              self._ep_count,
            'ep_return':            float(sum(rewards)),
            'position_in_lap':      float(ep.get('max_prog', 0.0)) / 100.0,
            'lap_completed':        float(bool(lap_completed)),
            'avg_speed_centerline': float(np.mean(speeds)) if speeds else 0.0,
            'avg_racing_line_err':  float(np.mean(rl_errs)) if rl_errs else 0.0,
            'offtrack_rate':        float(ep.get('offtrack_rate', 0.0)),
            'avg_jerk':             float(ep.get('avg_jerk', 0.0)),
            'avg_safe_speed_ratio': float(ep.get('avg_safe_speed_ratio', 1.0)),
        })
        self._current_ep = {}

    def _fit_from_step_buffer(self):
        rows = self._ep_buf
        if len(rows) < self.STEP_BUFFER_MIN_EPISODES:
            return
        self._fit_rows(rows)

    def _fit_rows(self, rows: list):
        if len(rows) < 10:
            return
        ep = np.array([r.get('episode', i) for i, r in enumerate(rows)], dtype=float)
        X_cols, reg_used = [], []
        for k in REGRESSOR_KEYS:
            col = np.array([r.get(k, np.nan) for r in rows], dtype=float)
            if np.isfinite(col).sum() > len(col) * 0.2:
                med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
                col = np.where(np.isfinite(col), col, med)
                mu, sd = col.mean(), col.std() + 1e-9
                X_cols.append((col - mu) / sd)
                reg_used.append(k)
        if not X_cols:
            return
        X = np.column_stack([np.ones(len(rows))] + X_cols)
        results = {'regressors': reg_used, 'series': {}}
        for sk in SUCCESS_KEYS:
            y = np.array([r.get(sk, np.nan) for r in rows], dtype=float)
            if not np.isfinite(y).any():
                continue
            med = np.nanmedian(y)
            y   = np.where(np.isfinite(y), y, med)
            trend = _ewma(y, self.alpha)
            detr  = y - trend
            e_min, e_max = ep.min(), ep.max() + 1e-9
            seg_idx = np.clip(
                ((ep - e_min) / (e_max - e_min) * self.n_segments).astype(int),
                0, self.n_segments - 1
            )
            season, seg_means = np.zeros_like(y), {}
            for s in range(self.n_segments):
                mask = seg_idx == s
                if mask.any():
                    m = detr[mask].mean()
                    season[mask] = m
                    seg_means[s] = m
            resid_ts     = y - trend - season
            beta, yhat_r = _ols(X, resid_ts)
            residual     = resid_ts - yhat_r
            results['series'][sk] = dict(
                y=y, trend=trend, season=season, regression=yhat_r,
                residual=residual, beta=beta, seg_means=seg_means, ep=ep,
            )
        self._last_results = results
        self._update_trend_cache(results)
        self._update_season_cache(results, rows)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _update_trend_cache(self, results: dict):
        sk = 'ep_return'
        if sk not in results.get('series', {}):
            keys = list(results.get('series', {}).keys())
            if not keys:
                return
            sk = keys[0]
        d     = results['series'][sk]
        trend = d['trend']
        window = min(len(trend), self.TREND_WINDOW)
        if window < 4:
            self._cached_trend = {}
            return
        recent = trend[-window:]
        slope, _ = np.polyfit(np.arange(window, dtype=float), recent, 1)
        rel_slope = slope / (abs(np.mean(recent)) + 1e-9)
        if   rel_slope >  0.005: phase = 'improving'
        elif rel_slope < -0.005: phase = 'declining'
        else:                    phase = 'plateau'
        self._cached_trend = {
            'phase':        phase,
            'return_slope': float(rel_slope),
            'trend_last':   float(trend[-1]),
            'trend_mean':   float(np.mean(recent)),
        }

    def _update_season_cache(self, results: dict, rows: list):
        sk = 'ep_return'
        if sk not in results.get('series', {}):
            keys = list(results.get('series', {}).keys())
            if not keys:
                return
            sk = keys[0]
        d         = results['series'][sk]
        seg_means = d.get('seg_means', {})
        if not seg_means:
            return
        sorted_segs    = sorted(seg_means.items(), key=lambda kv: kv[1])
        worst_segments = [int(sid) for sid, _ in sorted_segs[:4]]
        ep_arr  = d['ep']
        e_min, e_max = ep_arr.min(), ep_arr.max() + 1e-9
        seg_idx_all = np.clip(
            ((ep_arr - e_min) / (e_max - e_min) * self.n_segments).astype(int),
            0, self.n_segments - 1
        )
        segments_info = {}
        for seg_id in range(self.n_segments):
            mask = seg_idx_all == seg_id
            if not mask.any():
                continue
            seg_rows  = [rows[i] for i in np.where(mask)[0]]
            crashes   = sum(1 for r in seg_rows if r.get('lap_completed', 1.0) < 0.5)
            avg_speed = float(np.mean([r.get('avg_speed_centerline', 0.0) for r in seg_rows]))
            segments_info[seg_id] = {
                'crashes':   int(crashes),
                'avg_speed': avg_speed,
                'n_eps':     int(len(seg_rows)),
            }
        self._cached_season = {
            'worst_segments': worst_segments,
            'segments':       segments_info,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_trend(self) -> dict:
        """Return latest trend dict.  Keys: phase, return_slope, trend_last, trend_mean."""
        return dict(self._cached_trend)

    def get_season(self) -> dict:
        """Return season dict.  Keys: worst_segments, segments."""
        return dict(self._cached_season)

    def get_seasonal(self) -> dict:
        """Alias for get_season() — run.py uses this name in one call-site."""
        return self.get_season()

    # ------------------------------------------------------------------
    # Batch JSONL fit
    # ------------------------------------------------------------------
    def fit_from_jsonl(self, jsonl_path, out_png=None):
        rows = _load_jsonl(jsonl_path)
        if len(rows) < 10:
            print(f'[BSTS] too few rows ({len(rows)}) in {jsonl_path}')
            return None
        all_rows = rows + self._ep_buf
        if out_png is None:
            out_png = os.path.join(self.save_dir, 'bsts_decomposition.png')
        self._fit_rows(all_rows)
        results = self._last_results
        if not results:
            return None
        _apply_denim()
        ep = np.array([r.get('episode', i) for i, r in enumerate(all_rows)], dtype=float)
        self._plot(results, ep, out_png)
        for sk, d in results.get('series', {}).items():
            reg_used = results.get('regressors', [])
            coef_str = ', '.join(
                f'{k}={b:+.3f}' for k, b in zip(['int'] + reg_used, d['beta'])
            )
            print(
                f'[BSTS] {sk}: y_mean={d["y"].mean():.3f} '
                f'trend_last={d["trend"][-1]:.3f} '
                f'season_amp={float(np.ptp(list(d["seg_means"].values()))):.3f} '
                f'resid_std={d["residual"].std():.3f} | {coef_str}'
            )
        print(f'[BSTS] trend_cache={self._cached_trend}')
        return results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _plot(self, results, ep, out_png):
        if not HAS_MPL:
            return
        series = results['series']
        n_s    = len(series)
        if n_s == 0:
            return
        fig, axes = plt.subplots(4, n_s, figsize=(5.5 * n_s, 11), squeeze=False)
        panel_titles = [
            'Observed + Trend',
            'Season (per-restart)',
            'Regression (intermediary)',
            'Residual',
        ]
        reg_used = results['regressors']
        for j, (sk, d) in enumerate(series.items()):
            c1, c2 = COMPLEMENTARY_PAIRS[j % len(COMPLEMENTARY_PAIRS)]
            axes[0, j].plot(ep, d['y'],     lw=0.5, alpha=0.35, color=MUTED_GRAY, label='observed')
            axes[0, j].plot(ep, d['trend'], lw=2.2, color=c1,   label='trend')
            axes[0, j].set_title(f'{sk}\n{panel_titles[0]}', fontsize=9, fontweight='bold', color=DENIM_DARK)
            axes[0, j].legend(loc='best', fontsize=7, framealpha=0.8)
            axes[1, j].bar(
                range(self.n_segments),
                [d['seg_means'].get(s, 0) for s in range(self.n_segments)],
                color=c2, alpha=0.7, edgecolor=DENIM_DARK, linewidth=0.5,
            )
            axes[1, j].axhline(0, color=DENIM_DARK, lw=0.5)
            axes[1, j].set_title(panel_titles[1], fontsize=9, color=DENIM_DARK)
            axes[1, j].set_xlabel('restart segment', fontsize=7)
            axes[2, j].plot(ep, d['regression'], lw=1.2, color=TERRA_COTTA)
            coef_txt = '\n'.join(f'{k}: {b:+.3f}' for k, b in zip(reg_used, d['beta'][1:]))
            axes[2, j].text(
                0.02, 0.97, coef_txt, transform=axes[2, j].transAxes,
                fontsize=6.5, va='top', family='monospace', color=DENIM_DARK,
                bbox=dict(boxstyle='round', facecolor=BG_LIGHT, alpha=0.85, edgecolor=DENIM_MID),
            )
            axes[2, j].set_title(panel_titles[2], fontsize=9, color=DENIM_DARK)
            axes[3, j].plot(ep, d['residual'], lw=0.5, color=MUTED_GRAY, alpha=0.6)
            axes[3, j].axhline(0, color=DENIM_DARK, lw=0.5)
            axes[3, j].set_title(panel_titles[3], fontsize=9, color=DENIM_DARK)
            axes[3, j].set_xlabel('episode', fontsize=8)
        fig.suptitle(
            'BSTS Decomposition: Success ~ Trend + Season(per-restart) + Regression(intermediary) + Residual',
            fontsize=11, fontweight='bold', color=DENIM_DARK, y=0.99,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[BSTS-PLOT] saved {out_png}')
