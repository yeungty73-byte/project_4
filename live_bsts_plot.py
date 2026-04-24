#!/usr/bin/env python3
"""Unified live monitoring: BSTS trend diagnostics + dashboard + metrics CLI.

Merges formerly separate live_bsts_plot.py, live_dashboard.py, live_metrics.py.

METRIC TAXONOMY (all trend-up = good, per operationalization in run.py):

  SUCCESS_METRICS  (terminal racing outcomes the agent must achieve)
    progress                      : Track progress fraction in [0,1]
    avg_speed_centerline          : Mean speed weighted by centerline proximity
                                    (proxy for lap time, decoupled from distance)
    race_line_gradient_compliance : Episode-mean gradient alignment with dynamic
                                    optimal race line (Cardamone 2010)

  INTERMEDIARY  (hypothesized drivers, all trend-up = good)
    brake_field_compliance   : Alignment with brake_field gradient/potential
    jerk_rms                 : RMS longitudinal jerk (intentional dynamics)
    speed_mean               : Episode mean speed
    steer_activity           : RMS steering magnitude (attack indicator)
    waypoint_coverage        : Fraction of waypoints traversed
    speed_steer_waypoint     : speed * |steer| * dwp interaction
    steer_speed_coordination : Slow-in/fast-out coordination index
    late_corner_entry        : Pre-apex proximity to outer curb (flipped)
    early_corner_exit        : Line-of-sight distance exiting apex

References (see references.bib):
  Scott & Varian 2014; Brodersen et al. 2015; Durbin & Koopman 2012;
  Ng, Harada & Russell 1999; Remonda et al. 2025; Song et al. 2023;
  Lee et al. 2024; Feng & Manser 2017; Paradigm Shift Racing 2020/2023.
"""
import argparse, json, os, sys, glob, math
from pathlib import Path
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# =============================================================================
# METRIC TAXONOMY
# =============================================================================
# Tuples: (key, label, units, bound, color, direction)
# direction: +1 means "up is good" (all operationalized this way, per spec)

SUCCESS_METRICS = [
    ("progress",                      "Track Progress",          "frac",  (0, 1), "#8B7FBF", +1),
    ("avg_speed_centerline",          "Avg Speed (centerline-w.)", "m/s", None,   "#7FBFBF", +1),
    ("race_line_gradient_compliance", "Race-Line Gradient Comp.", "ratio", (0, 1), "#BF8B7F", +1),
]

INTERMEDIARY_METRICS = [
    ("brake_field_compliance",      "Brake-Field Compliance",      "ratio", (0, 1), "#8B7FBF", +1),
    ("jerk_rms",                    "Smoothness (1-Jerk RMS)",     "1-rms", (0, 1), "#7FBFBF", +1),
    ("speed_mean",                  "Speed (mean)",                "m/s", None,  "#BF8B7F", +1),
    ("steer_activity",              "Steer Activity (RMS)",        "rad", None,  "#7FBF8B", -1),
    ("waypoint_coverage",           "Waypoint Coverage",           "frac", (0, 1), "#BFBF7F", +1),
    ("gg_ellipse_utilisation",      "GG Ellipse Utilisation",      "ratio", (0, 1), "#BF7FBF", +1),
    ("trail_braking_quality",       "Trail Braking Quality",       "ratio", (0, 1), "#7FB5BF", +1),
    ("velocity_profile_compliance", "Velocity Profile Compliance", "ratio", (0, 1), "#BFA07F", +1),
    ("curvature_anticipation",      "Curvature Anticipation",      "ratio", (0, 1), "#9FBF7F", +1),
    ("heading_alignment_mean",      "Heading Alignment",           "ratio", (0, 1), "#7F9FBF", +1),
]

ALL_METRICS = SUCCESS_METRICS + INTERMEDIARY_METRICS

# =============================================================================
# I/O
# =============================================================================

def load_jsonl(log_dir):
    """Load all .jsonl episode rows from results/ (or any dir)."""
    pattern = os.path.join(log_dir, "*.jsonl")
    files = sorted(glob.glob(pattern))
    rows = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
    rows.sort(key=lambda r: (r.get("global_step", 0), r.get("episode", 0)))
    return rows


# =============================================================================
# STATISTICAL HELPERS (Durbin & Koopman 2012; Scott & Varian 2014)
# =============================================================================

def ema(arr, alpha=0.08):
    if len(arr) == 0:
        return np.array([])
    a = np.asarray(arr, dtype=float)
    out = np.empty_like(a)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1.0 - alpha) * out[i-1]
    return out


def local_slope(arr, window=30):
    """Rolling least-squares slope (trend signal)."""
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if n < 2:
        return np.zeros(n)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        x = np.arange(lo, i + 1, dtype=float)
        y = a[lo:i+1]
        if len(x) < 2 or np.all(np.isnan(y)):
            out[i] = 0.0
            continue
        x_ = x - x.mean()
        y_ = y - np.nanmean(y)
        denom = (x_ * x_).sum()
        out[i] = (x_ * y_).sum() / denom if denom > 0 else 0.0
    return out


def simple_trend(arr, window=20):
    """Sign of slope over last `window` samples, with magnitude."""
    a = np.asarray(arr, dtype=float)
    if len(a) < 3:
        return 0.0
    w = min(window, len(a))
    y = a[-w:]
    x = np.arange(w, dtype=float)
    x_ = x - x.mean()
    y_ = y - np.nanmean(y)
    denom = (x_ * x_).sum()
    return float((x_ * y_).sum() / denom) if denom > 0 else 0.0

# =============================================================================
# BSTS-STYLE TREND DIAGNOSIS
# =============================================================================

def diagnose(rows, window=30):
    """Return diag dict consumed by run.py's BSTSFeedback.adjust_weights.

    Extended: regress each success metric EMA on intermediary EMAs and keep
    top-3 by |beta|.  Exposed as diag["regression"][success_key].
    """
    diag = {"metrics": {}, "window": int(window)}
    emas = {}
    for key, label, units, bound, color, direction in (SUCCESS_METRICS + INTERMEDIARY_METRICS):
        vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        mask = ~np.isnan(vals)
        if mask.sum() < 3:
            continue
        v = vals[mask]
        e = ema(v, alpha=0.08)
        s = simple_trend(e, window=window)
        scale = max(1e-6, float(np.nanstd(e)))
        rel = float(s / scale)
        diag["metrics"][key] = {"value": float(e[-1]), "slope": float(s),
                                  "trend": int(np.sign(s)), "rel": rel,
                                  "direction": direction}
        emas[key] = e
    diag["success_trend"] = float(np.mean([diag["metrics"][k]["rel"] for k,*_ in SUCCESS_METRICS if k in diag["metrics"]] or [0.0]))
    diag["intermediary_trend"] = float(np.mean([diag["metrics"][k]["rel"] for k,*_ in INTERMEDIARY_METRICS if k in diag["metrics"]] or [0.0]))
    diag["regression"] = {}
    int_keys = [k for k,*_ in INTERMEDIARY_METRICS if k in emas]
    if int_keys:
        L = min(len(emas[k]) for k in int_keys)
        X_raw = np.column_stack([emas[k][-L:] for k in int_keys])
        mu = X_raw.mean(axis=0); sd = X_raw.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        X = (X_raw - mu) / sd
        for skey,*_ in SUCCESS_METRICS:
            if skey not in emas: continue
            y = emas[skey][-L:]
            if len(y) < max(5, len(int_keys)+1): continue
            y_c = y - y.mean()
            try:
                beta, *_r = np.linalg.lstsq(X, y_c, rcond=None)
                y_hat = X @ beta
                ss_res = float(np.sum((y_c - y_hat)**2))
                ss_tot = float(np.sum(y_c**2)) or 1e-12
                r2 = 1.0 - ss_res/ss_tot
            except Exception:
                continue
            ranked = sorted(zip(int_keys, beta.tolist()), key=lambda kb: abs(kb[1]), reverse=True)[:3]
            diag["regression"][skey] = [{"intermediary": k, "beta": float(b), "r2": float(r2)} for k,b in ranked]
    return diag


# =============================================================================
# PLOTTING
# =============================================================================

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.3)


def _plot_metric(ax, rows, key, label, units, bound, color):
    vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    x = np.arange(len(vals))
    ax.plot(x, vals, color=color, alpha=0.25, linewidth=0.8, label="raw")
    e = ema(vals)
    ax.plot(x, e, color=color, linewidth=2.0, label="EMA")
    if bound is not None:
        ax.set_ylim(bound[0], bound[1])
    ttl = label + (f" [{units}]" if units else "")
    ax.set_title(ttl, fontsize=9)
    ax.set_xlabel("episode", fontsize=7)
    _style_ax(ax)
    ax.tick_params(axis="both", labelsize=7)


def plot_bsts(rows, out_path, diag=None):
    """BSTS feature-selection dashboard: 1x3 success panels, each overlaid
    with top-3 intermediary predictors colored by regression rank."""
    if not _HAS_MPL:
        print("[bsts] matplotlib unavailable; skipping")
        return
    if diag is None:
        diag = diagnose(rows)
    reg = diag.get("regression", {})
    rank_palette = ["#d62728", "#2ca02c", "#9467bd"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    for ax, (skey, slabel, sunits, sbound, scolor, _sd) in zip(axes, SUCCESS_METRICS):
        y = np.array([r.get(skey, np.nan) for r in rows], dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            ax.set_title(f"{slabel} (insufficient data)", fontsize=10); _style_ax(ax); continue
        y_v = y[mask]; y_e = ema(y_v, alpha=0.08); x = np.arange(len(y_e))
        ax.plot(x, y_v, color=scolor, alpha=0.18, linewidth=0.8, label=f"{slabel} raw")
        ax.plot(x, y_e, color=scolor, linewidth=2.4, label=f"{slabel} EMA")
        y_lo, y_hi = float(np.nanmin(y_e)), float(np.nanmax(y_e))
        y_rng = (y_hi - y_lo) or 1.0
        for rank, item in enumerate(reg.get(skey, [])[:3]):
            ikey = item["intermediary"]; beta = item["beta"]; r2 = item["r2"]
            iv = np.array([r.get(ikey, np.nan) for r in rows], dtype=float)
            imask = ~np.isnan(iv)
            if imask.sum() < 3: continue
            ie = ema(iv[imask], alpha=0.08)
            n = min(len(ie), len(y_e)); ie = ie[-n:]
            xi = np.arange(n) + (len(y_e) - n)
            lo, hi = float(np.nanmin(ie)), float(np.nanmax(ie))
            rng = (hi - lo) or 1.0
            if beta >= 0:
                ie_s = y_lo + ((ie - lo)/rng) * y_rng
            else:
                ie_s = y_hi - ((ie - lo)/rng) * y_rng
            ax.plot(xi, ie_s, color=rank_palette[rank], linewidth=1.4, alpha=0.85,
                    label=f"#{rank+1} {ikey} (b={beta:+.2f}, r2={r2:.2f})")
        if sbound is not None: ax.set_ylim(sbound[0], sbound[1])
        ax.set_title(f"{slabel} [{sunits}]" if sunits else slabel, fontsize=11)
        ax.set_xlabel("episode", fontsize=8); ax.legend(fontsize=7, loc="best", framealpha=0.85)
        _style_ax(ax); ax.tick_params(axis="both", labelsize=7)
    fig.suptitle("BSTS Feature-Selection Dashboard (success ~ top-3 intermediary predictors)", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[bsts] wrote {out_path}")


def plot_dashboard(rows, out_path, diag=None):
    """Operational dashboard: success panel + intermediary panel + trend bars."""
    if not _HAS_MPL:
        print("[dash] matplotlib unavailable; skipping")
        return
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.35)
    # Success row
    for i, (key, label, units, bound, color, _d) in enumerate(SUCCESS_METRICS):
        ax = fig.add_subplot(gs[0, i])
        _plot_metric(ax, rows, key, label, units, bound, color)
        ax.set_title("[SUCCESS] " + label, fontsize=9, fontweight="bold")
    # Trend summary bar chart (rightmost)
    ax_bar = fig.add_subplot(gs[0, 3])
    if diag:
        names, trends = [], []
        for key, lab, _u, _b, _c, _d in ALL_METRICS:
            m = diag["metrics"].get(key, {})
            if m.get("value") is None:
                continue
            names.append(lab.split()[0][:10])
            trends.append(m["rel"])
        y = np.arange(len(names))
        ax_bar.barh(y, trends, color=["#7FBF8B" if t >= 0 else "#BF7F7F" for t in trends])
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(names, fontsize=6)
        ax_bar.axvline(0, color="k", linewidth=0.5)
        ax_bar.set_title("Normalized trend (up=good)", fontsize=9)
        _style_ax(ax_bar)
    # Intermediary grid
    for i, (key, label, units, bound, color, _d) in enumerate(INTERMEDIARY_METRICS[:8]):
        r, c = 1 + (i // 4), i % 4
        ax = fig.add_subplot(gs[r, c])
        _plot_metric(ax, rows, key, label, units, bound, color)
    fig.suptitle("DeepRacer Live Dashboard -- success vs intermediary metrics",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[dash] wrote {out_path}")

# =============================================================================
# TEXT-MODE ANALYZER (replaces live_metrics.py)
# =============================================================================

def analyze(log_dir, window=30, json_out=False):
    rows = load_jsonl(log_dir)
    diag = diagnose(rows, window=window)
    if json_out:
        print(json.dumps(diag, indent=2, default=float))
        return diag
    print(f"[analyze] n_episodes = {diag.get('n_episodes',0)}  window = {window}")
    print(f"[analyze] success_trend      = {diag.get('success_trend', 0.0):+.3f}")
    print(f"[analyze] intermediary_trend = {diag.get('intermediary_trend', 0.0):+.3f}")
    print("")
    print("SUCCESS metrics (up = good):")
    for key, lab, units, _b, _c, _d in SUCCESS_METRICS:
        m = diag["metrics"].get(key, {})
        v = m.get("value"); r = m.get("rel", 0.0)
        marker = "UP" if r > 0.02 else ("DN" if r < -0.02 else "--")
        vstr = f"{v:8.4f}" if v is not None else "   n/a "
        print(f"  [{marker}] {lab:32s} value={vstr}  rel_slope={r:+.3f}")
    print("")
    print("INTERMEDIARY metrics (up = good):")
    for key, lab, units, _b, _c, _d in INTERMEDIARY_METRICS:
        m = diag["metrics"].get(key, {})
        v = m.get("value"); r = m.get("rel", 0.0)
        marker = "UP" if r > 0.02 else ("DN" if r < -0.02 else "--")
        vstr = f"{v:8.4f}" if v is not None else "   n/a "
        print(f"  [{marker}] {lab:32s} value={vstr}  rel_slope={r:+.3f}")
    return diag


# =============================================================================
# CLI
# =============================================================================

VARIANT_LABELS = {
    "time_trial": "tt",
    "obstacle":   "oa",
    "h2h":        "h2h",
}
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log-dir", default="results", help="dir containing *.jsonl")
    ap.add_argument("--out-dir", default="results/live", help="figure output dir")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--mode", choices=["all", "bsts", "dash", "text", "json"], default="all")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_jsonl(args.log_dir)
    for variant, suffix in VARIANT_LABELS.items():
        vrows = [r for r in rows if r.get("variant") == variant]
        if len(vrows) < 4:
            continue   # not enough data yet for this race type
        vdiag = diagnose(vrows, window=args.window)
        plot_bsts(vrows, os.path.join(args.out_dir, f"dashboard_bsts_{suffix}.png"), diag=vdiag)
        plot_dashboard(vrows, os.path.join(args.out_dir, f"dashboard_{suffix}.png"), diag=vdiag)
        print(f"[live] wrote {suffix} plots ({len(vrows)} episodes)")
    diag = diagnose(rows, window=args.window)
    if args.mode in ("all", "bsts"):
        plot_bsts(rows, os.path.join(args.out_dir, "dashboard_bsts.png"))
    if args.mode in ("all", "dash"):
        plot_dashboard(rows, os.path.join(args.out_dir, "dashboard.png"), diag=diag)
    if args.mode in ("all", "text"):
        analyze(args.log_dir, window=args.window, json_out=False)
    if args.mode == "json":
        print(json.dumps(diag, indent=2, default=float))
    # Emit machine-readable diag for BSTSFeedback.adjust_weights
    with open(os.path.join(args.out_dir, "bsts_diag.json"), "w") as f:
        json.dump(diag, f, indent=2, default=float)


if __name__ == "__main__":
    main()
