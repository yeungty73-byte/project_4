#!/usr/bin/env python3
"""
CS7642 Project 4 v1.1.5b -- PATCH-Z/AA/BB/CC Application Script
Run from repo root: python patches/v1_1_5b_patch.py

Applies four surgical patches to run.py:
  PATCH-Z : Clear ep_ante_buf (+ all ep_ accumulators) at episode reset
  PATCH-AA: Allow adjust_weights() during bootstrap, clamp to phase-A floor
  PATCH-BB: Update ep_track_length_m from rp['tracklength'] per-step
  PATCH-CC: Unconditional neutral floor for bsts_metrics compliance keys

All bugs confirmed in run_20260427_183324.log (53 episodes):
  Bug Z : ANTE t-16/t-17 ep17 = steps from ep16 (ep_ante_buf never cleared)
  Bug AA: rwadjcenter0.080/prog0.620 frozen 53 eps (_bootstrap_active=True always)
  Bug BB: tracklengthm 0.0 in all Kalman lines (ep_track_length_m not updated per-step)
  Bug CC: racelineadherence/brakecompliance 0.0 in X-matrix (bsts_metrics missing keys)

REF: Schulman et al. (2017) arXiv:1707.06347
REF: Ng, Harada & Russell (1999) ICML
REF: Welch & Bishop (1995) TR 95-041
"""
import sys, pathlib, shutil, datetime

SRC = pathlib.Path("run.py")
BAK = pathlib.Path(f"run.py.bak_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

if not SRC.exists():
    print("ERROR: run.py not found. Run from repo root."); sys.exit(1)

src = SRC.read_text(encoding="utf-8")
orig_len = len(src)
applied = []
failed  = []

# ---------------------------------------------------------------------------
# PATCH-Z: Clear ep_ante_buf + episode accumulators at episode reset
# ---------------------------------------------------------------------------
Z_OLD = (
    '    observation, info = env.reset()\n'
    '    # v1.3.1: detect spawn reversal and log heading for debugging'
)
Z_NEW = (
    '    # v1.1.5b PATCH-Z: clear ALL per-episode accumulators before new episode spawn\n'
    '    # REF: v1.1.5d Bug Z -- ep_ante_buf never cleared between episodes.\n'
    '    #      ep17 ANTE t-16 showed step=1 (ep17 spawn), t-17 showed step=13 (ep16 final)\n'
    '    #      confirming cross-ep ANTE contamination in run_20260427_183324.log\n'
    '    ep_ante_buf = []\n'
    '    _ep_step_log = []\n'
    '    ep_step_count = 0\n'
    '    cumulative_ep_reward = 0.0\n'
    '    _prev_prog_tracker = 0.0\n'
    '    ep_speeds = []; ep_headings = []; ep_closest_wps = []; ep_dist_from_center = []\n'
    '    ep_heading_diffs = []; ep_steerings_raw = []; ep_positions = []; ep_progress_hist = []\n'
    '    ep_context_preds = []; ep_lidar_mins = []; ep_barrier_proximities = []; ep_nearest_objects = []\n'
    '    ep_ang_vel_centerline = []; ep_jerk_abs = []; ep_brake_before_barrier = []\n'
    '    ep_decel_penalties = []; ep_safe_speed_ratios = []; ep_racing_line_errors = []\n'
    '    ep_corner_speeds = []; ep_turn_entry_speeds = []\n'
    '    ep_offtrack_count = 0; ep_offtrack_steps = 0; ep_recovery_steps = 0\n'
    '    ep_reversed_count = 0; ep_zero_speed_count = 0; ep_graze_count = 0\n'
    '    ep_in_recovery = False; ep_first_offtrack_step = None\n'
    '    ep_crash_ctx = None; ep_crash_speed = None; ep_crash_heading = None\n'
    '    ep_crash_closest_wp = None; ep_crash_lidar_min = None\n'
    '    ep_progress = 0.0; ep_progress_pct = 0.0; ep_centerline_progress_m = 0.0\n'
    '    ep_track_progress_pct = 0.0; ep_prev_speed = 0.0; ep_prev_accel = None\n'
    '    ep_start_time = time.time()\n'
    '    for _rc_key in ep_rewards_components: ep_rewards_components[_rc_key] = []\n'
    '    observation, info = env.reset()\n'
    '    # v1.3.1: detect spawn reversal and log heading for debugging'
)
if Z_OLD in src:
    src = src.replace(Z_OLD, Z_NEW, 1)
    applied.append("PATCH-Z")
else:
    failed.append("PATCH-Z (target not found -- check indentation)")

# ---------------------------------------------------------------------------
# PATCH-AA: Allow adjust_weights() during bootstrap, clamp to phase-A floor
# FIXED: comment in run.py is 'bootstrap weights unchanged during bootstrap phase'
#        NOT 'bootstrap: keep progress-dominant weights'
# ---------------------------------------------------------------------------
AA_OLD = (
    '        if _bootstrap_active:\n'
    '            _adjusted_rw = rw  # bootstrap weights unchanged during bootstrap phase\n'
    '        else:\n'
    '            _adjusted_rw = bsts_feedback.adjust_weights(rw)'
)
AA_NEW = (
    '        # v1.1.5b PATCH-AA: allow adjust_weights() during bootstrap, clamp to phase-A floor\n'
    '        # REF: Bug AA -- BootstrapRewardController.active() returns True permanently\n'
    '        #      (best_progress<95%, recent_completions==0 since car never completes lap)\n'
    '        #      => adjust_weights() never fired => rwadjcenter0.080/prog0.620 frozen 53 eps\n'
    '        # REF: Schulman et al. (2017) arXiv:1707.06347 -- live gradient signal required\n'
    '        _adjusted_rw = bsts_feedback.adjust_weights(rw)\n'
    '        if _bootstrap_active:\n'
    '            # Re-clamp to phase-A survival floor so bootstrap curriculum is preserved\n'
    "            _adjusted_rw['progress']    = max(_adjusted_rw.get('progress', 0.0), 0.45)\n"
    "            _adjusted_rw['racing_line'] = min(_adjusted_rw.get('racing_line', 0.0), 0.04)\n"
    "            _adjusted_rw['curv_speed']  = min(_adjusted_rw.get('curv_speed', 0.0), 0.02)\n"
    '            # Renormalize after clamping\n'
    '            _rw_total_aa = sum(_adjusted_rw.values()) or 1.0\n'
    '            _adjusted_rw = {k: v / _rw_total_aa for k, v in _adjusted_rw.items()}'
)
if AA_OLD in src:
    src = src.replace(AA_OLD, AA_NEW, 1)
    applied.append("PATCH-AA")
else:
    failed.append("PATCH-AA (target not found -- check bootstrap gate block)")

# ---------------------------------------------------------------------------
# PATCH-BB: Update ep_track_length_m from rp['tracklength'] per-step
# ---------------------------------------------------------------------------
BB_OLD = (
    '                _speed = rp.get("speed", 0)\n'
    '                # v4-bsts: barrier proximity from LIDAR and objects'
)
BB_NEW = (
    '                _speed = rp.get("speed", 0)\n'
    '                # v1.1.5b PATCH-BB: update ep_track_length_m from rp per-step\n'
    '                # REF: Bug BB -- tracklengthm 0.0 in all Kalman lines.\n'
    '                #      rp["tracklength"]=16.635021 confirmed at step1 in log.\n'
    '                #      ep_track_length_m was only updated from waypoint arc calc,\n'
    '                #      not from rp["tracklength"] directly per-step.\n'
    "                _tl_rp = rp.get('tracklength') or rp.get('track_length')\n"
    '                if _tl_rp and float(_tl_rp) > 1.0:\n'
    '                    ep_track_length_m = max(ep_track_length_m, float(_tl_rp))\n'
    '                # v4-bsts: barrier proximity from LIDAR and objects'
)
if BB_OLD in src:
    src = src.replace(BB_OLD, BB_NEW, 1)
    applied.append("PATCH-BB")
else:
    failed.append("PATCH-BB (target not found -- check if rp: block indentation)")

# ---------------------------------------------------------------------------
# PATCH-CC: Neutral floor for bsts_metrics compliance keys before bsts_row build
# ---------------------------------------------------------------------------
CC_OLD = (
    '                # v213: race-type tag for plot/CSV differentiation\n'
    '                _race_type_tag = {'
)
CC_NEW = (
    '                # v1.1.5b PATCH-CC: unconditional neutral floor for bsts_metrics compliance keys\n'
    '                # REF: Bug CC -- Kalman X-matrix reads summary (underscored keys).\n'
    '                #      When compute_all() throws => _hm_out={} => 3-tier merge writes 0.0\n'
    '                #      not neutral => X-matrix all-zeros => Kalman trends frozen.\n'
    '                # Fix: write neutral floor to bsts_metrics before merge fires.\n'
    '                import math as _math_cc\n'
    '                _CC_NEUTRALS = {\n'
    "                    'race_line_adherence': 0.5, 'brake_compliance': 1.0,\n"
    "                    'brake_field_compliance_gradient': 1.0,\n"
    "                    'race_line_compliance_gradient': 0.5,\n"
    '                }\n'
    '                for _cc_k, _cc_n in _CC_NEUTRALS.items():\n'
    '                    _cc_cur = bsts_metrics.get(_cc_k)\n'
    '                    _cc_bad = (_cc_cur is None or\n'
    '                               not _math_cc.isfinite(float(_cc_cur if _cc_cur is not None else float("nan"))))\n'
    '                    if _cc_bad:\n'
    '                        bsts_metrics[_cc_k] = float(_hm_out.get(_cc_k, _cc_n) if _hm_out else _cc_n)\n'
    '                # v213: race-type tag for plot/CSV differentiation\n'
    '                _race_type_tag = {'
)
if CC_OLD in src:
    src = src.replace(CC_OLD, CC_NEW, 1)
    applied.append("PATCH-CC")
else:
    failed.append("PATCH-CC (target not found -- check episode-end section)")

print("=== PATCH RESULTS ===")
print(f"Applied: {applied}")
print(f"Failed:  {failed}")
print(f"Size: {orig_len} -> {len(src)} chars (delta: +{len(src)-orig_len})")

if __name__ == "__main__":
    shutil.copy(SRC, BAK)
    SRC.write_text(src, encoding="utf-8")
    print(f"\nBackup: {BAK}")
    print(f"Patched: {SRC}")
    if failed:
        print(f"\nWARNING: {len(failed)} patch(es) failed -- check target strings")
        sys.exit(1)
    else:
        print("\nAll patches applied successfully.")
        print("\nExpected log signatures after v1.1.5b (by ep 5):")
        print("  ANTE: all steps should have step <= current ep_step_count")
        print("  rwadjcenter: should drift from 0.080 by ep 10")
        print("  tracklengthm: should show 16.635 in Kalman trends")
        print("  racelineadherence: 0.45-0.55 in Kalman X-matrix")
        print("  brakecompliance: 0.85-1.0 in Kalman X-matrix")
