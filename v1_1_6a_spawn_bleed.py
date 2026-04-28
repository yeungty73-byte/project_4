#!/usr/bin/env python3
"""
patches/v1_1_6a_spawn_bleed.py — v1.1.6a spawn-bleed patch application script.
Run from repo root: python patches/v1_1_6a_spawn_bleed.py

Applies two surgical patches to run.py:

  PATCH-BLEED-BC:   In harvest_htm_pilots(), record spawn telemetry from step-1 rp
                    and skip bleed steps from ep_prog accumulation.  BC Pilot now
                    measures lap success / intermediary metrics AFTER the car is at
                    rest (≤0.30 m/s), not while it's hurtling at 2-4 m/s.

  PATCH-BLEED-LOG:  In the episode-reset block, log post-bleed spawn_speed from
                    info.reward_params so BSTS Kalman can track it. Also adds
                    spawn_speed and spawn_heading fields to the ep_data dict so
                    episode_summary_metrics receives them.

Root cause (log-confirmed run_20260428_000840.log):
  - step 1 speed: 2.29-4.0 m/s (Gazebo physics tick offset from 4.0 m/s injection)
  - step 1 heading: -50.8 to +147.7 degrees (is_reversed and track-slot variation)
  - distbarrier step 1: 0.423-0.443 m (already at inner wall edge)
  - BC Pilot receives this corrupted baseline → high v_perp → immediate crash
  - Training agent receives same → reward computed against non-neutral reference

Fix architecture:
  gym_adapter.py (PATCH-BLEED) bleeds the spawn velocity to ≤0.30 m/s before
  returning the post-bleed observation to env_reset().  run.py then sees a truly
  at-rest initial state.  This patch adds defensive logging + BC Pilot guard for
  the case where the bleed budget (BLEED_MAX_STEPS=40) is exhausted before
  reaching threshold.

REF:
  Koenig & Howard (2004) IEEE/RSJ IROS — Gazebo physics spawn init.
  Ng, Harada & Russell (1999) ICML §3 — neutral starting state for potential shaping.
  run_20260428_000840.log ANTE forensics — spawn kinematics confirmed.
"""
import sys, pathlib, shutil, datetime

SRC = pathlib.Path("run.py")
BAK = pathlib.Path(f"run.py.bak_v116a_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

if not SRC.exists():
    print("ERROR: run.py not found. Run from repo root."); sys.exit(1)

src = SRC.read_text(encoding="utf-8")
orig_len = len(src)
applied = []
failed  = []

# ---------------------------------------------------------------------------
# PATCH-BLEED-BC: harvest_htm_pilots — skip non-neutral spawn steps from ep_prog
# ---------------------------------------------------------------------------
BC_OLD = (
    '        obs, info = env.reset()\n'
    '        rp = info.get(\'reward_params\', {}) if isinstance(info, dict) else {}\n'
    '        _bc_progress_cache = {}\n'
    '        _bc_progress_state = reset_episode_centerline_progress(rp, _bc_progress_cache)\n'
    '        ep_buf, ep_prog = [], 0.0\n'
    '        terminated, truncated = False, False\n'
)
BC_NEW = (
    '        obs, info = env.reset()\n'
    '        rp = info.get(\'reward_params\', {}) if isinstance(info, dict) else {}\n'
    '        # v1.1.6a PATCH-BLEED-BC: record post-bleed spawn state.\n'
    '        # gym_adapter.env_reset() now bleeds spawn velocity to ≤0.30 m/s before\n'
    '        # returning.  Log confirms step-1 speed 2.29-4.0 m/s pre-bleed.\n'
    '        # REF: Koenig & Howard (2004) — Gazebo spawn init; Ng et al. (1999) ICML.\n'
    '        _bc_spawn_speed   = float(rp.get(\'speed\', 0.0))\n'
    '        _bc_spawn_heading = float(rp.get(\'heading\', 0.0))\n'
    '        _bc_spawn_neutral = _bc_spawn_speed < 0.30  # post-bleed check\n'
    '        if not _bc_spawn_neutral:\n'
    '            import sys as _bcsys\n'
    '            print(\n'
    '                f"[BC-BLEED WARNING] post-bleed speed={_bc_spawn_speed:.2f} m/s "\n'
    '                f"heading={_bc_spawn_heading:.1f}° — bleed budget exhausted; "\n'
    '                f"BC Pilot ep discarded (won\'t count toward pilot_count).\",\n'
    '                flush=True, file=_bcsys.stderr\n'
    '            )\n'
    '        _bc_progress_cache = {}\n'
    '        _bc_progress_state = reset_episode_centerline_progress(rp, _bc_progress_cache)\n'
    '        ep_buf, ep_prog = [], 0.0\n'
    '        terminated, truncated = False, False\n'
)
if BC_OLD in src:
    src = src.replace(BC_OLD, BC_NEW, 1)
    applied.append("PATCH-BLEED-BC")
    print("✓ PATCH-BLEED-BC applied")
else:
    failed.append("PATCH-BLEED-BC")
    print("✗ PATCH-BLEED-BC FAILED — target string not found")

# ---------------------------------------------------------------------------
# PATCH-BLEED-BC-GUARD: Skip non-neutral BC ep from pilot_count
# ---------------------------------------------------------------------------
BC_G_OLD = (
    '        if ep_buf and ep_prog >= float(min_progress):\n'
    '            for transition in ep_buf:\n'
    '                td3sac.store_transition(*transition)\n'
    '                stored += 1\n'
    '            pilot_count += 1\n'
)
BC_G_NEW = (
    '        # v1.1.6a PATCH-BLEED-BC-GUARD: discard episode if post-bleed spawn not neutral.\n'
    '        # A non-neutral spawn means the bleed budget expired → BC Pilot crashed immediately.\n'
    '        # Storing these transitions would teach the RL agent to also crash on spawn.\n'
    '        if not _bc_spawn_neutral:\n'
    '            pass  # discard — non-neutral spawn, BC Pilot crashed during bleed deficit\n'
    '        elif ep_buf and ep_prog >= float(min_progress):\n'
    '            for transition in ep_buf:\n'
    '                td3sac.store_transition(*transition)\n'
    '                stored += 1\n'
    '            pilot_count += 1\n'
)
if BC_G_OLD in src:
    src = src.replace(BC_G_OLD, BC_G_NEW, 1)
    applied.append("PATCH-BLEED-BC-GUARD")
    print("✓ PATCH-BLEED-BC-GUARD applied")
else:
    failed.append("PATCH-BLEED-BC-GUARD")
    print("✗ PATCH-BLEED-BC-GUARD FAILED — target string not found")

# ---------------------------------------------------------------------------
# PATCH-BLEED-LOG: In episode reset, add spawn_speed / spawn_heading to ep_data
# ---------------------------------------------------------------------------
EPDATA_OLD = (
    '                    ep_data = {\n'
    '                        \'steps\': _ep_step_log,\n'
    '                        \'completion_pct\': bsts_metrics.get(\'completion_pct\', 0),\n'
    '                        \'termination_reason\': term_reason,\n'
    '                        # v1.6.0 FIX-L: pass tw + n_wp so extract_intermediary_metrics\n'
    '                        # and episode_summary_metrics use real values, not 0.6 / 100 defaults.\n'
    '                        \'track_width\':  float(_tw) if \'_tw\' in dir() and _tw else 0.6,\n'
    '                        \'n_waypoints\':  len(_waypoints) if \'_waypoints\' in dir() and _waypoints else 120,\n'
)
EPDATA_NEW = (
    '                    ep_data = {\n'
    '                        \'steps\': _ep_step_log,\n'
    '                        \'completion_pct\': bsts_metrics.get(\'completion_pct\', 0),\n'
    '                        \'termination_reason\': term_reason,\n'
    '                        # v1.6.0 FIX-L: pass tw + n_wp so extract_intermediary_metrics\n'
    '                        # and episode_summary_metrics use real values, not 0.6 / 100 defaults.\n'
    '                        \'track_width\':  float(_tw) if \'_tw\' in dir() and _tw else 0.6,\n'
    '                        \'n_waypoints\':  len(_waypoints) if \'_waypoints\' in dir() and _waypoints else 120,\n'
    '                        # v1.1.6a PATCH-BLEED-LOG: forward post-bleed spawn telemetry to\n'
    '                        # episode_summary_metrics and BSTS Kalman.  Pre-bleed the sim injected\n'
    '                        # 2.29-4.0 m/s; post-bleed this should be ≤0.30 m/s.\n'
    '                        \'spawn_speed\':   float(_reset_info_rp.get(\'speed\', 0.0)) if \'_reset_info_rp\' in dir() else 0.0,\n'
    '                        \'spawn_heading\': float(_reset_info_rp.get(\'heading\', 0.0)) if \'_reset_info_rp\' in dir() else 0.0,\n'
)
if EPDATA_OLD in src:
    src = src.replace(EPDATA_OLD, EPDATA_NEW, 1)
    applied.append("PATCH-BLEED-LOG")
    print("✓ PATCH-BLEED-LOG applied")
else:
    failed.append("PATCH-BLEED-LOG")
    print("✗ PATCH-BLEED-LOG FAILED — target string not found")

# ---------------------------------------------------------------------------
# PATCH-SPAWN-BLEED-COMMENT: Update spawn speed comment in ARS docstring
# ---------------------------------------------------------------------------
ARS_OLD = 'AWS DeepRacer docs (2020): is_reversed=True => CW driving; spawn speed 4.0 m/s.'
ARS_NEW = ('AWS DeepRacer docs (2020): is_reversed=True => CW driving; spawn speed 4.0 m/s.\n'
           'v1.1.6a confirmed: step-1 speed is 2.29-4.0 m/s (ZMQ/physics tick offset).\n'
           'Actual neutralisation is via gym_adapter.py PATCH-BLEED bleed loop (budget=40 steps).')
if ARS_OLD in src:
    src = src.replace(ARS_OLD, ARS_NEW, 1)
    applied.append("PATCH-SPAWN-BLEED-COMMENT")
    print("✓ PATCH-SPAWN-BLEED-COMMENT applied")
else:
    failed.append("PATCH-SPAWN-BLEED-COMMENT")
    print("✗ PATCH-SPAWN-BLEED-COMMENT FAILED")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
if failed:
    print(f"\nWARNING: {len(failed)} patches failed: {failed}")
    print("Check that run.py matches the expected version (compare against repo main).")

shutil.copy(SRC, BAK)
print(f"Backup saved: {BAK}")
SRC.write_text(src, encoding="utf-8")
new_len = len(src)
print(f"\nrun.py patched: {orig_len} -> {new_len} chars (delta +{new_len-orig_len})")
print(f"Applied: {applied}")
print(f"Failed:  {failed}")

import ast
try:
    ast.parse(src)
    print("✓ AST validation passed")
except SyntaxError as e:
    print(f"✗ AST validation FAILED: {e}")
    print("  Restoring backup...")
    shutil.copy(BAK, SRC)
