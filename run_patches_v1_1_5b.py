#!/usr/bin/env python3
"""run_patches_v1_1_5b.py
One-shot patcher for run.py: applies all 5 v1.1.5b surgical patches.
Run once from the project root: python run_patches_v1_1_5b.py
Then delete this file.

Patches applied:
  P1: process_action TPA throttle headroom cap
  P2: spawn_penalty multiplicative (not additive -5.0)
  P3: bsts_row tracklengthm fix (ep_track_length_m)
  P4: bsts_row racelineadherence/brakecompliance key aliases
  P5: update_tpa() + TensorBoard curriculum/ scalars at episode end

REF: Bengio et al. (2009) ICML -- curriculum learning easy-first ordering.
REF: Amodei et al. (2016) arXiv:1606.06565 -- avoid net-negative reward.
REF: Almakhayita et al. (2025) PLoS ONE -- adaptive reward design.
"""
import os, sys, shutil, ast

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")

PATCHES = [
    # ----------------------------------------------------------------
    # P1: process_action -- TPA throttle headroom cap
    # ----------------------------------------------------------------
    (
        "P1_process_action_TPA",
        """        if actdim >= 2 and a.size >= 2:
            # remap throttle channel from tanh [-1,1] \u2192 env [0,1]
            a[1] = (a[1] + 1.0) / 2.0
            # v1.1.2: hard floor \u2014 car CANNOT drive backward
            a[1] = max(0.0, float(a[1]))
        return np.clip(a, actionspace.low, actionspace.high).astype(np.float32)""",
        """        if actdim >= 2 and a.size >= 2:
            # remap throttle channel from tanh [-1,1] -> env [0,1]
            a[1] = (a[1] + 1.0) / 2.0
            # v1.1.2: hard floor -- car CANNOT drive backward
            a[1] = max(0.0, float(a[1]))
            # v1.5.0b: TPA throttle headroom cap -- rises from 0.55->1.0 as car masters track
            # REF: Bengio et al. (2009) ICML -- start with low action complexity.
            try:
                if '_shaper' in globals() and _shaper is not None:
                    _tpa_head = _shaper.process_action_scale()
                    a[1] = min(a[1], float(_tpa_head))
            except Exception:
                pass
        return np.clip(a, actionspace.low, actionspace.high).astype(np.float32)""",
    ),
    # ----------------------------------------------------------------
    # P2: spawn_penalty multiplicative gate
    # ----------------------------------------------------------------
    (
        "P2_spawn_penalty_multiplicative",
        """            # v1.4.0 BUG-FIX: spawn_penalty must enter PPO reward buffer (not just cumulative_ep_reward)
            # Without this, reversed-episode signal never reaches PPO gradient.
            # _spawn_penalty is -5.0 for is_reversed=True; 0.0 otherwise.
            # Applied only at ep_step_count==1 to avoid double-counting.
            # REF: Schulman et al. (2017) -- PPO gradient requires in-buffer reward signal.
            _step_reward_final = float(reward)
            if ep_step_count == 1 and '_spawn_penalty' in dir() and _spawn_penalty != 0.0:
                _step_reward_final += _spawn_penalty
                _spawn_penalty = 0.0  # consumed; do NOT also add in episode-end block
            rewards[step] = tensor(np.array(_step_reward_final))""",
        """            # v1.5.0b FIX-spawn: spawn_penalty is now MULTIPLICATIVE (not additive -5.0).
            # Additive -5.0 was causing net negative rewards in short reversed episodes,
            # triggering the freeze-trap (agent learns reward=0 by staying still).
            # Multiplicative zero-out: reversed step 1 reward = 0. Non-reversed: unchanged.
            # REF: Amodei et al. (2016) -- avoid net-negative reward signals.
            # REF: Schulman et al. (2017) -- PPO gradient requires in-buffer reward signal.
            _step_reward_final = float(reward)
            if ep_step_count == 1 and '_spawn_penalty' in dir() and _spawn_penalty != 0.0:
                # Multiplicative: reversed episode step 1 earns 0 (not -5.0)
                _step_reward_final = _step_reward_final * 0.0  # zero, not negative
                _spawn_penalty = 0.0  # consumed
                logger.debug(f"[ARS] ep={episode_count} reversed step1 reward zeroed (multiplicative spawn gate)")
            rewards[step] = tensor(np.array(_step_reward_final))""",
    ),
    # ----------------------------------------------------------------
    # P3: bsts_row tracklengthm fix
    # ----------------------------------------------------------------
    (
        "P3_tracklengthm_bsts_row",
        """                    'rl_blend':      round(_rl_blend,4),
                    'env_signal':    round(_env_signal if '_env_signal' in dir() else 0.0,4),""",
        """                    # v1.5.0b FIX-O2: tracklengthm was 0.0 because bsts_row never set it.
                    # ep_track_length_m is updated per-step via update_episode_centerline_progress.
                    # REF: BSTS-Kalman regressor matrix uses tracklengthm as a feature.
                    'tracklengthm':  round(float(ep_track_length_m) if ep_track_length_m > 0 else 16.6, 4),
                    'rl_blend':      round(_rl_blend,4),
                    'env_signal':    round(_env_signal if '_env_signal' in dir() else 0.0,4),""",
    ),
    # ----------------------------------------------------------------
    # P4: bsts_row key aliases for BSTS-Kalman regnames
    # ----------------------------------------------------------------
    (
        "P4_bsts_row_key_aliases",
        "                bsts_row.update(_hm_out)",
        """                bsts_row.update(_hm_out)
                # v1.5.0b FIX-K2: BSTS-Kalman regnames use underscore-free keys
                # ('racelineadherence', 'brakecompliance') but _hm_out stores
                # 'race_line_adherence', 'brake_compliance'.
                # Explicitly alias so the regressor X-matrix is non-zero.
                # REF: BSTS-Kalman BSTSKalmanFilter regnames suffix bug (v1.1.4 audit).
                bsts_row.setdefault('racelineadherence',
                    float(_hm_out.get('race_line_adherence', 0.5)))
                bsts_row.setdefault('brakecompliance',
                    float(_hm_out.get('brake_compliance', 1.0)))
                bsts_row.setdefault('avgspeedcenterline',
                    float(bsts_row.get('avg_speed',
                          sum(ep_speeds)/max(len(ep_speeds),1) if ep_speeds else 0.0)))""",
    ),
    # ----------------------------------------------------------------
    # P5: update_tpa() + TensorBoard curriculum/ scalars
    # ----------------------------------------------------------------
    (
        "P5_update_tpa_tensorboard",
        """                    # v1.4.2: TelemetryFeedbackAnnealer -- advance phase on actual mastery.
                    # Phase 0->1: any lap completion. Phase 1->2: completion>=80% + adherence>=0.45.
                    # REF: Almakhayita et al. (2025) PLoS ONE -- adaptive reward design.
                    if hasattr(_shaper, 'update_phase') and 'bsts_row' in dir() and bsts_row:
                        _phase_prev = _shaper.current_phase
                        _phase_new  = _shaper.update_phase(bsts_row)
                        if _phase_new != _phase_prev:
                            logger.info(f\"[ARS-TFA] Phase {_phase_prev}->{_phase_new} ep={episode_count} \"
                                       f\"completion_ema={getattr(_shaper,'_tfa_completion_ema',0):.2f}\")""",
        """                    # v1.4.2: TelemetryFeedbackAnnealer -- advance phase on actual mastery.
                    # Phase 0->1: any lap completion. Phase 1->2: completion>=80% + adherence>=0.45.
                    # REF: Almakhayita et al. (2025) PLoS ONE -- adaptive reward design.
                    if hasattr(_shaper, 'update_phase') and 'bsts_row' in dir() and bsts_row:
                        _phase_prev = _shaper.current_phase
                        _phase_new  = _shaper.update_phase(bsts_row)
                        if _phase_new != _phase_prev:
                            logger.info(f"[ARS-TFA] Phase {_phase_prev}->{_phase_new} ep={episode_count} "
                                       f"completion_ema={getattr(_shaper,'_tfa_completion_ema',0):.2f}")
                    # v1.5.0b: TrackProgressAnnealer update -- continuous soft curriculum.
                    # Ingests track_progress_pct + avg_speed + race_line_compliance_gradient
                    # from bsts_row to compute phase_blend and throttle headroom ceiling.
                    # REF: Bengio et al. (2009) ICML -- curriculum learning easy-first.
                    if hasattr(_shaper, 'update_tpa') and 'bsts_row' in dir() and bsts_row:
                        _tpa_blend = _shaper.update_tpa(bsts_row)
                        _tpa_diag  = _shaper.tpa_diagnostics() if hasattr(_shaper, 'tpa_diagnostics') else {}
                        writer.add_scalar("curriculum/tpa_phase_blend",       _tpa_blend,                          global_step)
                        writer.add_scalar("curriculum/tpa_throttle_headroom", _tpa_diag.get("tpa_throttle_headroom", 1.0), global_step)
                        writer.add_scalar("curriculum/tpa_ema_progress",      _tpa_diag.get("tpa_ema_progress", 0.0),      global_step)
                        writer.add_scalar("curriculum/tpa_ema_speed",         _tpa_diag.get("tpa_ema_speed", 0.0),         global_step)
                        writer.add_scalar("curriculum/tpa_ema_rl",            _tpa_diag.get("tpa_ema_rl", 0.5),            global_step)
                        if episode_count % 10 == 0:
                            logger.info(
                                f"[TPA] ep={episode_count} blend={_tpa_blend:.3f} "
                                f"headroom={_tpa_diag.get('tpa_throttle_headroom',1.0):.3f} "
                                f"prog_ema={_tpa_diag.get('tpa_ema_progress',0):.3f} "
                                f"spd_ema={_tpa_diag.get('tpa_ema_speed',0):.3f} "
                                f"rl_ema={_tpa_diag.get('tpa_ema_rl',0.5):.3f}"
                            )""",
    ),
]


def apply_patches(src):
    applied = []
    for name, old, new in PATCHES:
        if old in src:
            src = src.replace(old, new, 1)
            applied.append(name)
            print(f"  [OK] {name}")
        else:
            print(f"  [MISS] {name} -- target string not found", file=sys.stderr)
    return src, applied


def main():
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found", file=sys.stderr)
        sys.exit(1)
    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()
    backup = TARGET + ".bak_v1_1_5b"
    shutil.copy2(TARGET, backup)
    print(f"Backup: {backup}")
    patched, applied = apply_patches(src)
    # AST validation
    try:
        ast.parse(patched)
        print(f"AST OK after {len(applied)}/{len(PATCHES)} patches")
    except SyntaxError as e:
        print(f"SYNTAX ERROR after patching: {e}", file=sys.stderr)
        shutil.copy2(backup, TARGET)
        sys.exit(1)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"run.py updated. Applied: {applied}")


if __name__ == "__main__":
    main()
