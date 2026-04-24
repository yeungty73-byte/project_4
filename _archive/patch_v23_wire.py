#!/usr/bin/env python3
"""patch_v23_wire.py - Wire dangling analyze_logs imports into run.py training loop.

Connects:
  1. extract_intermediary_metrics -> replaces placeholder intermediary
  2. bsts_compliance_report -> periodic full BSTS decomposition
  3. compute_anneal_recommendations -> auto-tuning from BSTS
  4. compute_optimal_race_line + score_race_line_compliance -> race line analysis
  5. Adds _ep_step_log collection during episode for proper intermediary extraction
"""
import re, os

def apply():
    base = os.path.dirname(os.path.abspath(__file__))
    run_path = os.path.join(base, 'run.py')
    with open(run_path, 'r') as f:
        src = f.read()
    lines = src.split('\n')
    changes = []

    # --- 1. Add _ep_step_log initialization alongside other ep_ lists ---
    # Find 'ep_heading_diffs = []' and add _ep_step_log = [] after it
    for i, line in enumerate(lines):
        if 'ep_heading_diffs = []' in line and '_ep_step_log' not in lines[i-1] and '_ep_step_log' not in lines[i+1]:
            indent = line[:len(line) - len(line.lstrip())]
            lines.insert(i + 1, f'{indent}_ep_step_log = []  # collect per-step dicts for extract_intermediary_metrics')
            changes.append(f'Added _ep_step_log init after line {i+1}')
            break

    # Also add _ep_step_log = [] in the episode-reset section (after ep_racing_line_errors = [])
    for i, line in enumerate(lines):
        if 'ep_racing_line_errors = []' in line and i > 500:  # only the reset in the loop, not init
            # Check if _ep_step_log reset already exists nearby
            nearby = '\n'.join(lines[max(0,i-3):i+5])
            if '_ep_step_log' not in nearby:
                indent = line[:len(line) - len(line.lstrip())]
                lines.insert(i + 1, f'{indent}_ep_step_log = []')
                changes.append(f'Added _ep_step_log reset after line {i+1}')
            break

    # --- 2. Add step data collection in the step loop ---
    # Find where ep_speeds.append or ep_steerings_raw.append happens and add _ep_step_log.append
    step_log_added = False
    for i, line in enumerate(lines):
        if 'ep_steerings_raw.append(' in line and not step_log_added:
            indent = line[:len(line) - len(line.lstrip())]
            # Add after the steerings append
            insert_code = f"""{indent}# Collect step dict for extract_intermediary_metrics
{indent}try:
{indent}    _step_info = info if isinstance(info, dict) else {{}}
{indent}    _ep_step_log.append({{
{indent}        'x': _step_info.get('x', _step_info.get('X', 0)),
{indent}        'y': _step_info.get('y', _step_info.get('Y', 0)),
{indent}        'heading': _step_info.get('heading', _step_info.get('yaw', 0)),
{indent}        'speed': _step_info.get('speed', ep_prev_speed),
{indent}        'steering_angle': float(action[0]) if hasattr(action, '__getitem__') else 0,
{indent}        'throttle': float(action[1]) if hasattr(action, '__getitem__') and len(action) > 1 else 0,
{indent}        'reward': float(reward),
{indent}        'all_wheels_on_track': _step_info.get('all_wheels_on_track', True),
{indent}    }})
{indent}except Exception:
{indent}    pass"""
            lines.insert(i + 1, insert_code)
            step_log_added = True
            changes.append(f'Added _ep_step_log.append after line {i+1}')
            break

    # --- 3. Replace placeholder intermediary with extract_intermediary_metrics ---
    for i, line in enumerate(lines):
        if 'intermediary = {m: [0] for m in INTERMEDIARY_METRICS}' in line:
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}# Wire: extract real intermediary metrics from episode step log"
            lines.insert(i + 1, f"{indent}try:")
            lines.insert(i + 2, f"{indent}    intermediary = extract_intermediary_metrics({{'steps': _ep_step_log}})")
            lines.insert(i + 3, f"{indent}except Exception:")
            lines.insert(i + 4, f"{indent}    intermediary = {{m: [0] for m in INTERMEDIARY_METRICS}}  # fallback")
            changes.append(f'Replaced placeholder intermediary at line {i+1}')
            break

    # --- 4. Replace ep_data empty steps with _ep_step_log ---
    for i, line in enumerate(lines):
        if "ep_data = {'steps': []" in line:
            lines[i] = line.replace("'steps': []", "'steps': _ep_step_log")
            changes.append(f'Replaced ep_data steps=[] with _ep_step_log at line {i+1}')
            break

    # --- 5. Add BSTS compliance report + anneal recommendations after buffer clear ---
    for i, line in enumerate(lines):
        if '_kf_episode_buffer.clear()' in line and i > 500:
            indent = line[:len(line) - len(line.lstrip())]
            bsts_block = f"""
{indent}# === BSTS Compliance Report (every 50 episodes) ===
{indent}if episode_count > 0 and episode_count % 50 == 0:
{indent}    try:
{indent}        _bsts_matrix = list(_kf_episode_buffer) if _kf_episode_buffer else []
{indent}        # Use accumulated summaries from all prior episodes
{indent}        if hasattr(bsts_feedback, '_all_summaries'):
{indent}            _bsts_matrix = bsts_feedback._all_summaries
{indent}        if len(_bsts_matrix) >= 10:
{indent}            _bsts_rpt = bsts_compliance_report(_bsts_matrix)
{indent}            _anneal_recs = compute_anneal_recommendations(_bsts_rpt, _bsts_matrix)
{indent}            logger.info(f"[BSTS-Report] trend={{_bsts_rpt.get('trend','?')}} "
{indent}                        f"seasonal={{_bsts_rpt.get('seasonal_period','?')}} "
{indent}                        f"LR_rec={{_anneal_recs.get('learning_rate','?')}} "
{indent}                        f"residual_std={{_anneal_recs.get('residual_std',0):.4f}}")
{indent}            for sm, trend in _bsts_rpt.get('per_metric_trends', {{}}).items():
{indent}                logger.info(f"  {{sm}}: {{trend}}")
{indent}            for rec in _bsts_rpt.get('recommendations', []):
{indent}                logger.info(f"  [REC] {{rec.get('action','')}}")
{indent}            # Apply anneal recommendations to reward weights
{indent}            for k, v in _anneal_recs.get('reward_weight_adjustments', {{}}).items():
{indent}                logger.info(f"  [ANNEAL] {{k}}: {{v}}")
{indent}    except Exception as _be:
{indent}        logger.debug(f"BSTS report skip: {{_be}}")

{indent}# === Race Line Analysis (every 100 episodes) ===
{indent}if episode_count > 0 and episode_count % 100 == 0:
{indent}    try:
{indent}        _wps = []
{indent}        if hasattr(bsts_feedback, '_all_summaries'):
{indent}            for _ep in bsts_feedback._all_summaries[-20:]:
{indent}                if _ep.get('lap_completion_pct', 0) > 80:
{indent}                    _wps = [(s.get('x',0), s.get('y',0)) for s in _ep.get('_steps', [])]
{indent}                    break
{indent}        if not _wps and hasattr(env, 'waypoints'):
{indent}            _wps = [(w[0], w[1]) for w in env.waypoints]
{indent}        if _wps and len(_wps) >= 4:
{indent}            _rl = compute_optimal_race_line(_wps)
{indent}            logger.info(f"[RaceLine] brake_integral={{_rl.get('brake_zone_integral',0):.3f}} "
{indent}                        f"brake_pts={{len(_rl.get('brake_points',[]))}}")
{indent}    except Exception as _re:
{indent}        logger.debug(f"Race line analysis skip: {{_re}}")"""
            lines.insert(i + 1, bsts_block)
            changes.append(f'Added BSTS compliance + race line analysis after line {i+1}')
            break

    # --- 6. Add _all_summaries accumulator to bsts_feedback ---
    # Find where bsts_feedback is created and add _all_summaries attribute
    for i, line in enumerate(lines):
        if 'bsts_feedback = BSTSFeedback(' in line:
            indent = line[:len(line) - len(line.lstrip())]
            lines.insert(i + 1, f'{indent}bsts_feedback._all_summaries = []  # accumulate all episode summaries for periodic BSTS report')
            changes.append(f'Added _all_summaries init after line {i+1}')
            break

    # Add accumulator at the _kf_episode_buffer.append line
    for i, line in enumerate(lines):
        if '_kf_episode_buffer.append(summary)' in line:
            indent = line[:len(line) - len(line.lstrip())]
            lines.insert(i + 1, f'{indent}if hasattr(bsts_feedback, "_all_summaries"): bsts_feedback._all_summaries.append(summary)')
            changes.append(f'Added _all_summaries accumulator after line {i+1}')
            break

    src_new = '\n'.join(lines)
    with open(run_path, 'w') as f:
        f.write(src_new)

    import py_compile
    try:
        py_compile.compile(run_path, doraise=True)
        print(f'  SYNTAX OK: run.py ({len(changes)} changes)')
    except py_compile.PyCompileError as e:
        print(f'  SYNTAX ERROR: {e}')

    for c in changes:
        print(f'  {c}')

if __name__ == '__main__':
    apply()
