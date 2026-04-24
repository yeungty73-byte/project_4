# import json, sys, os  # removed: unused

def analyze(fpath):
# REF: Scott, S. L. & Varian, H. R. (2014). Predicting the present with Bayesian structural time series. Int. J. Math. Model. Numer. Optim.
    rows = []
    with open(fpath) as f:
        for line in f:
            try: rows.append(json.loads(line.strip()))
            except: pass
    if not rows:
        print('No data'); return
    n = len(rows)
    # Window analysis
    windows = [('first50', rows[:50]), ('last50', rows[-50:]), ('last10', rows[-10:])]
    keys = ['progress','ep_return','avg_speed','crash','offtrack_rate',
            'avg_safe_speed_ratio','avg_racing_line_err','avg_decel_penalty',
            'min_dist_from_center','curvature_x_speed','reversed_count','zero_speed_count']
    print(f'=== Live Metrics Analysis ({n} episodes) ===')
    print(f'Global steps: {rows[0]["global_step"]} -> {rows[-1]["global_step"]}')
    print()
    for wname, wdata in windows:
        if not wdata: continue
        print(f'--- {wname} (n={len(wdata)}) ---')
        for k in keys:
            vals = [r.get(k, 0) for r in wdata]
            vals = [v for v in vals if v != float('-inf') and v != float('inf')]
            if vals:
                avg = sum(vals)/len(vals)
                mn, mx = min(vals), max(vals)
                print(f'  {k:30s} avg={avg:8.4f}  min={mn:8.4f}  max={mx:8.4f}')
        print()
    _fb = {"crash_rate": crash_rate, "offtrack_rate": offtrack, "avg_progress": avg_prog, "avg_speed": sum(speed_vals)/len(speed_vals) if speed_vals else 0.0, "n_episodes": n, "trends": {}}
    # Reward hacking detection
    print('=== REWARD HACKING CHECKS ===')
    last100 = rows[-100:] if len(rows)>=100 else rows
    prog_vals = [r.get('progress',0) for r in last100]
    avg_prog = sum(prog_vals)/len(prog_vals) if prog_vals else 0
    if avg_prog < 1.0:
        print(f'  [WARN] Avg progress={avg_prog:.2f}% - car barely moving!')
    speed_vals = [r.get('avg_speed',0) for r in last100]
    speed_std = (sum((v-sum(speed_vals)/len(speed_vals))**2 for v in speed_vals)/len(speed_vals))**0.5
    if speed_std < 0.01:
        print(f'  [WARN] Speed std={speed_std:.4f} - constant speed (possible reward hack)')
    zero_ct = sum(1 for r in last100 if r.get('zero_speed_count',0) > 0)
    if zero_ct > len(last100)*0.5:
        print(f'  [WARN] {zero_ct}/{len(last100)} episodes have zero-speed steps')
    rev_ct = sum(r.get('reversed_count',0) for r in last100)
    if rev_ct > len(last100)*0.3:
        print(f'  [WARN] High reversal count: {rev_ct} across {len(last100)} episodes')
    ret_vals = [r.get('ep_return',0) for r in last100 if r.get('ep_return',0) != float('-inf')]
    if not ret_vals:
        print(f'  [WARN] All ep_return=-Infinity - episodes terminating immediately!')
    elif sum(r.get('ep_return',0)==float('-inf') for r in last100) > len(last100)*0.8:
        print(f'  [WARN] >80% episodes have -Inf return')
    crash_rate = sum(r.get('crash',0) for r in last100)/len(last100)
    offtrack = sum(r.get('offtrack_rate',0) for r in last100)/len(last100)
    print(f'  Crash rate: {crash_rate:.2%}, Offtrack rate: {offtrack:.4f}')
    print(f'  Avg progress: {avg_prog:.2f}%, Avg speed: {sum(speed_vals)/len(speed_vals):.3f}')
    print()
    # Trend: compare first half vs second half
    if n >= 20:
        h = n//2
        first, second = rows[:h], rows[h:]
        print('=== TREND (first half vs second half) ===')
        for k in ['progress','avg_speed','avg_safe_speed_ratio','avg_racing_line_err','crash']:
            v1 = [r.get(k,0) for r in first if r.get(k,0)!=float('-inf')]
            v2 = [r.get(k,0) for r in second if r.get(k,0)!=float('-inf')]
            if v1 and v2:
                a1, a2 = sum(v1)/len(v1), sum(v2)/len(v2)
                delta = a2 - a1
                arrow = '↑' if delta>0.001 else ('↓' if delta<-0.001 else '→')
                print(f'  {k:30s} {a1:8.4f} -> {a2:8.4f} ({arrow} {delta:+.4f})')
                _fb["trends"][k] = delta

    return _fb

if __name__=='__main__':
    import glob
    files = sorted(glob.glob('results/v5_metrics_*.jsonl'))
    if files:
        print(f'Analyzing: {files[-1]}')
        analyze(files[-1])
    else:
        print('No metrics files found')
