# denim_theme imports removed (unused)
import json, csv, os
import numpy as np
from collections import defaultdict, Counter

class FailureAnalyzer:
    """Post-hoc forensic analysis of crash episodes.
    Reads forensic_episodes.jsonl and bsts_metrics.csv to identify
    kryptonite patterns, stuck waypoints, and failure combinations."""

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.forensic_path = os.path.join(log_dir, "forensic_episodes.jsonl")
        self.csv_path = os.path.join(log_dir, "bsts_metrics.csv")

    def load_forensic_episodes(self):
        eps = []
        if not os.path.exists(self.forensic_path):
            return eps
        with open(self.forensic_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    eps.append(json.loads(line))
        return eps

    def load_all_episodes(self):
        rows = []
        if not os.path.exists(self.csv_path):
            return rows
        with open(self.csv_path) as f:
            rows = list(csv.DictReader(f))
        return rows

    def stuck_waypoint_analysis(self, top_n=10):
        """Find waypoints where the car crashes most often."""
        eps = self.load_forensic_episodes()
        if not eps:
            return {}
        wp_counter = Counter()
        for ep in eps:
            wp_end = ep.get('closest_waypoint_end', None)
            if wp_end is not None:
                wp_counter[int(float(wp_end))] += 1
        most_common = wp_counter.most_common(top_n)
        return {"stuck_waypoints": most_common, "total_crashes": len(eps)}

    def kryptonite_summary(self):
        """Summarize kryptonite failure patterns."""
        eps = self.load_forensic_episodes()
        kryp_episodes = [e for e in eps if e.get('kryptonite_flag')]
        if not kryp_episodes:
            return {"kryptonite_count": 0, "patterns": {}}
        pattern_counter = Counter(e.get('kryptonite_desc', "unknown") for e in kryp_episodes)
        pattern_details = {}
        for pattern, count in pattern_counter.items():
            matching = [e for e in kryp_episodes if e.get('kryptonite_desc') == pattern]
            speeds = [e.get('ep_speed_mean', 0) for e in matching]
            steers = [abs(e.get('steering_angle_mean', 0)) for e in matching]
            progress_vals = [e.get('progress', 0) for e in matching]
            pattern_details[pattern] = {
                "count": count,
                "avg_speed": float(np.mean(speeds)) if speeds else 0,
                "avg_abs_steer": float(np.mean(steers)) if steers else 0,
                "avg_progress": float(np.mean(progress_vals)) if progress_vals else 0,
            }
        return {
            "kryptonite_count": len(kryp_episodes),
            "total_crashes": len(eps),
            "kryptonite_rate": len(kryp_episodes) / max(len(eps), 1),
            "patterns": pattern_details
        }

    def graze_vs_avoid_analysis(self):
        """Compare episodes that grazed walls vs clean avoidance."""
        all_eps = self.load_all_episodes()
        if not all_eps:
            return {}
        grazers = []
        clean = []
        for ep in all_eps:
            gc = int(float(ep.get('graze_count', 0)))
            crashed = int(float(ep.get('is_crashed', 0)))
            if gc > 0 and not crashed:
                grazers.append(ep)
            elif gc == 0 and not crashed:
                clean.append(ep)
        def avg_metric(eps_list, key):
            vals = [float(e[key]) for e in eps_list if e.get(key, "") != ""]
            return float(np.mean(vals)) if vals else 0
        return {
            "graze_episodes": len(grazers),
            "clean_episodes": len(clean),
            "graze_avg_reward": avg_metric(grazers, "ep_reward"),
            "clean_avg_reward": avg_metric(clean, "ep_reward"),
            "graze_avg_progress": avg_metric(grazers, "progress"),
            "clean_avg_progress": avg_metric(clean, "progress"),
            "graze_avg_speed": avg_metric(grazers, "ep_speed_mean"),
            "clean_avg_speed": avg_metric(clean, "ep_speed_mean"),
        }

    def curvature_speed_analysis(self):
        """Analyze curvature*speed correlation with crashes."""
        all_eps = self.load_all_episodes()
        if not all_eps:
            return {}
        crashed = [e for e in all_eps if int(float(e.get('is_crashed', 0)))]
        survived = [e for e in all_eps if not int(float(e.get('is_crashed', 0)))]
        def avg(lst, key):
            vals = [float(e[key]) for e in lst if e.get(key, "") != ""]
            return float(np.mean(vals)) if vals else 0
        return {
            "crashed_curvxspeed": avg(crashed, "curvature_x_speed_mean"),
            "survived_curvxspeed": avg(survived, "curvature_x_speed_mean"),
            "crashed_cornering": avg(crashed, "cornering_score"),
            "survived_cornering": avg(survived, "cornering_score"),
            "crashed_straight_speed": avg(crashed, "straight_score"),
            "survived_straight_speed": avg(survived, "straight_score"),
        }

    def cornering_technique_analysis(self):
        """Analyze cornering technique: how well does the agent handle turns?"""
        all_eps = self.load_all_episodes()
        if not all_eps:
            return {}
        scores = [float(e['cornering_score']) for e in all_eps if e.get('cornering_score', "") != ""]
        if not scores:
            return {}
        quartiles = np.percentile(scores, [25, 50, 75])
        good_corner = [e for e in all_eps if e.get('cornering_score', "") != "" and float(e['cornering_score']) >= quartiles[2]]
        poor_corner = [e for e in all_eps if e.get('cornering_score', "") != "" and float(e['cornering_score']) <= quartiles[0]]
        def crash_rate(lst):
            if not lst:
                return 0
            return np.mean([int(float(e.get('is_crashed', 0))) for e in lst])
        return {
            "cornering_score_p25": float(quartiles[0]),
            "cornering_score_p50": float(quartiles[1]),
            "cornering_score_p75": float(quartiles[2]),
            "good_corner_crash_rate": float(crash_rate(good_corner)),
            "poor_corner_crash_rate": float(crash_rate(poor_corner)),
            "good_corner_avg_progress": float(np.mean([float(e.get('progress', 0)) for e in good_corner])) if good_corner else 0,
            "poor_corner_avg_progress": float(np.mean([float(e.get('progress', 0)) for e in poor_corner])) if poor_corner else 0,
        }

    def full_report(self):
        """Generate comprehensive failure analysis report."""
        report = {
            "stuck_waypoints": self.stuck_waypoint_analysis(),
            "kryptonite": self.kryptonite_summary(),
            "graze_vs_avoid": self.graze_vs_avoid_analysis(),
            "curvature_speed": self.curvature_speed_analysis(),
            "cornering": self.cornering_technique_analysis(),
        }
        report_path = os.path.join(self.log_dir, "failure_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Failure report saved to {report_path}")
        return report


def print_report(report):
    """Pretty-print failure analysis report."""
    print("\n" + "="*60)
    print("FAILURE ANALYSIS REPORT")
    print("="*60)
    sw = report.get('stuck_waypoints', {})
    if sw:
        print(f"\nTotal crashes: {sw.get('total_crashes', 0)}")
        print("Top stuck waypoints:")
        for wp, count in sw.get('stuck_waypoints', []):
            print(f"  WP {wp}: {count} crashes")
    kr = report.get('kryptonite', {})
    if kr.get('kryptonite_count', 0) > 0:
        print(f"\nKryptonite episodes: {kr['kryptonite_count']} / {kr['total_crashes']} crashes ({kr['kryptonite_rate']:.1%})")
        for pattern, details in kr.get('patterns', {}).items():
            print(f"  {pattern}: {details['count']}x, avg_speed={details['avg_speed']:.2f}, avg_progress={details['avg_progress']:.1f}%")
    ga = report.get('graze_vs_avoid', {})
    if ga:
        print(f"\nGraze vs Avoid:")
        print(f"  Graze eps: {ga.get('graze_episodes', 0)}, avg reward={ga.get('graze_avg_reward', 0):.1f}, progress={ga.get('graze_avg_progress', 0):.1f}%")
        print(f"  Clean eps: {ga.get('clean_episodes', 0)}, avg reward={ga.get('clean_avg_reward', 0):.1f}, progress={ga.get('clean_avg_progress', 0):.1f}%")
    cs = report.get('curvature_speed', {})
    if cs:
        print(f"\nCurvature x Speed:")
        print(f"  Crashed: curv*spd={cs.get('crashed_curvxspeed', 0):.2f}, cornering={cs.get('crashed_cornering', 0):.2f}")
        print(f"  Survived: curv*spd={cs.get('survived_curvxspeed', 0):.2f}, cornering={cs.get('survived_cornering', 0):.2f}")
    co = report.get('cornering', {})
    if co:
        print(f"\nCornering Technique:")
        print(f"  Score quartiles: p25={co.get('cornering_score_p25', 0):.2f}, p50={co.get('cornering_score_p50', 0):.2f}, p75={co.get('cornering_score_p75', 0):.2f}")
        print(f"  Good corner crash rate: {co.get('good_corner_crash_rate', 0):.1%}")
        print(f"  Poor corner crash rate: {co.get('poor_corner_crash_rate', 0):.1%}")
    print("="*60)


class FailurePointSampler:
# REF: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine Learning, 3(1), 9-44.
    """Samples and stores success/failure instances at key track points.

    Divides the track into segments by progress % and records
    per-step state snapshots when episodes fail (crash/offtrack/stuck)
    vs succeed (high progress or lap completion).
    """

    def __init__(self, save_dir='results', num_segments=10, max_samples=50):
        import os
        import random
        from collections import defaultdict

        self.save_dir = save_dir
        self.num_segments = num_segments
        self.max_samples = max_samples
        os.makedirs(save_dir, exist_ok=True)

        self.failure_instances = defaultdict(list)
        self.success_instances = defaultdict(list)
        self.ep_steps = []
        self.ep_episode_id = 0
        self._random = random
        self._defaultdict = defaultdict

    def record_step(self, step_data):
        self.ep_steps.append(step_data)

    def _get_segment(self, progress):
        return min(int(progress / (100.0 / self.num_segments)), self.num_segments - 1)

    def end_episode(self, ep_progress, ep_return, ep_length, terminated_reason='unknown', failure_wp=None, impact_velocity=0.0, barrier_type='unknown', crash_v_perp=0.0):
        self.ep_episode_id += 1
        if not self.ep_steps:
            return

        seg = self._get_segment(ep_progress)
        is_success = (ep_progress >= 95.0)
        is_failure = (ep_progress < 50.0)

        window = min(20, len(self.ep_steps))
        critical = self.ep_steps[-window:]

        mid_s = len(self.ep_steps) // 3
        mid_e = 2 * len(self.ep_steps) // 3
        n_mid = min(5, mid_e - mid_s)
        if n_mid > 0 and mid_e > mid_s:
            mid_idx = sorted(self._random.sample(range(mid_s, mid_e), min(n_mid, mid_e - mid_s)))
            mid_steps = [self.ep_steps[i] for i in mid_idx]
        else:
            mid_steps = []

        inst = {
            'episode_id': self.ep_episode_id,
            'progress': ep_progress,
            'return': ep_return,
            'length': ep_length,
            'reason': terminated_reason,
            'failure_wp': failure_wp,  # v39: waypoint index where failure occurred
            'impact_velocity': round(impact_velocity, 3),  # v39: speed at DQ
            'barrier_type': barrier_type,  # v39: wall/obstacle/bot
            'crash_v_perp': round(crash_v_perp, 3),  # v39: perpendicular impact velocity
            'segment': seg,
            'critical_steps': critical,
            'mid_steps': mid_steps,
            'first_step': self.ep_steps[0],
        }

        if is_failure:
            bucket = self.failure_instances[seg]
            if len(bucket) < self.max_samples:
                bucket.append(inst)
            else:
                idx = self._random.randint(0, self.ep_episode_id)
                if idx < self.max_samples:
                    bucket[idx] = inst

        if is_success:
            stride = max(1, len(self.ep_steps) // self.num_segments)
            for sd in self.ep_steps[::stride]:
                s = self._get_segment(sd.get('progress', 0))
                bucket = self.success_instances[s]
                snap = {
                    'episode_id': self.ep_episode_id,
                    'ep_progress': ep_progress,
                    'ep_return': ep_return,
                }
                snap.update(sd)

                if len(bucket) < self.max_samples:
                    bucket.append(snap)
                else:
                    idx = self._random.randint(0, self.ep_episode_id)
                    if idx < self.max_samples:
                        bucket[idx] = snap

        self.ep_steps = []

    def get_failure_hotspots(self):
        counts = {s: len(v) for s, v in self.failure_instances.items()}
        return sorted(counts.items(), key=lambda x: -x[1])

    def save(self, suffix=''):
        import json
        import os
        import numpy as np

        fo = {str(k): v for k, v in self.failure_instances.items()}
        so = {str(k): v for k, v in self.success_instances.items()}

        fp = os.path.join(self.save_dir, f'failure_instances{suffix}.json')
        sp = os.path.join(self.save_dir, f'success_instances{suffix}.json')
        hp = os.path.join(self.save_dir, f'failure_hotspots{suffix}.json')

        with open(fp, 'w') as f:
            json.dump(fo, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))

        with open(sp, 'w') as f:
            json.dump(so, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))

        hotspots = self.get_failure_hotspots()
        comps = []
        for seg_id, cnt in hotspots:
            fails = self.failure_instances.get(seg_id, [])
            succs = self.success_instances.get(seg_id, [])
            if fails:
                reasons = self._defaultdict(int)
                for i in fails:
                    reasons[i.get('reason', 'unknown')] += 1
                comps.append({
                    'segment': seg_id,
                    'fail_count': len(fails),
                    'hotspot_count': cnt,  # from most_common()
                    'success_count': len(succs),
                    'avg_progress': float(np.mean([i['progress'] for i in fails])),
                    'avg_return': float(np.mean([i['return'] for i in fails])),
                    'avg_impact_velocity': float(np.mean([i.get('impact_velocity',0) for i in fails])),  # v39
                    'avg_crash_v_perp': float(np.mean([i.get('crash_v_perp',0) for i in fails])),  # v39
                    'barrier_types': dict([(b, sum(1 for i in fails if i.get('barrier_type')==b)) for b in set(i.get('barrier_type','unknown') for i in fails)]),  # v39
                    'failure_wps': [i.get('failure_wp') for i in fails if i.get('failure_wp') is not None],  # v39: track locations
                    'reasons': dict(reasons),
                })

        with open(hp, 'w') as f:
            json.dump(comps, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))

        return fp, sp, hp