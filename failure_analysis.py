"""failure_analysis.py  v5.0 — forensic sampler, wired into replay and reward.

What changed in v5.0
---------------------
  1. FailurePointSampler.end_episode() now computes and returns a 'hotspot_signal'
     dict that run.py can use IMMEDIATELY to:
       a) upweight replay buffer samples from high-failure track segments
       b) add a per-step 'failure_zone_penalty' bonus (negative) when the
          agent enters a known hotspot segment

  2. get_failure_hotspots() now returns a normalised float dict so run.py
     doesn't have to post-process — keys are segment IDs, values are
     failure_density in [0, 1].

  3. FailurePointSampler.step_segment_id(progress_pct) is a new helper
     that run.py calls each step to decide whether to apply the hotspot
     bonus without iterating over the whole hotspot list.

  4. FailureAnalyzer is retained for post-hoc CLI reporting.

  5. save() now writes forensic_episodes.jsonl per-episode (not end-of-run)
     so crash antecedents are never lost if training crashes.

Hotspot wiring contract (copy into run.py step-reward block)
------------------------------------------------------------
  # --- v5 hotspot penalty ---
  _seg = failure_sampler.step_segment_id(ep_progress)
  _hd  = failure_sampler.hotspot_density(_seg)        # float [0,1]
  if _hd > 0.4 and not _offtrack:
      reward -= 0.05 * _hd                            # discourage drift into graveyard

  # --- v5 replay prioritisation (call once per episode end) ---
  _hs_sig = failure_sampler.end_episode(...)          # returns hotspot_signal dict
  if _hs_sig['is_hotspot']:
      td3sac.replay.boost_priority(_seg, factor=1.5)  # if replay supports it
"""
from __future__ import annotations
import json, csv, os, math, random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple


# ================================================================
# FailurePointSampler  (online, wired to hot path)
# ================================================================

class FailurePointSampler:
    """Per-episode failure forensics with live hotspot feedback.

    Segments the track into num_segments bins by progress %.
    For each episode end, records critical steps and returns a
    hotspot_signal usable by run.py's reward and replay logic.

    REF: Sutton (1988) TD-learning; prioritised ER (Schaul et al. 2016).
    """

    def __init__(self, save_dir: str = 'results',
                 num_segments: int = 10,
                 max_samples: int = 50,
                 hotspot_threshold: float = 0.4):
        self.save_dir          = save_dir
        self.num_segments      = num_segments
        self.max_samples       = max_samples
        self.hotspot_threshold = hotspot_threshold
        os.makedirs(save_dir, exist_ok=True)

        self.failure_instances: Dict[int, List[dict]] = defaultdict(list)
        self.success_instances: Dict[int, List[dict]] = defaultdict(list)
        self._fail_counts: np.ndarray = np.zeros(num_segments, dtype=np.float32)
        self._total_eps:   int        = 0
        self._ep_steps:    List[dict] = []
        self._ep_id:       int        = 0

        # Path for live forensic JSONL (appended per crash)
        self._forensic_path = os.path.join(save_dir, 'forensic_episodes.jsonl')

    # ----------------------------------------------------------
    # Step-level helpers (call from run.py per step)
    # ----------------------------------------------------------

    def record_step(self, step_data: dict):
        """Append one step record to the current episode buffer."""
        self._ep_steps.append(step_data)

    def step_segment_id(self, progress_pct: float) -> int:
        """Return the segment ID for a given progress %.  O(1)."""
        return int(min(progress_pct / (100.0 / self.num_segments),
                       self.num_segments - 1))

    def hotspot_density(self, segment_id: int) -> float:
        """
        Normalised failure density for segment [0, 1].
        0 = never failed here; 1 = highest failure density so far.
        """
        if self._total_eps < 5:
            return 0.0          # not enough data yet
        total = self._fail_counts.sum()
        if total < 1:
            return 0.0
        seg_id = int(np.clip(segment_id, 0, self.num_segments - 1))
        return float(self._fail_counts[seg_id] / total)

    def is_hotspot(self, segment_id: int) -> bool:
        return self.hotspot_density(segment_id) >= self.hotspot_threshold

    # ----------------------------------------------------------
    # Episode end  (call from run.py at episode termination)
    # ----------------------------------------------------------

    def end_episode(
        self,
        ep_progress:   float,
        ep_return:     float,
        ep_length:     int,
        terminated_reason: str = 'unknown',
        failure_wp:    Optional[int]   = None,
        impact_velocity: float         = 0.0,
        barrier_type:  str             = 'unknown',
        crash_v_perp:  float           = 0.0,
    ) -> dict:
        """
        Finalise episode.  Returns hotspot_signal dict for run.py.

        hotspot_signal keys
        -------------------
          segment_id    : int    which track segment this episode ended in
          is_hotspot    : bool   True if this segment is a hotspot
          density       : float  [0,1] failure density of this segment
          is_failure    : bool   True if ep was a failure episode
          replay_boost  : float  suggested replay priority multiplier (1.0 = no boost)
        """
        self._ep_id    += 1
        self._total_eps += 1
        seg = self.step_segment_id(ep_progress)

        is_failure = (ep_progress < 50.0 or
                      terminated_reason in ('crashed', 'off_track', 'stuck'))
        is_success = (ep_progress >= 95.0)

        # Build instance snapshot
        window   = min(20, len(self._ep_steps))
        critical = self._ep_steps[-window:]

        mid_s = len(self._ep_steps) // 3
        mid_e = 2 * len(self._ep_steps) // 3
        mid_steps: List[dict] = []
        if mid_e > mid_s:
            n_mid = min(5, mid_e - mid_s)
            idxs  = sorted(random.sample(range(mid_s, mid_e),
                                         min(n_mid, mid_e - mid_s)))
            mid_steps = [self._ep_steps[i] for i in idxs]

        inst = dict(
            episode_id      = self._ep_id,
            progress        = ep_progress,
            ep_return       = ep_return,
            length          = ep_length,
            reason          = terminated_reason,
            failure_wp      = failure_wp,
            impact_velocity = round(float(impact_velocity), 3),
            barrier_type    = barrier_type,
            crash_v_perp    = round(float(crash_v_perp), 3),
            segment         = seg,
            critical_steps  = critical,
            mid_steps       = mid_steps,
            first_step      = self._ep_steps[0] if self._ep_steps else {},
        )

        # Reservoir-sample failures
        if is_failure:
            self._fail_counts[seg] += 1.0
            bucket = self.failure_instances[seg]
            if len(bucket) < self.max_samples:
                bucket.append(inst)
            else:
                idx = random.randint(0, self._ep_id)
                if idx < self.max_samples:
                    bucket[idx] = inst

            # Append to live forensic JSONL immediately
            try:
                with open(self._forensic_path, 'a') as f:
                    json.dump(inst, f,
                              default=lambda o: o.tolist()
                              if hasattr(o, 'tolist') else float(o))
                    f.write('\n')
            except Exception:
                pass

        # Reservoir-sample successes
        if is_success:
            stride = max(1, len(self._ep_steps) // self.num_segments)
            for step in self._ep_steps[::stride]:
                s  = self.step_segment_id(float(step.get('progress', 0)))
                snap = dict(episode_id=self._ep_id,
                            ep_progress=ep_progress,
                            ep_return=ep_return, **step)
                bucket = self.success_instances[s]
                if len(bucket) < self.max_samples:
                    bucket.append(snap)
                else:
                    idx = random.randint(0, self._ep_id)
                    if idx < self.max_samples:
                        bucket[idx] = snap

        self._ep_steps = []

        # Hotspot signal for run.py
        density = self.hotspot_density(seg)
        return dict(
            segment_id   = seg,
            is_hotspot   = density >= self.hotspot_threshold,
            density      = density,
            is_failure   = is_failure,
            replay_boost = float(1.0 + density) if is_failure else 1.0,
        )

    # ----------------------------------------------------------
    # Hotspot summary (for reward shaping & TensorBoard)
    # ----------------------------------------------------------

    def get_failure_hotspots(self) -> Dict[int, float]:
        """
        Return normalised failure density per segment.
        dict[segment_id -> density_float]  (0.0 = no failures).
        Used by run.py for per-step hotspot penalty.
        """
        total = self._fail_counts.sum()
        if total < 1:
            return {s: 0.0 for s in range(self.num_segments)}
        return {s: float(self._fail_counts[s] / total)
                for s in range(self.num_segments)}

    def hotspot_vector(self) -> np.ndarray:
        """
        Normalised density as float32 array shape (num_segments,).
        Suitable for TensorBoard histograms and replay priority multiplication.
        """
        total = self._fail_counts.sum()
        if total < 1:
            return np.zeros(self.num_segments, dtype=np.float32)
        return (self._fail_counts / total).astype(np.float32)

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def save(self, suffix: str = '') -> Tuple[str, str, str]:
        """Write failure_instances, success_instances, failure_hotspots JSON."""
        _serial = lambda o: o.tolist() if hasattr(o, 'tolist') else float(o)

        fo = {str(k): v for k, v in self.failure_instances.items()}
        so = {str(k): v for k, v in self.success_instances.items()}
        fp = os.path.join(self.save_dir, f'failure_instances{suffix}.json')
        sp = os.path.join(self.save_dir, f'success_instances{suffix}.json')
        hp = os.path.join(self.save_dir, f'failure_hotspots{suffix}.json')

        with open(fp, 'w') as f:
            json.dump(fo, f, indent=2, default=_serial)
        with open(sp, 'w') as f:
            json.dump(so, f, indent=2, default=_serial)

        hotspots = sorted(self.failure_instances.items(),
                          key=lambda x: -len(x[1]))
        comps = []
        for seg_id, fails in hotspots:
            succs = self.success_instances.get(seg_id, [])
            reasons = Counter(i.get('reason', 'unknown') for i in fails)
            barriers = Counter(i.get('barrier_type', 'unknown') for i in fails)
            comps.append(dict(
                segment             = seg_id,
                fail_count          = len(fails),
                success_count       = len(succs),
                density             = float(self.hotspot_density(seg_id)),
                avg_progress        = float(np.mean([i['progress'] for i in fails])),
                avg_return          = float(np.mean([i['ep_return'] for i in fails])),
                avg_impact_velocity = float(np.mean([i.get('impact_velocity', 0) for i in fails])),
                avg_crash_v_perp    = float(np.mean([i.get('crash_v_perp', 0) for i in fails])),
                barrier_types       = dict(barriers),
                failure_wps         = [i.get('failure_wp') for i in fails
                                       if i.get('failure_wp') is not None],
                reasons             = dict(reasons),
            ))
        with open(hp, 'w') as f:
            json.dump(comps, f, indent=2, default=_serial)
        return fp, sp, hp


# ================================================================
# FailureAnalyzer  (post-hoc CLI reporting, unchanged API)
# ================================================================

class FailureAnalyzer:
    """Post-hoc forensic analysis from saved JSONL + CSV."""

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir       = log_dir
        self.forensic_path = os.path.join(log_dir, 'forensic_episodes.jsonl')
        self.csv_path      = os.path.join(log_dir, 'bsts_metrics.csv')

    def _load_forensic(self) -> List[dict]:
        eps = []
        if not os.path.exists(self.forensic_path):
            return eps
        with open(self.forensic_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        eps.append(json.loads(line))
                    except Exception:
                        pass
        return eps

    def _load_csv(self) -> List[dict]:
        if not os.path.exists(self.csv_path):
            return []
        with open(self.csv_path) as f:
            rows = list(csv.DictReader(f))
        out = []
        for r in rows:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = v
            out.append(row)
        return out

    def stuck_waypoint_analysis(self, top_n: int = 10) -> dict:
        eps = self._load_forensic()
        if not eps:
            return {}
        c = Counter()
        for ep in eps:
            wp = ep.get('failure_wp') or ep.get('closest_waypoint_end')
            if wp is not None:
                c[int(float(wp))] += 1
        return dict(stuck_waypoints=c.most_common(top_n), total_crashes=len(eps))

    def graze_vs_avoid_analysis(self) -> dict:
        rows = self._load_csv()
        if not rows:
            return {}
        def _avg(lst, k):
            vals = [float(r[k]) for r in lst if r.get(k, '') != '']
            return float(np.mean(vals)) if vals else 0.0
        grazers = [r for r in rows if int(float(r.get('graze_count', 0))) > 0
                   and not int(float(r.get('crash', 0)))]
        clean   = [r for r in rows if int(float(r.get('graze_count', 0))) == 0
                   and not int(float(r.get('crash', 0)))]
        return dict(
            graze_episodes        = len(grazers),
            clean_episodes        = len(clean),
            graze_avg_progress    = _avg(grazers, 'progress'),
            clean_avg_progress    = _avg(clean, 'progress'),
            graze_avg_speed       = _avg(grazers, 'ep_speed_mean'),
            clean_avg_speed       = _avg(clean, 'ep_speed_mean'),
        )

    def curvature_speed_analysis(self) -> dict:
        rows = self._load_csv()
        if not rows:
            return {}
        crashed  = [r for r in rows if int(float(r.get('crash', 0)))]
        survived = [r for r in rows if not int(float(r.get('crash', 0)))]
        def _avg(lst, k):
            vals = [float(r[k]) for r in lst if r.get(k, '') != '']
            return float(np.mean(vals)) if vals else 0.0
        return dict(
            crashed_curvxspeed   = _avg(crashed,  'curvature_x_speed'),
            survived_curvxspeed  = _avg(survived, 'curvature_x_speed'),
            crashed_avg_speed    = _avg(crashed,  'avg_speed'),
            survived_avg_speed   = _avg(survived, 'avg_speed'),
        )

    def full_report(self) -> dict:
        report = dict(
            stuck_waypoints = self.stuck_waypoint_analysis(),
            graze_vs_avoid  = self.graze_vs_avoid_analysis(),
            curvature_speed = self.curvature_speed_analysis(),
        )
        out = os.path.join(self.log_dir, 'failure_report.json')
        with open(out, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f'[FailureAnalyzer] report saved to {out}')
        return report
