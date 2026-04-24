import csv
import os
import time
import numpy as np
from datetime import datetime

class EpisodeMetricsLogger:
# REF: Amazon Web Services. (2020). AWS DeepRacer: Developer documentation. https://docs.aws.amazon.com/deepracer/
    HEADER = [
        'episode_id', 'global_step', 'wall_time', 'datetime',
        'track_name', 'race_type',
        'progress', 'lap_time', 'episode_return', 'episode_length',
        'termination_reason',
        'avg_speed', 'avg_center_dist', 'avg_heading_diff', 'max_steering'
    ]
    def __init__(self, log_dir, run_name, track_name='unknown', race_type='time_trial'):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, f'{run_name}_episodes.csv')
        self.track_name = track_name
        self.race_type = race_type
        self.episode_count = 0
        self._step_speeds = []
        self._step_center_dists = []
        self._step_heading_diffs = []
        self._step_steerings = []
        write_header = not os.path.exists(self.filepath)
        self._file = open(self.filepath, 'a', newline='')
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(self.HEADER)
            self._file.flush()
    def reset_episode(self):
        self._step_speeds = []
        self._step_center_dists = []
        self._step_heading_diffs = []
        self._step_steerings = []
    def log_step(self, info, action=None):
        rp = info.get('reward_params', {})
        self._step_speeds.append(rp.get('speed', 0.0))
        self._step_center_dists.append(abs(rp.get('distance_from_center', 0.0)))
        heading = rp.get('heading', 0.0)
        waypoints = rp.get('waypoints', [])
        closest_wp = rp.get('closest_waypoints', [0, 1])
        if len(waypoints) > max(closest_wp):
            import math
            prev_wp = waypoints[closest_wp[0]]
            next_wp = waypoints[closest_wp[1]]
            track_dir = math.degrees(math.atan2(next_wp[1] - prev_wp[1], next_wp[0] - prev_wp[0]))
            diff = abs(track_dir - heading)
            if diff > 180:
                diff = 360 - diff
            self._step_heading_diffs.append(diff)
        else:
            self._step_heading_diffs.append(0.0)
        if action is not None:
            if hasattr(action, '__len__') and len(action) > 1:
                self._step_steerings.append(abs(float(action[-1])))
            else:
                self._step_steerings.append(0.0)
    def log_episode_end(self, global_step, info, episode_return, episode_length,
                        terminated, truncated):
        rp = info.get('reward_params', {})
        progress = rp.get('progress', 0.0)
        if progress >= 100.0:
            ep_info = info.get('episode', {})
            if isinstance(ep_info.get('t'), np.ndarray):
                lap_time = float(ep_info['t'].mean())
            elif 't' in ep_info:
                lap_time = float(ep_info['t'])
            else:
                lap_time = float('nan')
        else:
            lap_time = float('nan')
        if not terminated and not truncated:
            term_reason = 'running'
        elif progress >= 100.0:
            term_reason = 'completed'
        elif rp.get('is_crashed', False):
            term_reason = 'crashed'
        elif not rp.get('all_wheels_on_track', True):
            term_reason = 'off_track'
        elif rp.get('is_reversed', False):
            term_reason = 'reversed'
        elif truncated and not terminated:
            term_reason = 'truncated'
        else:
            term_reason = 'unknown'
        now = time.time()
        self.episode_count += 1
        row = [
            self.episode_count, global_step, now,
            datetime.fromtimestamp(now).isoformat(),
            self.track_name, self.race_type,
            progress, lap_time, episode_return, episode_length,
            term_reason,
            np.mean(self._step_speeds) if self._step_speeds else 0.0,
            np.mean(self._step_center_dists) if self._step_center_dists else 0.0,
            np.mean(self._step_heading_diffs) if self._step_heading_diffs else 0.0,
            max(self._step_steerings) if self._step_steerings else 0.0,
        ]
        self._writer.writerow(row)
        self._file.flush()
        self.reset_episode()
        return {'progress': progress, 'lap_time': lap_time, 'termination_reason': term_reason}
    def close(self):
        self._file.close()
