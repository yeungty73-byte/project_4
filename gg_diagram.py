"""Vehicle dynamics modules: G-G diagram, trail braking, velocity profile.

This module provides three grown-up functionalities from motorsport
engineering that serve as both reward signals (in run.py) and
intermediary metrics (in harmonized_metrics.py).

Each metric has a clear CAUSAL HYPOTHESIS about its relationship
to the success metric (avg_speed_centerline):

  1. GG Ellipse Utilisation:
     H: Higher grip utilisation -> faster cornering -> higher avg_speed.
     Math: u = sqrt((a_lat/a_max)^2 + (a_lon/a_max)^2), u in [0,1]
     Ref: Brach SAE-2011-01-0094; fswiki.us/G-g_Diagram

  2. Trail Braking Overlap:
     H: More trail braking -> better weight transfer -> faster corner
        entry -> higher avg_speed.
     Math: overlap = |brake| * |steer| when both > threshold
     Ref: Driver61 trail braking tutorial; Popometer.io telemetry

  3. Velocity Profile Compliance:
     H: Matching optimal speed targets per waypoint -> minimal time
        loss -> higher avg_speed.
     Math: compliance = 1 - |v - v_target|/v_target, per waypoint
     Ref: Heilmeier et al. min-curvature + speed profile; Kapania 2015

  4. Predictive Curvature Anticipation:
     H: Agent that anticipates upcoming curvature (brakes early,
        positions for apex) -> smoother + faster -> higher avg_speed.
     Math: curvature_readiness = cos(heading_to_apex) * brake_proximity
     Ref: Dynamic Lookahead PPO (2026 arXiv:2603.28625)
"""
import numpy as np
from config_loader import CFG


# ================================================================
#  1. GG Ellipse Utilisation
# ================================================================

class GGDiagram:
    """Track friction-ellipse utilisation across a trajectory.

    The g-g diagram plots lateral vs longitudinal acceleration.
    A vehicle should always be near the boundary of the friction ellipse:
        (a_lat / a_lat_max)^2 + (a_lon / a_lon_max)^2 <= 1
    Utilisation = how much of available grip the agent actually uses.
    """

    def __init__(self, mu=None, g=9.81, dt=0.05):
        self.mu = mu or CFG.get("brake_field", {}).get("mu", 0.7)
        self.g = g
        self.dt = dt
        self.a_max = self.mu * self.g
        self._prev_speed = None
        self._prev_heading = None
        self._utilisations = []

    def reset(self):
        self._prev_speed = None
        self._prev_heading = None
        self._utilisations = []

    def step(self, speed: float, heading_rad: float) -> float:
        """Returns instantaneous utilisation [0, 1]."""
        if self._prev_speed is None:
            self._prev_speed = speed
            self._prev_heading = heading_rad
            return 0.0
        a_lon = (speed - self._prev_speed) / self.dt
        d_heading = (heading_rad - self._prev_heading + np.pi) % (2*np.pi) - np.pi
        a_lat = speed * d_heading / self.dt
        self._prev_speed = speed
        self._prev_heading = heading_rad
        r_sq = (a_lat / self.a_max)**2 + (a_lon / self.a_max)**2
        u = min(np.sqrt(r_sq), 1.0)
        self._utilisations.append(u)
        return u

    @property
    def mean_utilisation(self) -> float:
        return float(np.mean(self._utilisations)) if self._utilisations else 0.0


# ================================================================
#  2. Trail Braking Detector
# ================================================================

class TrailBrakingDetector:
    """Detect and score trail braking: simultaneous braking + steering.

    Trail braking = applying brake pressure while turning.
    In telemetry: overlap region where |brake| > thresh AND |steer| > thresh.
    Quality = how smoothly brake is released as steering increases.
    Ref: Driver61; total-car-control.co.uk; Popometer.io
    """

    def __init__(self, brake_thresh=0.05, steer_thresh=0.05):
        self.brake_thresh = brake_thresh
        self.steer_thresh = steer_thresh
        self._overlaps = []  # per-step overlap quality
        self._n_corner_entries = 0
        self._n_trail_braked = 0

    def reset(self):
        self._overlaps = []
        self._n_corner_entries = 0
        self._n_trail_braked = 0

    def step(self, brake_pct: float, steer_abs: float,
             in_corner_entry: bool = False) -> float:
        """Returns trail braking quality [0, 1] for this step.
        Higher = better overlap of braking + turning."""
        is_braking = brake_pct > self.brake_thresh
        is_steering = steer_abs > self.steer_thresh
        if in_corner_entry:
            self._n_corner_entries += 1
            if is_braking and is_steering:
                self._n_trail_braked += 1
        if is_braking and is_steering:
            # Quality: proportional to both inputs being used
            overlap = min(brake_pct, 1.0) * min(steer_abs / 0.5, 1.0)
            self._overlaps.append(overlap)
            return float(np.clip(overlap, 0.0, 1.0))
        return 0.0

    @property
    def trail_brake_ratio(self) -> float:
        """Fraction of corner entries with trail braking."""
        if self._n_corner_entries == 0:
            return 0.0
        return self._n_trail_braked / self._n_corner_entries

    @property
    def mean_overlap(self) -> float:
        return float(np.mean(self._overlaps)) if self._overlaps else 0.0


# ================================================================
#  3. Velocity Profile Compliance
# ================================================================

def optimal_speed_at_curvature(curvature: float, mu: float = 0.7,
                               g: float = 9.81, v_min: float = 0.5,
                               v_max: float = 4.0) -> float:
    """v_target = sqrt(mu * g / max(curvature, eps)).
    Ref: v = sqrt(r * g) where r = 1/curvature.
    Capped to [v_min, v_max]."""
    if curvature < 1e-6:
        return v_max  # straight
    r = 1.0 / curvature
    v = np.sqrt(mu * g * r)
    return float(np.clip(v, v_min, v_max))


def velocity_profile_compliance(speed: float, v_target: float) -> float:
    """How well the agent matches the optimal speed.
    Returns 1 - |v - v_target| / v_target, clipped to [0, 1]."""
    if v_target < 1e-6:
        return 1.0
    return float(np.clip(1.0 - abs(speed - v_target) / v_target, 0.0, 1.0))


# ================================================================
#  4. Predictive Curvature Anticipation
# ================================================================

def curvature_at_waypoints(waypoints: np.ndarray, indices: list) -> list:
    """Compute curvature at given waypoint indices.
    Uses three-point method: curv = 2*|cross(v1,v2)| / (|v1|*|v2|*|v1-v2|)."""
    n = len(waypoints)
    curvatures = []
    for i in indices:
        i0 = (i - 1) % n
        i1 = i % n
        i2 = (i + 1) % n
        v1 = waypoints[i1] - waypoints[i0]
        v2 = waypoints[i2] - waypoints[i1]
        cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(v2 - v1)
        curvatures.append(2.0 * cross / (denom + 1e-8))
    return curvatures


def multi_horizon_curvature(waypoints: np.ndarray, current_wp: int,
                            horizons=(3, 6, 10, 15)) -> np.ndarray:
    """Compute curvature at multiple lookahead horizons.
    Returns array of shape (len(horizons),) with curvature values.
    Ref: Dynamic Lookahead PPO (arXiv:2603.28625) maps speed +
    multi-horizon curvature to optimal lookahead distance."""
    n = len(waypoints)
    indices = [(current_wp + h) % n for h in horizons]
    return np.array(curvature_at_waypoints(waypoints, indices))


def curvature_anticipation_score(speed: float, curvatures_ahead: np.ndarray,
                                  is_braking: bool, mu: float = 0.7,
                                  g: float = 9.81) -> float:
    """Score how well the agent anticipates upcoming curvature.
    If high curvature ahead and agent is already braking -> good.
    If high curvature ahead and agent is accelerating -> bad.
    Returns [0, 1]."""
    if len(curvatures_ahead) == 0:
        return 0.5
    max_curv = float(np.max(curvatures_ahead))
    if max_curv < 0.02:  # straight ahead
        return 0.5  # neutral
    # How much braking is needed
    v_target = optimal_speed_at_curvature(max_curv, mu, g)
    need_decel = max(0.0, speed - v_target)
    if need_decel > 0.5 and is_braking:
        return 1.0  # correctly anticipating
    elif need_decel > 0.5 and not is_braking:
        return 0.0  # failing to anticipate
    elif need_decel <= 0.5:
        return 0.7  # mild situation, no strong signal
    return 0.5
