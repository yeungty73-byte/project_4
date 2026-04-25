"""Brake-field reward module.

Physics: braking distance d_brake = v^2 / (2 * mu * g)   [Wikipedia/Braking_distance]
The "brake field" is the region before each corner where the agent
SHOULD be decelerating.  We reward the agent for being inside this
field when approaching a corner, via a gradient w.r.t. proximity:

  Phi(s) = clamp(1 - dist_to_corner / (safety * d_brake), 0, 1)

This is a potential function suitable for PBRS.
"""
import numpy as np
from config_loader import CFG


def braking_distance(speed: float, mu: float = None, g: float = None,
                     safety: float = None) -> float:
    """d = v^2 / (2*mu*g) * safety_factor + jerk_onset_buffer

    REF: Brayshaw & Harrison (2005) Quasi-steady state lap simulation. Proc. IMechE Part D.
         Equation 3: braking distance under constant deceleration.
    REF: Balaban et al. (2018) Jerk as indicator of driving intention. Veh. Sys. Dyn.
         ~80ms brake onset lag before force builds: adds v * 0.08s to effective distance.

    The jerk_onset_dist models the real-world delay between the driver deciding to brake
    and the brakes actually generating significant deceleration force (~80ms for 1/18-scale).
    At 4 m/s this adds 0.32m — non-trivial vs the track width of ~1.07m.
    """
    _cfg = CFG.get("brake_field", {})
    mu  = mu  or _cfg.get("mu",             0.7)
    g   = g   or _cfg.get("g",              9.81)
    sf  = safety or _cfg.get("safety_margin", 1.2)
    # Physics braking distance
    d_physics = speed ** 2 / (2.0 * mu * g + 1e-8)
    # v1.1.0: jerk-onset buffer — ~80ms before brake force builds
    # REF: Balaban et al. (2018)
    _jerk_onset_dist = speed * 0.08
    return d_physics * sf + _jerk_onset_dist


class BrakeField:
    """Computes brake-field potential and compliance per step."""

    def __init__(self, waypoints=None):
        _cfg = CFG.get("brake_field", {})
        self.mu = _cfg.get("mu", 0.7)
        self.g = _cfg.get("g", 9.81)
        self.safety = _cfg.get("safety_margin", 1.2)
        self.lookahead = _cfg.get("lookahead_waypoints", 8)
        self.waypoints = waypoints  # np.ndarray (N, 2)
        self._corner_indices = None
        self._braking_events = []
        self._in_field_count = 0

    def set_waypoints(self, waypoints: np.ndarray):
        self.waypoints = waypoints
        self._corner_indices = None  # invalidate cache

    def _find_corners(self, curvature_threshold: float = 0.05):
        """Identify waypoint indices where curvature exceeds threshold."""
        if self.waypoints is None or len(self.waypoints) < 3:
            self._corner_indices = []
            return
        wps = self.waypoints
        corners = []
        for i in range(1, len(wps) - 1):
            v1 = wps[i] - wps[i-1]
            v2 = wps[i+1] - wps[i]
            cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
            norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
            curv = cross / norm
            if curv > curvature_threshold:
                corners.append(i)
        self._corner_indices = corners

    def potential(self, wp_idx: int, speed: float) -> float:
        """Compute brake-field potential Phi(s) in [0, 1]."""
        if self._corner_indices is None:
            self._find_corners()
        if not self._corner_indices or self.waypoints is None:
            return 0.0

        d_brake = braking_distance(speed, self.mu, self.g) * self.safety

        # Find nearest upcoming corner
        n = len(self.waypoints)
        min_corner_dist = float("inf")
        for ci in self._corner_indices:
            # Distance in waypoint-index space, wrapped
            idx_dist = (ci - wp_idx) % n
            if 0 < idx_dist <= self.lookahead:
                # Approximate physical distance
                phys_dist = 0.0
                for j in range(wp_idx, wp_idx + idx_dist):
                    j1 = j % n
                    j2 = (j + 1) % n
                    phys_dist += np.linalg.norm(
                        self.waypoints[j2] - self.waypoints[j1]
                    )
                min_corner_dist = min(min_corner_dist, phys_dist)

        if min_corner_dist == float("inf") or d_brake < 1e-6:
            return 0.0

        phi = np.clip(1.0 - min_corner_dist / d_brake, 0.0, 1.0)
        return float(phi)

    def step(self, wp_idx: int, speed: float, is_braking: bool) -> dict:
        """Called each env step. Returns enrichment dict for step record."""
        phi = self.potential(wp_idx, speed)
        in_field = phi > 0.0
        if is_braking:
            self._braking_events.append(in_field)
            if in_field:
                self._in_field_count += 1
        return dict(
            brake_potential=phi,
            in_brake_field=in_field,
            is_braking=is_braking,
        )

    @property
    def compliance(self) -> float:
        if not self._braking_events:
            return 1.0
        return self._in_field_count / len(self._braking_events)

    def reset(self):
        self._braking_events = []
        self._in_field_count = 0
