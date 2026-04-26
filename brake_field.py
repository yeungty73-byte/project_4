"""Brake-field reward module — v1.1.0.
Physics: d = v²/(2μg)·sf + v·0.08 [Brayshaw&Harrison 2005; Balaban 2018]
v1.1.0: SwinUNet++ clearance gating (multiplicative only — no freeze trap).
v1.1.1: Added reset() method for episode boundary clearing.
"""
import numpy as np
from config_loader import CFG

def braking_distance(speed, mu=None, g=None, safety=None):
    _cfg = CFG.get("brake_field", {})
    mu = mu or _cfg.get("mu", 0.7)
    g = g or _cfg.get("g", 9.81)
    sf = safety or _cfg.get("safety_margin", 1.2)
    return (speed**2 / (2.0*mu*g + 1e-8)) * sf + speed*0.08

class BrakeField:
    def __init__(self, waypoints=None):
        _cfg = CFG.get("brake_field", {})
        self.mu       = _cfg.get("mu", 0.7)
        self.g        = _cfg.get("g", 9.81)
        self.safety   = _cfg.get("safety_margin", 1.2)
        self.lookahead= _cfg.get("lookahead_waypoints", 8)
        self.waypoints= waypoints
        self._corner_indices = None
        self._braking_events = []
        self._in_field_count = 0

    def reset(self):
        """v1.1.1: Clear per-episode accumulators at episode boundary.
        Called by run.py at the top of each episode reset block.
        Does NOT clear waypoints or corner cache — those are track-level, not episode-level.
        """
        self._braking_events = []
        self._in_field_count = 0

    def set_waypoints(self, waypoints):
        self.waypoints = waypoints
        self._corner_indices = None

    def _find_corners(self, thresh=0.05):
        if self.waypoints is None or len(self.waypoints) < 3:
            self._corner_indices = []; return
        wps = self.waypoints; corners = []
        for i in range(1, len(wps)-1):
            v1 = wps[i]-wps[i-1]; v2 = wps[i+1]-wps[i]
            cross = abs(v1[0]*v2[1]-v1[1]*v2[0])
            norm = (np.linalg.norm(v1)*np.linalg.norm(v2))+1e-8
            if cross/norm > thresh: corners.append(i)
        self._corner_indices = corners

    def potential(self, wp_idx, speed):
        if self._corner_indices is None: self._find_corners()
        if not self._corner_indices or self.waypoints is None: return 0.0
        d_brake = braking_distance(speed, self.mu, self.g)*self.safety
        n = len(self.waypoints); min_dist = float("inf")
        for ci in self._corner_indices:
            hops = (ci-wp_idx) % n
            if hops <= self.lookahead:
                seg = np.linalg.norm(self.waypoints[(ci+1)%n]-self.waypoints[ci])
                min_dist = min(min_dist, hops*max(seg,0.1))
        if min_dist == float("inf"): return 0.0
        return float(min(max(0.0, 1.0-min_dist/max(d_brake,0.01)), 1.0))

    def step(self, wp_idx, speed, is_braking, swin_clearance=None):
        """
        Returns dict: potential, potential_gated, in_brake_field,
        compliance, swin_front_clear, reward_multiplier.
        swin_clearance: np.ndarray(4,) [front,left,right,rear] from SwinUNet++.
        Attenuation/boost is MULTIPLICATIVE — no freeze trap.
        """
        phi = self.potential(wp_idx, speed)
        in_field = phi > 0.25
        front_clear = 0.5  # neutral when Swin unavailable
        if swin_clearance is not None and len(swin_clearance) >= 1:
            front_clear = float(np.clip(swin_clearance[0], 0.0, 1.0))
        if in_field:
            if front_clear > 0.7:   # open road — reduce false-positive braking
                gate = 1.0 / (1.0 + (front_clear-0.7)*1.5)
            elif front_clear < 0.3: # obstacle ahead — boost braking urgency
                gate = 1.0 + (0.3-front_clear)*0.8
            else:
                gate = 1.0
        else:
            gate = 1.0
        phi_gated = float(np.clip(phi*gate, 0.0, 1.0))
        compliance = in_field and is_braking
        if in_field and is_braking:
            rw_mult = 1.0 + phi_gated*0.5
        elif in_field and not is_braking:
            rw_mult = 1.0 / (1.0 + phi_gated*0.6)
        else:
            rw_mult = 1.0
        if compliance:
            self._in_field_count += 1
            self._braking_events.append({"wp":wp_idx,"speed":speed,"phi":phi_gated})
        return {"potential":phi,"potential_gated":phi_gated,
                "in_brake_field":in_field,"compliance":compliance,
                "swin_front_clear":front_clear,"reward_multiplier":float(rw_mult)}

    @property
    def compliance_rate(self):
        if not self._braking_events: return 0.0
        return self._in_field_count/max(len(self._braking_events),1)
