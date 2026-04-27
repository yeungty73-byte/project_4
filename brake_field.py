"""brake_field.py — v1.2.0
Physics: real-time perp-velocity constraint.  v_perp = v·|sin(heading - barrier_angle)|
Compliance is CONTINUOUS [0,1] proportional to how well agent follows the field.
brake_potential key always present (was missing in v1.1.x → KeyError in run.py).
REF: Brayshaw & Harrison (2005); Heinzmann & Zelinsky (2003).
"""
import math
import numpy as np
from typing import Optional
from config_loader import CFG


def braking_distance(speed, mu=0.7, g=9.81, safety=1.2):
    return (speed ** 2 / (2.0 * mu * g + 1e-8)) * safety + speed * 0.08


def v_perp_impact(speed, heading_rad, barrier_angle_rad):
    return float(abs(speed * math.sin(heading_rad - barrier_angle_rad)))


def required_decel_to_zero_perp(speed, heading_rad, barrier_angle_rad,
                                  dist_to_barrier, mu=0.7, g=9.81):
    vp = v_perp_impact(speed, heading_rad, barrier_angle_rad)
    d  = max(dist_to_barrier, 0.01)
    return float(vp ** 2 / (2.0 * d))


def brake_compliance_gradient(speed, heading_rad, barrier_angle_rad,
                                dist_to_barrier, is_braking, actual_decel,
                                mu=0.7, g=9.81):
    """Continuous compliance ∈ [0,1]. Proportional to decel quality vs required."""
    vp_needed  = required_decel_to_zero_perp(speed, heading_rad, barrier_angle_rad,
                                              dist_to_barrier, mu, g)
    vp_current = v_perp_impact(speed, heading_rad, barrier_angle_rad)
    if vp_current < 0.05:
        return 1.0
    urgency = float(np.clip(vp_needed / (mu * g + 1e-8), 0.0, 2.0))
    if urgency < 0.15:
        return 1.0
    if is_braking and actual_decel > 0:
        decel_ratio = float(np.clip(actual_decel / (vp_needed + 1e-8), 0.0, 1.0))
        return float(0.3 + 0.7 * decel_ratio)
    return float(np.clip(1.0 - urgency * 0.8, 0.05, 0.85))


class BrakeField:
    def __init__(self, waypoints=None):
        _cfg           = CFG.get("brake_field", {})
        self.mu        = _cfg.get("mu", 0.7)
        self.g         = _cfg.get("g", 9.81)
        self.safety    = _cfg.get("safety_margin", 1.2)
        self.lookahead = _cfg.get("lookahead_waypoints", 8)
        self.waypoints = (np.array([[w[0], w[1]] for w in waypoints], dtype=float)
                          if waypoints is not None else None)
        self._corner_indices  = None
        self._compliance_sum  = 0.0
        self._step_count      = 0
        self._braking_events  = []
        self._in_field_count  = 0

    def reset(self):
        self._compliance_sum = 0.0
        self._step_count     = 0
        self._braking_events = []
        self._in_field_count = 0

    def set_waypoints(self, waypoints):
        self.waypoints       = np.array([[w[0], w[1]] for w in waypoints], dtype=float)
        self._corner_indices = None

    def _find_corners(self, thresh=0.05):
        if self.waypoints is None or len(self.waypoints) < 3:
            self._corner_indices = []
            return
        wps     = self.waypoints
        corners = []
        for i in range(1, len(wps) - 1):
            v1    = wps[i] - wps[i - 1]
            v2    = wps[i + 1] - wps[i]
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            norm  = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
            if cross / norm > thresh:
                corners.append(i)
        self._corner_indices = corners

    def _corner_potential(self, wp_idx, speed):
        if self._corner_indices is None:
            self._find_corners()
        if not self._corner_indices or self.waypoints is None:
            return 0.0
        d_brake  = braking_distance(speed, self.mu, self.g) * self.safety
        n        = len(self.waypoints)
        min_dist = float("inf")
        for ci in self._corner_indices:
            hops = (ci - wp_idx) % n
            if hops <= self.lookahead:
                seg      = np.linalg.norm(self.waypoints[(ci + 1) % n] - self.waypoints[ci])
                min_dist = min(min_dist, hops * max(seg, 0.1))
        if min_dist == float("inf"):
            return 0.0
        return float(np.clip(1.0 - min_dist / max(d_brake, 0.01), 0.0, 1.0))

    def step(self, wp_idx, speed, heading_rad=0.0, is_braking=False,
             actual_decel=0.0, barrier_dist=5.0, barrier_angle=0.0,
             swin_clearance=None):
        try:
            front_clear = right_clear = left_clear = 0.5
            if swin_clearance is not None and len(swin_clearance) >= 3:
                front_clear = float(np.clip(swin_clearance[0], 0.0, 1.0))
                left_clear  = float(np.clip(swin_clearance[1], 0.0, 1.0))
                right_clear = float(np.clip(swin_clearance[2], 0.0, 1.0))

            eff_dist = barrier_dist
            if front_clear > 0.85 and barrier_dist > 3.0:
                eff_dist = min(10.0, barrier_dist * (1.0 + front_clear))

            vp_needed  = required_decel_to_zero_perp(speed, heading_rad, barrier_angle,
                                                      eff_dist, self.mu, self.g)
            vp_current = v_perp_impact(speed, heading_rad, barrier_angle)
            urgency    = float(np.clip(vp_needed / (self.mu * self.g + 1e-8), 0.0, 2.0))
            in_field   = urgency > 0.15

            phi = self._corner_potential(wp_idx, speed)
            if front_clear < 0.8 and eff_dist < 3.0:
                phi_eff = float(np.clip(max(phi, urgency * 0.5), 0.0, 1.0))
            else:
                phi_eff = phi

            gate = 1.0
            if in_field:
                if front_clear > 0.7:
                    gate = 1.0 / (1.0 + (front_clear - 0.7) * 1.5)
                elif front_clear < 0.3:
                    gate = 1.0 + (0.3 - front_clear) * 0.8
            phi_gated       = float(np.clip(phi_eff * gate, 0.0, 1.0))
            brake_potential = float(np.clip(max(phi_gated, urgency * 0.5), 0.0, 1.0))

            c_grad  = brake_compliance_gradient(speed, heading_rad, barrier_angle,
                                                eff_dist, is_braking, actual_decel,
                                                self.mu, self.g)
            if in_field:
                rw_mult = float(np.clip(1.0 + (c_grad - 0.5) * 1.0, 0.2, 2.0))
            else:
                rw_mult = 1.0

            self._step_count     += 1
            self._compliance_sum += c_grad
            if in_field:
                self._in_field_count += 1
                self._braking_events.append({"wp": wp_idx, "speed": speed,
                                              "c_grad": c_grad, "urgency": urgency})

            return {
                "potential":            phi,
                "potential_gated":      phi_gated,
                "brake_potential":      brake_potential,
                "in_brake_field":       in_field,
                "compliance_gradient":  c_grad,
                "reward_multiplier":    rw_mult,
                "v_perp":               vp_current,
                "urgency":              urgency,
                "swin_front_clear":     front_clear,
                "compliance":           bool(in_field and is_braking),
            }
        except Exception:
            return {"potential": 0.0, "potential_gated": 0.0, "brake_potential": 0.0,
                    "in_brake_field": False, "compliance_gradient": 1.0,
                    "reward_multiplier": 1.0, "v_perp": 0.0, "urgency": 0.0,
                    "swin_front_clear": 0.5, "compliance": False}

    @property
    def compliance_rate(self):
        if not self._braking_events:
            return 0.0
        return self._in_field_count / max(len(self._braking_events), 1)

    @property
    def mean_compliance_gradient(self):
        if self._step_count == 0:
            return 1.0
        return float(np.clip(self._compliance_sum / self._step_count, 0.0, 1.0))
