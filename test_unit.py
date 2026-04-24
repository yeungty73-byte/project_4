"""Unit tests for the overhauled codebase.
Run:  python -m pytest test_unit.py -v
"""
import numpy as np
import pytest

# ---- Metrics tests ----

def test_avg_speed_centerline_perfect():
    from harmonized_metrics import avg_speed_centerline
    speeds = np.array([2.0, 3.0, 2.5])
    d_center = np.array([0.0, 0.0, 0.0])  # perfectly centered
    result = avg_speed_centerline(speeds, d_center, track_width=0.6)
    assert abs(result - 2.5) < 1e-6, f"Expected 2.5, got {result}"

def test_avg_speed_centerline_offtrack():
    from harmonized_metrics import avg_speed_centerline
    speeds = np.array([2.0])
    d_center = np.array([0.3])  # at edge (half_w = 0.3)
    result = avg_speed_centerline(speeds, d_center, track_width=0.6)
    assert result == 0.0, f"Expected 0.0 at edge, got {result}"

def test_track_progress_clamp():
    from harmonized_metrics import track_progress
    assert track_progress(1.5) == 1.0
    assert track_progress(-0.1) == 0.0

def test_smoothness_constant_steer():
    from harmonized_metrics import smoothness_jerk_rms
    steps = [{"steering": 0.5}] * 10
    assert smoothness_jerk_rms(steps) == 1.0  # zero jerk

def test_compute_all_returns_both_tiers():
    from harmonized_metrics import compute_all
    steps = [{"speed": 2.0, "distance_from_center": 0.05, "steering": 0.1}] * 5
    m = compute_all(steps, final_progress=0.8, n_waypoints=50)
    assert "avg_speed_centerline" in m
    assert "race_line_adherence" in m
    assert "gg_ellipse_utilisation" in m

# ---- VIF collinearity audit ----

def test_success_metrics_no_collinearity():
    """avg_speed_centerline and track_progress should have VIF < 5."""
    from harmonized_metrics import compute_all
    rng = np.random.RandomState(42)
    records = []
    for _ in range(200):
        prog = rng.uniform(0.3, 1.0)
        speed = rng.uniform(0.5, 4.0)
        d = rng.uniform(0.0, 0.25)
        steps = [{"speed": speed, "distance_from_center": d, "steering": 0.0}]
        m = compute_all(steps, prog, n_waypoints=50)
        records.append((m["avg_speed_centerline"], m["track_progress"]))
    arr = np.array(records)
    corr = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
    r_sq = corr ** 2
    vif = 1.0 / (1.0 - r_sq + 1e-12)
    assert vif < 5.0, f"VIF={vif:.2f} >= 5 means collinearity!"

# ---- Brake field tests ----

def test_braking_distance():
    from brake_field import braking_distance
    d = braking_distance(4.0, mu=0.7, g=9.81)
    expected = 16.0 / (2 * 0.7 * 9.81)
    assert abs(d - expected) < 1e-6

# ---- GG diagram tests ----

def test_gg_stationary():
    from gg_diagram import GGDiagram
    gg = GGDiagram(mu=0.7)
    gg.step(0.0, 0.0)
    u = gg.step(0.0, 0.0)
    assert u == 0.0  # no acceleration

# ---- Config tests ----

def test_config_loads():
    from config_loader import CFG
    assert "env" in CFG
    assert CFG["agent"]["gamma"] == 0.99

# ---- Smoke import ----

def test_smoke_imports():
    import harmonized_metrics
    import brake_field
    import gg_diagram
    import reward_shaping
    import config_loader
    import race_line_engine
    import corner_analysis
    import context_aware_agent
    import agents

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
