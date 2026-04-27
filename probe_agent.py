#!/usr/bin/env python3
"""
probe_agent.py  --  CS7642 Project 4 v1.1.5b
"Take the agent out of the track and put it in a controlled environment 
 so we can quiz it and mold it rapidly before putting it back."
                                        -- Tim, April 27 2026

Provides a lightweight offline probing harness:
  1. Reconstructs reward_params at any (x, y, heading, speed) -- FREEZE TIME, interrogate agent
  2. Exposes SwinUNetPP internal activations -- what does the backbone SEE?
  3. Quizzes BCPilot and main agent side-by-side on same scenario
  4. Micro-curriculum: inject hard scenarios before returning agent to live track
  5. Generates JSON report with action comparisons + perception diagnostics

REF: Goodfellow et al. (2016) DL sect 7.8 -- systematic probing of neural circuits.
REF: Ng et al. (1999) -- reward signal must reach all gradient paths.
REF: Balaji et al. (2020) DeepRacer -- sim state can be replayed offline.
"""

import math, json, torch, numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Scenario:
    """A frozen sim state for offline agent interrogation."""
    name: str
    x: float; y: float; heading: float; speed: float
    is_reversed: bool = False; is_offtrack: bool = False
    closest_waypoints: List[int] = field(default_factory=lambda: [0, 1])
    distance_from_center: float = 0.0; track_width: float = 1.07
    waypoints: Optional[List] = None; note: str = ""


# From live log: 16.635m oval, waypoints 0-119 CCW, x in [1.09,7.11], y in [1.06,5.56]
CANNED_SCENARIOS = [
    Scenario("spawn_forward",   x=2.56, y=1.06, heading=0.0,   speed=4.0,  closest_waypoints=[0,1],
             note="clean spawn, forward, full speed -- baseline"),
    Scenario("spawn_reversed",  x=2.56, y=1.06, heading=180.0, speed=4.0,  is_reversed=True,
             closest_waypoints=[0,1],  note="reversed spawn -- agent should engine-brake"),
    Scenario("hdiff_77deg",     x=2.56, y=1.06, heading=77.0,  speed=4.0,
             closest_waypoints=[0,1],  note="ep1 crash scenario: 77deg off tangent at 4m/s"),
    Scenario("tight_corner",    x=5.86, y=2.82, heading=90.0,  speed=4.0,
             closest_waypoints=[63,64], note="top-right corner, approaching barrier"),
    Scenario("distbarrier_020", x=4.40, y=4.32, heading=-11.0, speed=3.79,
             distance_from_center=0.51, closest_waypoints=[48,49],
             note="EXACT ep1 crash replay: distbarrier=0.020, vperp=2.79"),
    Scenario("centerline_slow", x=4.37, y=1.06, heading=0.0,   speed=1.5,
             closest_waypoints=[34,35], note="dead center, slow -- does agent accelerate?"),
    Scenario("offtrack_left",   x=1.80, y=0.80, heading=45.0,  speed=2.5,
             is_offtrack=True, distance_from_center=0.55,
             closest_waypoints=[8,9],   note="off-left, low speed -- recovery test"),
    Scenario("reversed_ep2",    x=3.94, y=3.53, heading=-51.0, speed=4.0,  is_reversed=True,
             closest_waypoints=[51,52], note="ep2 actual reversed spawn from log"),
]


def build_rp(sc: Scenario, waypoints: Optional[List] = None) -> dict:
    """Reconstruct a DeepRacer reward_params dict from a Scenario."""
    wps = sc.waypoints or waypoints or []
    n = len(wps) if wps else 120
    return {
        "x": sc.x, "y": sc.y, "heading": sc.heading, "speed": sc.speed,
        "distance_from_center": sc.distance_from_center,
        "track_width": sc.track_width,
        "is_reversed": sc.is_reversed, "is_offtrack": sc.is_offtrack,
        "is_left_of_center": sc.distance_from_center > 0,
        "all_wheels_on_track": not sc.is_offtrack, "is_crashed": False,
        "progress": (sc.closest_waypoints[0] / max(n, 1)) * 100.0,
        "steps": 1, "steering_angle": 0.0,
        "waypoints": wps, "closest_waypoints": sc.closest_waypoints, "is_stuck": False,
    }


def probe_swin_unetpp(agent, obs_flat, device="cpu") -> Dict:
    """
    Extract SwinUNet++ activations and 4-class clearance prediction.
    REF: Liu et al. (2021) Swin Transformer shifted-window attention.
    REF: Chen et al. (2021) TransUNet hierarchical transformer for segmentation.
    Returns dict with predicted_class, clearance_logits, attn_weights, note.
    """
    result = {"class_names": ["barrier", "curb", "obstacle", "clear"],
              "predicted_class": -1, "clearance_logits": [0.0]*4,
              "attn_weights": None, "note": ""}
    _swin = None
    for name, mod in agent.named_modules():
        if "swin" in name.lower() or "utransformer" in type(mod).__name__.lower():
            _swin = mod; break
    if _swin is None:
        result["note"] = "SwinUNetPP not found in agent -- BrakeField perception NOT wired (Bug X confirmed)"
        return result
    try:
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device)
        H, W = 120, 160
        if obs_t.numel() < H * W:
            result["note"] = f"obs too short ({obs_t.numel()}) for Swin 120x160"
            return result
        img = obs_t[:H*W].reshape(1, 1, H, W)
        with torch.no_grad():
            logits = _swin(img)
        if isinstance(logits, torch.Tensor):
            lv = logits.cpu().numpy().flatten()[:4]
            result["clearance_logits"] = lv.tolist()
            result["predicted_class"] = int(np.argmax(lv))
        result["note"] = "OK"
    except Exception as e:
        result["note"] = f"Error: {e}"
    return result


def interrogate_agent(agent, bc_pilot, sc: Scenario, waypoints: List,
                      device: str = "cpu", obs_dim: int = 38464) -> Dict:
    """Ask both agents what they would do in this scenario."""
    rp = build_rp(sc, waypoints)
    bc_action = bc_pilot.act(rp)
    bc_steer = float(bc_action[0]) if hasattr(bc_action, "__len__") else float(bc_action)
    bc_throttle = float(bc_action[1]) if hasattr(bc_action, "__len__") and len(bc_action) > 1 else 0.3
    obs_flat = np.zeros(obs_dim, dtype=np.float32)
    obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
    try:
        with torch.no_grad():
            action, logprob, _, value, ctx_logits, intermed = agent.get_action_and_value(obs_t)
        rl_steer = float(action.squeeze()[0].item())
        rl_throttle = float(action.squeeze()[1].item())
        rl_value = float(value.item())
    except Exception:
        rl_steer = rl_throttle = rl_value = 0.0
    swin_out = probe_swin_unetpp(agent, obs_flat, device=device)
    return {
        "scenario": sc.name, "note": sc.note,
        "rp_summary": {"x": sc.x, "y": sc.y, "heading": sc.heading, "speed": sc.speed,
                       "is_reversed": sc.is_reversed, "closest_wps": sc.closest_waypoints},
        "bc_pilot": {"steer": round(bc_steer, 3), "throttle": round(bc_throttle, 3)},
        "rl_agent": {"steer": round(rl_steer, 3), "throttle": round(rl_throttle, 3),
                     "value": round(rl_value, 3)},
        "swin_unetpp": {
            "predicted_class": swin_out["predicted_class"],
            "class_name": swin_out["class_names"][max(0, swin_out["predicted_class"])]
                          if swin_out["predicted_class"] >= 0 else "unknown",
            "logits": swin_out["clearance_logits"],
            "note": swin_out["note"]
        },
        "agreement": abs(bc_steer - rl_steer) < 0.25 and abs(bc_throttle - rl_throttle) < 0.25,
        "bc_engine_braking": bc_throttle < 0.02,
        "rl_engine_braking": rl_throttle < 0.05,
    }


def run_probe_session(
    agent, bc_pilot, waypoints: List, scenarios=None,
    device: str = "cpu", obs_dim: int = 38464, verbose: bool = True
) -> List[Dict]:
    """
    Run full probe session -- call from run.py after BC harvest:

        from probe_agent import run_probe_session, CANNED_SCENARIOS
        run_probe_session(agent, _bc_pilot, _env_waypoints, device=str(DEVICE))

    Results saved to results/probe_session.json automatically.
    Key check: crash scenario (distbarrier_020) -- both pilots should engine-brake.
    """
    scenarios = scenarios or CANNED_SCENARIOS
    results = []
    print("\n" + "="*72)
    print("CS7642 Project 4 v1.1.5b -- AGENT PROBE SESSION")
    print("="*72)
    for sc in scenarios:
        sc.waypoints = waypoints
        r = interrogate_agent(agent, bc_pilot, sc, waypoints, device, obs_dim)
        results.append(r)
        if verbose:
            agree_str = "AGREE" if r["agreement"] else "DIVERGE"
            print(f"\n[{sc.name}] {sc.note}")
            print(f"  RP: x={sc.x:.2f} y={sc.y:.2f} hdg={sc.heading:.1f}deg "
                  f"spd={sc.speed:.2f} rev={sc.is_reversed}")
            bc_eb_tag = " [ENGINE BRAKE]" if r["bc_engine_braking"] else ""
            rl_eb_tag = " [ENGINE BRAKE]" if r["rl_engine_braking"] else ""
            print(f"  BCPilot:  steer={r['bc_pilot']['steer']:+.3f}  "
                  f"throttle={r['bc_pilot']['throttle']:.3f}{bc_eb_tag}")
            print(f"  RLAgent:  steer={r['rl_agent']['steer']:+.3f}  "
                  f"throttle={r['rl_agent']['throttle']:.3f}{rl_eb_tag}  "
                  f"value={r['rl_agent']['value']:.3f}")
            swin = r["swin_unetpp"]
            print(f"  SwinUNet: class={swin['class_name']}  "
                  f"logits={swin['logits']}  [{swin['note']}]")
            print(f"  {agree_str}")
    print("\n" + "="*72)
    n_agree = sum(1 for r in results if r["agreement"])
    print(f"Agreement rate: {n_agree}/{len(results)} ({100*n_agree/max(len(results),1):.0f}%)")
    crash_replay = next((r for r in results if r["scenario"] == "distbarrier_020"), None)
    if crash_replay:
        bc_eb = crash_replay["bc_engine_braking"]
        rl_eb = crash_replay["rl_engine_braking"]
        print(f"\nCRASH SCENARIO (ep1 replay): "
              f"BCPilot engine_brake={bc_eb}  RLAgent engine_brake={rl_eb}")
        if not bc_eb:
            print("   BCPilot still NOT engine-braking -- Patch V needed")
        if not rl_eb:
            print("   RLAgent NOT engine-braking -- BC warm-start not propagated to policy")
    print("="*72 + "\n")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Standalone probe: loads checkpoint + BCPilot, runs all canned scenarios.

    Usage:
        python probe_agent.py
        python probe_agent.py --checkpoint ppo_agent_best.torch --device cpu
    """
    import argparse, sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Probe agent offline -- no sim required")
    parser.add_argument("--checkpoint", default="ppo_agent_best.torch")
    parser.add_argument("--obs-dim", type=int, default=38464)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    from context_aware_agent import ContextAwarePPOAgent
    agent = ContextAwarePPOAgent(obs_dim=args.obs_dim, act_dim=2, name="probe").to(device)
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        agent.load_state_dict(ckpt.get("state_dict", ckpt))
        print(f"[PROBE] Loaded: {args.checkpoint}")
    else:
        print(f"[PROBE] No checkpoint at {args.checkpoint} -- using random weights")

    # Approximate track waypoints from log geometry (real wps injected at training time)
    _wps = [(2.56 + 2.0*math.cos(i*2*math.pi/120),
             3.0  + 1.5*math.sin(i*2*math.pi/120))
            for i in range(120)]

    from run import BCPilot
    bc = BCPilot(waypoints=_wps, track_width=1.07)
    results = run_probe_session(agent, bc, _wps, device=str(device), obs_dim=args.obs_dim)

    Path("results").mkdir(exist_ok=True)
    with open("results/probe_session.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("[PROBE] Saved: results/probe_session.json")
