#!/usr/bin/env python3
"""
apply_v116b_patches_v2.py  --  DIRECT OVERWRITE version
Reads run.py, applies 3 surgical patches, writes run.py in-place.
Run from the project root:  python apply_v116b_patches_v2.py
"""
import shutil, sys, re
from pathlib import Path

SRC = Path("run.py")
BAK = Path("run.py.bak116b_v2")

# ── helpers ──────────────────────────────────────────────────────────────────

def must_replace(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        print(f"  FAIL  {label}: anchor not found – patch aborted")
        sys.exit(1)
    count = src.count(old)
    if count > 1:
        print(f"  WARN  {label}: anchor appears {count}×, replacing first only")
    result = src.replace(old, new, 1)
    print(f"  OK    {label}")
    return result

# ── load ─────────────────────────────────────────────────────────────────────

src = SRC.read_text(encoding="utf-8")
shutil.copy(SRC, BAK)
print(f"Backup → {BAK}")

# ════════════════════════════════════════════════════════════════════════════
# PATCH 1 – SPAWN-BLEED
#   Anchor: the SPAWN log line that says "-- sim always spawns at 4.0 m/s"
#   It appears in the INNER episode-reset block (mid-training loop).
#   We inject a bleed loop right after the spawn-logging block, before
#   the existing shaper.episode_start call.
# ════════════════════════════════════════════════════════════════════════════

PATCH1_OLD = (
    "resetinforp = info.get(\"reward_params\", {}) if isinstance(info, dict) else {}\n"
    "        spawnhdg = float(resetinforp.get(\"heading\", 0.0))\n"
    "        if bool(resetinforp.get(\"is_reversed\", False)):\n"
    "            logger.info(f\"SPAWN isReversed=True ep={episodecount} heading={spawnhdg:.1f}\")\n"
    "        # -- sim always spawns at 4.0 m/s"
)

PATCH1_NEW = (
    "resetinforp = info.get(\"reward_params\", {}) if isinstance(info, dict) else {}\n"
    "        spawnhdg = float(resetinforp.get(\"heading\", 0.0))\n"
    "        if bool(resetinforp.get(\"is_reversed\", False)):\n"
    "            logger.info(f\"SPAWN isReversed=True ep={episodecount} heading={spawnhdg:.1f}\")\n"
    "        # PATCH-BLEED v1.1.6b ─ neutralise Gazebo's 4.0 m/s spawn injection\n"
    "        # REF: AWS DeepRacer sim hardcodes spawn_speed=4.0 m/s via deepracer_drive plugin.\n"
    "        # Stopping distance at 4.0 m/s (a=17.5 m/s²) ≈ 0.46 m > 0.42 m track clearance.\n"
    "        # Physics guarantees a crash at step 1 regardless of reward signal until bleed.\n"
    "        # REF: Levine et al. (2020) – BC demonstrations must share operating distribution.\n"
    "        _bleed_max_steps = 30\n"
    "        _bleed_steps = 0\n"
    "        _bleed_spawn_spd = float(resetinforp.get(\"speed\", 4.0))\n"
    "        _bleed_final_spd = _bleed_spawn_spd\n"
    "        _bleed_obs = observation  # observation from env.reset above\n"
    "        _bleed_info = info\n"
    "        _bleed_action_zero = (\n"
    "            np.zeros(env.action_space.shape, dtype=np.float32)\n"
    "            if hasattr(env.action_space, 'shape') and env.action_space.shape\n"
    "            else np.array([0], dtype=np.int64)\n"
    "        )\n"
    "        # For continuous Box([-1,1],[-1,1]): steer=0, throttle=-1 (min) → engine off\n"
    "        if hasattr(env.action_space, 'low') and env.action_space.shape and env.action_space.shape[0] >= 2:\n"
    "            _bleed_action_zero = np.array([0.0, float(env.action_space.low[1])], dtype=np.float32)\n"
    "        for _bs in range(_bleed_max_steps):\n"
    "            _bleed_obs, _bleed_rew, _bleed_term, _bleed_trunc, _bleed_info = env.step(_bleed_action_zero)\n"
    "            _bleed_rp = _bleed_info.get(\"reward_params\", {}) if isinstance(_bleed_info, dict) else {}\n"
    "            _bleed_final_spd = float(_bleed_rp.get(\"speed\", _bleed_final_spd))\n"
    "            _bleed_steps += 1\n"
    "            if _bleed_final_spd < 0.30:\n"
    "                break\n"
    "            if _bleed_term or _bleed_trunc:\n"
    "                # rare: crash during bleed — accept and log\n"
    "                break\n"
    "        logger.info(\n"
    "            f\"SPAWN_BLEED ep={episodecount} \"\n"
    "            f\"spawn_spd={_bleed_spawn_spd:.2f} \"\n"
    "            f\"bleed_steps={_bleed_steps} \"\n"
    "            f\"bleed_final_spd={_bleed_final_spd:.3f} \"\n"
    "            f\"is_reversed={bool(resetinforp.get('is_reversed', False))}\"\n"
    "        )\n"
    "        # Replace observation/info with post-bleed state so the agent starts from rest\n"
    "        observation = _bleed_obs\n"
    "        info = _bleed_info\n"
    "        resetinforp = info.get(\"reward_params\", {}) if isinstance(info, dict) else {}\n"
    "        # -- sim always spawns at 4.0 m/s"
)

src = must_replace(src, PATCH1_OLD, PATCH1_NEW, "PATCH-BLEED (mid-training episode reset)")


# ════════════════════════════════════════════════════════════════════════════
# PATCH 2 – SPAWN-BLEED in harvesthtmpilots (BCPilot / HTMPilot loop)
#   Anchor: the `obs, info = env.reset` line inside harvesthtmpilots,
#   followed by `rp = info.get("reward_params", ...)` and bcprogresscache reset.
# ════════════════════════════════════════════════════════════════════════════

PATCH2_OLD = (
    "    for ep in range(max(n_episodes * 3, n_episodes)):\n"
    "        obs, info = env.reset()\n"
    "        rp = info.get(\"reward_params\", {}) if isinstance(info, dict) else {}\n"
    "        bcprogresscache"
)

PATCH2_NEW = (
    "    for ep in range(max(n_episodes * 3, n_episodes)):\n"
    "        obs, info = env.reset()\n"
    "        # PATCH-BCPILOT-IDLE v1.1.6b: bleed spawn speed before BCPilot acts\n"
    "        # BCPilot demonstrations at 4.0 m/s spawn are out-of-distribution for\n"
    "        # the policy's operating regime. Bleed to near-rest first.\n"
    "        # REF: Levine et al. (2020) Offline RL tutorial – demo distribution must match.\n"
    "        _bp_bleed_rp = info.get(\"reward_params\", {}) if isinstance(info, dict) else {}\n"
    "        _bp_spawn_spd = float(_bp_bleed_rp.get(\"speed\", 4.0))\n"
    "        _bp_bleed_final = _bp_spawn_spd\n"
    "        if _bp_spawn_spd > 0.30:\n"
    "            _bp_zero = (\n"
    "                np.array([0.0, float(env.action_space.low[1])], dtype=np.float32)\n"
    "                if hasattr(env.action_space, 'low') and env.action_space.shape and env.action_space.shape[0] >= 2\n"
    "                else np.zeros(env.action_space.shape if hasattr(env.action_space,'shape') and env.action_space.shape else (1,), dtype=np.float32)\n"
    "            )\n"
    "            for _bpb in range(30):\n"
    "                obs, _bpr, _bpt, _bptu, info = env.step(_bp_zero)\n"
    "                _bp_bleed_rp = info.get(\"reward_params\", {}) if isinstance(info, dict) else {}\n"
    "                _bp_bleed_final = float(_bp_bleed_rp.get(\"speed\", _bp_bleed_final))\n"
    "                if _bp_bleed_final < 0.30 or _bpt or _bptu:\n"
    "                    break\n"
    "            logger.info(f\"BCPilot SPAWN_BLEED ep={ep} spawn={_bp_spawn_spd:.2f} final={_bp_bleed_final:.3f}\")\n"
    "        rp = info.get(\"reward_params\", {}) if isinstance(info, dict) else {}\n"
    "        bcprogresscache"
)

src = must_replace(src, PATCH2_OLD, PATCH2_NEW, "PATCH-BCPILOT-IDLE (harvesthtmpilots bleed)")


# ════════════════════════════════════════════════════════════════════════════
# PATCH 3 – SPEEDCAP inside process_action
#   Current code:
#     try:
#         if shaper in globals() and shaper is not None:
#             tpa_head = shaper.processactionscale
#             a[1] = min(a[1], float(tpa_head))
#     except Exception:
#         pass
#   We replace this with a version that ALSO applies the phase speed cap.
# ════════════════════════════════════════════════════════════════════════════

PATCH3_OLD = (
    "    try:\n"
    "        if shaper in globals() and shaper is not None:\n"
    "            tpa_head = shaper.processactionscale\n"
    "            a[1] = min(a[1], float(tpa_head))\n"
    "    except Exception:\n"
    "        pass\n"
    "    return np.clip(a, action_space.low, action_space.high).astype(np.float32)"
)

PATCH3_NEW = (
    "    # PATCH-SPEEDCAP v1.1.6b: phase-gated throttle ceiling\n"
    "    # Phase 0 (post-bleed bootstrap): 40% ≈ 1.6 m/s  Phase 1: 65%  Phase 2: uncapped\n"
    "    # Applied AFTER bleed so agent cannot re-accelerate before finding the racing line.\n"
    "    # REF: Bengio et al. (2009) ICML – curriculum from low action complexity.\n"
    "    try:\n"
    "        _phase_cap = get_phase_speed_cap(shaper if 'shaper' in globals() else None)\n"
    "        a[1] = min(a[1], float(_phase_cap))\n"
    "        if 'shaper' in globals() and shaper is not None:\n"
    "            tpa_head = shaper.processactionscale\n"
    "            a[1] = min(a[1], float(tpa_head))\n"
    "        logger.debug(\n"
    "            f\"process_action throttle capped to {_phase_cap:.2f} \"\n"
    "            f\"(phase={getattr(shaper,'currentphase',0) if 'shaper' in globals() and shaper is not None else 0})\"\n"
    "        )\n"
    "    except Exception:\n"
    "        pass\n"
    "    return np.clip(a, action_space.low, action_space.high).astype(np.float32)"
)

src = must_replace(src, PATCH3_OLD, PATCH3_NEW, "PATCH-SPEEDCAP (process_action throttle cap)")

# ── write ─────────────────────────────────────────────────────────────────────

SRC.write_text(src, encoding="utf-8")
print("\n=== v1.1.6b Patches Applied (v2) ===")
print(" ✓ PATCH-BLEED         — mid-training episode reset neutralises 4.0 m/s spawn")
print(" ✓ PATCH-BCPILOT-IDLE  — harvesthtmpilots BCPilot bleed")
print(" ✓ PATCH-SPEEDCAP      — process_action phase-gated throttle ceiling")
print()
print("Expected new log signatures:")
print("  SPAWN_BLEED ep=N spawn_spd=3.xx bleed_steps=5 bleed_final_spd=0.2x is_reversed=False")
print("  BCPilot SPAWN_BLEED ep=0 spawn=3.xx final=0.2x")
print("  process_action throttle capped to 0.40 (phase=0)")
print()
print("Run: python run.py")
