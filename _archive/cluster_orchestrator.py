#!/usr/bin/env python3
"""Cluster Orchestrator v1.0 - Federated Interleaved Multi-Variant Training

Launches 5 training variations across N cluster nodes with permuted orders.
Variations: 3 time-trial tracks + 1 head-to-bot + 1 obstacle-avoidance
Each cluster trains ALL variations but in a different permuted order.
Federated weight pooling via federated_pool.py after each variation phase.
"""
import os, sys, json, time, shutil, hashlib, itertools, subprocess, signal
import yaml
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "configs"
FEDERATED_DIR = BASE_DIR / "federated_pool"
RESULTS_DIR = BASE_DIR / "results"
DASHBOARD_DIR = RESULTS_DIR / "dashboard"
CRASH_LOG = RESULTS_DIR / "crash_hotspots.jsonl"

# 5 variations: label -> (config_yaml, cli_flags)
VARIATIONS = [
    ("tt_wide",   "environment_params_tt_reinvent.yaml", ["--track", "reInvent2019_wide"]),
    ("tt_medium", "environment_params_tt_vegas.yaml",    ["--track", "reInvent2019_track"]),
    ("tt_narrow", "environment_params_tt_bowtie.yaml",   ["--track", "reInvent2019_track_narrow"]),
    ("h2b",       "environment_params_h2b.yaml",         ["--track", "reInvent2019_wide", "--h2b"]),
    ("obstacle",  "environment_params_oa.yaml",          ["--track", "reInvent2019_wide", "--obstacle"]),
]

# Pre-computed permutations for up to 5 clusters (Latin-square style)
# Each cluster sees all 5 variations but in different order
PERMUTATION_SEEDS = [
    [0, 1, 2, 3, 4],  # Cluster 0: wide, medium, narrow, h2b, obstacle
    [3, 0, 4, 1, 2],  # Cluster 1: h2b, wide, obstacle, medium, narrow
    [4, 2, 0, 1, 3],  # Cluster 2: obstacle, narrow, wide, medium, h2b
    [1, 3, 2, 4, 0],  # Cluster 3: medium, h2b, narrow, obstacle, wide
    [2, 4, 3, 0, 1],  # Cluster 4: narrow, obstacle, h2b, wide, medium
]

STEPS_PER_PHASE = 20000  # timesteps per variation phase before switching


def get_cluster_id():
    """Derive unique cluster ID from hostname."""
    import socket
    hostname = socket.gethostname()
    h = int(hashlib.md5(hostname.encode()).hexdigest()[:8], 16)
    return h % len(PERMUTATION_SEEDS)


def get_schedule(cluster_id: int) -> list:
    """Return the variation schedule for this cluster."""
    perm = PERMUTATION_SEEDS[cluster_id % len(PERMUTATION_SEEDS)]
    return [VARIATIONS[i] for i in perm]


def swap_env_config(variant_yaml: str):
    """Copy the variant config to the active environment_params.yaml."""
    src = CONFIG_DIR / variant_yaml
    dst = BASE_DIR / "configs" / "environment_params.yaml"
    shutil.copy2(src, dst)
    print(f"[ORCHESTRATOR] Swapped config to {variant_yaml}")


def run_phase(label: str, yaml_file: str, cli_flags: list, phase_steps: int,
              checkpoint_path: str = None) -> subprocess.Popen:
    """Launch a single training phase."""
    swap_env_config(yaml_file)
    PYTHON = os.environ.get("DEEPRACER_PY") or sys.executable
    cmd = [PYTHON, "-u", "run.py"] + cli_flags
    cmd += ["--total_timesteps", str(phase_steps)]
    if checkpoint_path and os.path.exists(checkpoint_path):
        cmd += ["--checkpoint", checkpoint_path]
    
    log_path = RESULTS_DIR / f"phase_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(f"[ORCHESTRATOR] Starting phase: {label} ({phase_steps} steps)")
    print(f"[ORCHESTRATOR] CMD: {' '.join(cmd)}")
    print(f"[ORCHESTRATOR] Log: {log_path}")
    
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                            cwd=str(BASE_DIR))
    return proc, log_fh


def find_latest_checkpoint() -> str:
    """Find the most recent model checkpoint."""
    ckpt_dir = RESULTS_DIR
    candidates = list(ckpt_dir.glob("**/ctx_ppo_agent.pt"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def federated_merge():
    """Trigger federated weight averaging across all cluster checkpoints."""
    try:
        from federated_pool import FederatedPool
        pool = FederatedPool(pool_dir=str(FEDERATED_DIR), max_pool_size=5)
        pool.load_manifest()
        ckpt = find_latest_checkpoint()
        if ckpt:
            pool.submit(ckpt)
            merged = pool.fedavg()
            if merged:
                print(f"[ORCHESTRATOR] Federated merge complete: {merged}")
                return merged
    except Exception as e:
        print(f"[ORCHESTRATOR] Federated merge failed: {e}")
    return None


def main():
    cluster_id = get_cluster_id()
    schedule = get_schedule(cluster_id)
    
    print(f"="*60)
    print(f"[ORCHESTRATOR] Cluster ID: {cluster_id}")
    print(f"[ORCHESTRATOR] Schedule: {[s[0] for s in schedule]}")
    print(f"[ORCHESTRATOR] Steps per phase: {STEPS_PER_PHASE}")
    print(f"="*60)
    
    FEDERATED_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write schedule manifest
    manifest = {
        "cluster_id": cluster_id,
        "schedule": [s[0] for s in schedule],
        "steps_per_phase": STEPS_PER_PHASE,
        "start_time": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / f"schedule_cluster{cluster_id}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    checkpoint = find_latest_checkpoint()
    
    for phase_idx, (label, yaml_file, cli_flags) in enumerate(schedule):
        print(f"{'='*60}")
        print(f"[ORCHESTRATOR] Phase {phase_idx+1}/{len(schedule)}: {label}")
        print(f"{'='*60}")
        
        proc, log_fh = run_phase(label, yaml_file, cli_flags,
                                  STEPS_PER_PHASE, checkpoint)
        proc.wait()
        log_fh.close()
        
        if proc.returncode != 0:
            print(f"[ORCHESTRATOR] WARNING: Phase {label} exited with code {proc.returncode}")
        
        checkpoint = find_latest_checkpoint()
        print(f"[ORCHESTRATOR] Phase {label} complete. Checkpoint: {checkpoint}")
        
        # Federated merge after each phase
        merged = federated_merge()
        if merged:
            checkpoint = merged
            print(f"[ORCHESTRATOR] Using federated checkpoint: {checkpoint}")
    
    print(f"{'='*60}")
    print(f"[ORCHESTRATOR] All phases complete for cluster {cluster_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
