# P4 DeepRacer — Research-Driven PPO for AWS DeepRacer

**CS 7642 Reinforcement Learning** | Georgia Institute of Technology

A PPO-based autonomous racing agent for the AWS DeepRacer simulator,
incorporating insights from autonomous racing and reward shaping literature.
Features federated multi-worker training across PACE ICE cluster nodes.

## Key Equations

### Harmonized Track Progress (concave amplifier)

$$p^{\star} = 1 - \left(1 - \frac{p_{\text{raw}}}{100}\right)^2$$

Lifts episode-level variance above the BSTS Kalman observation noise floor (SNR ≥ 1.5 at Phase-A entry).

### Four-Channel Potential-Based Reward (Ng-invariant)

$$\tilde{r}(s,a,s') = r(s,a,s') + \gamma\,\Phi(s') - \Phi(s)$$

$$\Phi = w_p\,\Phi_{\text{prog}} + w_b\,\Phi_{\text{brake}} + w_r\,\Phi_{\text{race}} + w_h\,\Phi_{\text{head}}$$

Weights $w_t$ anneal across curriculum phases A→B→C (see `adaptive_reward_shaper.py`).

### Brake-Field Speed Cap

$$v_{\text{cap}}(\kappa) = \sqrt{\frac{\mu g}{|\kappa|}}$$

Curvature-limited velocity ceiling from quasi-static tire dynamics.

### ICM Intrinsic Reward (Pathak et al., 2017)

$$r^i_t = \eta \cdot \frac{1}{2}\,\left\|\phi(s_{t+1}) - \hat{\phi}(s_{t+1} \mid s_t, a_t)\right\|^2 \cdot w\!\left(\text{seg}(s_t)\right)$$

Forward-model prediction error in encoder space, scaled by crash-density hotspot weight $w \in [1, 3]$ from `failure_analysis.py`.

### PPO Clipped Surrogate Objective

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\mathrm{clip}(r_t(\theta),\,1{-}\epsilon,\,1{+}\epsilon)\hat{A}_t\right)\right]$$

---

## v1.0.7 — Waypoint Tracking + DeepRacer Termination Conditions

### Key Changes
- **Waypoint-based progress tracking**: tracks max waypoint reached per episode
  instead of just progress percentage, giving precise track position at termination
- **Proper DeepRacer termination mapping**: crashed, offtrack, reversed,
  lap_completed, immobilized, time_up (per DeepRacer sim spec)
- **TD3-only ablation** (v1.0.6): PPO update epochs=0, only TD3 twin-critic updates
- **BSTS Kalman filter**: online trend/seasonal/regression decomposition for
  reward weight annealing

## Repository Structure

```
project_4/
  run.py                    # Main training loop (PPO + TD3 + BSTS annealing)
  agents.py                 # PPOAgent, RandomAgent
  context_aware_agent.py    # ContextAwarePPOAgent (straight/curve/obstacle)
  td3_sac_ensemble.py       # TD3+SAC twin-critic ensemble
  icm.py                    # Intrinsic Curiosity Module (Pathak et al. 2017)
  corner_analysis.py        # Curvature, braking, racing line rewards
  race_line_engine.py       # Multi-strategy racing line computation
  brake_field.py            # Continuous brake-line vector field
  utils.py                  # Device, seed, BSTSTracker, ReplayBuffer, etc.
  analyze_logs.py           # BSTSKalmanFilter, intermediary/success metrics
  bsts_seasonal.py          # BSTS seasonal tracker (lap segments)
  live_metrics.py           # Live metric analysis
  live_dashboard.py         # Console summary + PNG dashboard
  failure_analysis.py       # FailurePointSampler
  stuck_tracker.py          # Stuck detection + early termination
  federated_pool.py         # Federated checkpoint pooling
  references.bib            # Academic references
  configs/
    hyper_params.yaml       # PPO/training hyperparameters
    environment_params.yaml # DeepRacer environment config
  scripts/
    start_deepracer.sh      # Start DeepRacer sim (Singularity)
    stop_deepracer.sh       # Stop sim
    restart_deepracer.sh    # Restart sim
    check_status.sh         # Check sim status
    cleanup_deepracer.sh    # Clean up processes
    run_part2.sh            # Run part 2
    run_vegas.sh            # Run Vegas track variant
```

## Usage

```bash
# 1. Start DeepRacer simulation environment
bash scripts/start_deepracer.sh

# 2. Activate conda environment
conda activate deepracer

# 3. Run training
python run.py
```

## Metrics Tracked

- **Waypoint progression**: max waypoint / total waypoints per episode
- **Termination reasons**: crashed, offtrack, reversed, lap_completed, immobilized, time_up
- **BSTS intermediary**: curvature compliance, perpendicular velocity, brake zone adherence
- **Success metrics**: lap completion %, reward per step, off-track rate, crash rate

## Apptainer (Singularity) sim setup — troubleshooting

If `start_deepracer.sh` exits with:
- `FATAL: Image file already exists: "deepracer_base.sif" - will not overwrite`
- `FATAL: Unable to build from deepracer.def: … no such file or directory`
- `FATAL: While checking container encryption: could not open image .../project_4/deepracer.sif`

then the script is falling through the pull/build paths and trying to start an instance from a non-existent local `deepracer.sif`. The working images normally live one directory up. Fix:

```bash
cd ~/scratch/P4_deepracer/project_4
rm -f deepracer_base.sif                     # remove stale partial
ln -sf ../deepracer.sif deepracer.sif         # symlink working derived image
ln -sf ../deepracer_base.sif deepracer_base.sif
bash start_deepracer.sh                       # pull/build FATALs are benign; instance run will succeed
```

Verify:
```bash
apptainer instance list                        # should show: deepracer <PID> <IP> .../project_4/deepracer.sif
apptainer exec instance://deepracer pgrep -af gzserver   # Gazebo process running
apptainer exec instance://deepracer pgrep -af rollout    # markov.rollout_worker running
```

**Important:** `run.py` in this folder is a standalone PPO loop that does not connect to the DeepRacer sim. Real training happens via `markov.rollout_worker` inside the container (started automatically by `start_deepracer.sh`). The `[GUARD]` log line introduced in v201 will report `JSONL is EMPTY` if `run.py` is mistakenly run on a node without the sim wiring.

## v202 — Gym bridge port discovery (Apptainer)

The ZMQ client used to be hardcoded to `tcp://127.0.0.1:8888`. Under Apptainer there is
no `-p 8888:8888` port-map (unlike Docker), so the sim's ZMQ REP socket binds on a
different ephemeral port in the host network namespace and the Python client silently
failed — PPO ran for hours on an environment whose `terminated` never went True.

### Fix
1. `packages/deepracer_gym/zmq_client.py` now reads `GYM_HOST`/`GYM_PORT` from env.
2. `env_for_client.sh` exports the user-hashed ports (same scheme as `start_deepracer.sh`).
3. `run.py` (v202) does a TCP preflight on `$GYM_HOST:$GYM_PORT` and aborts with a
   clear message if unreachable — no more 100k stub-episode runs.

### Workflow
```
bash start_deepracer.sh            # bring up Apptainer sim (one time per session)
ss -ltnp | grep rollout_worker     # find actual ZMQ REP port of markov.rollout_worker
export GYM_PORT=<that port>        # or: source ./env_for_client.sh
python run.py --environment deepracer-v0 --results_dir results/live
```

If the preflight aborts, the rollout_worker is not listening. Check `pgrep -fa rollout_worker`
and the sim's start log under `~/scratch/P4_deepracer/apptainer/`.

### v202b — auto-discover GYM_PORT
`env_for_client.sh` now queries `pgrep markov.rollout_worker` + `ss -ltnp` to find the actual
ZMQ REP port the running rollout_worker is listening on, and exports it as `GYM_PORT`.
Falls back to the hashed port if no rollout_worker is found.

Also fixed `run.py` preflight `SystemExit` path (prior `import sys` inside the conditional
shadowed the module-level `sys` reference via Python's local-name binding rules).

### Known remaining issue — ZMQ protocol mismatch
Even after a successful TCP preflight and `env.reset() succeeded on attempt 1`, the JSONL
stays 0 bytes: `terminated` never flips True. Direct probe:
`socket.REQ -> msgpack.packb({'ready':1}) -> recv()` times out. The `markov.rollout_worker`
inside the Apptainer image expects a different message schema than what
`packages/deepracer_gym/zmq_client.py` sends. Next step: inspect
`/opt/amazon/markov/rollout_worker*.py` inside the container to align request keys.
