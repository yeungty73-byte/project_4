"""
configs/reward_function.py
Default reward function for DeepRacer P4 — passed as DEFAULT_REWARD_FUNCTION
into DeepracerGymEnv.  This fires INSIDE env.step() via the gym adapter.

IMPORTANT (v1.1.2 audit):
  run.py computes its own shaped reward independently (AnnealingScheduler +
  compute_reward()).  To avoid a dual-reward signal, make_environment() in
  utils.py now passes reward_function=_identity_reward so that env.step()
  returns reward=1.0 (neutral) and run.py's shaped reward is the ONLY signal.

  This file is kept as reference / fallback for standalone eval runs that do
  NOT go through run.py (e.g. jupyter notebooks, demo()).

Params reference:
  https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-reference.html
"""


def reward_function(params):
    """Unified reward: TT / OA / H2B compatible."""
    track_width     = params["track_width"]
    distance_center = params["distance_from_center"]
    all_wheels_on   = params["all_wheels_on_track"]
    speed           = params["speed"]
    steering_abs    = abs(params["steering_angle"])
    progress        = params["progress"]
    steps           = params["steps"]
    is_offtrack     = params["is_offtrack"]
    is_crashed      = params["is_crashed"]

    if is_offtrack or is_crashed:
        return 1e-3

    # Centerline (Gaussian bands)
    m1, m2, m3 = 0.1 * track_width, 0.25 * track_width, 0.5 * track_width
    if distance_center <= m1:
        center_reward = 1.0
    elif distance_center <= m2:
        center_reward = 0.5
    elif distance_center <= m3:
        center_reward = 0.1
    else:
        center_reward = 1e-3

    # Speed
    if speed < 1.0:
        speed_reward = 0.5
    elif speed < 3.0:
        speed_reward = 0.5 + 0.5 * (speed - 1.0) / 2.0
    else:
        speed_reward = 1.0

    # Steering penalty
    steer_penalty = 0.8 if steering_abs > 15.0 else 1.0

    # Progress
    progress_bonus = (progress / 100.0) * 2.0 if steps > 0 else 0.0
    if progress >= 99.9:
        progress_bonus += 10.0

    # Heading alignment
    heading_reward = max(1.0 - (steering_abs / 30.0), 0.0)

    # Obstacle/bot proximity
    obstacle_bonus = 0.0
    if params.get("objects_distance"):
        closest_dist = min(params["objects_distance"])
        if closest_dist < 0.5:
            obstacle_bonus = -0.5
        elif closest_dist < 1.0:
            obstacle_bonus = 0.0
        else:
            obstacle_bonus = 0.2

    reward = (
        center_reward * 1.5
        + speed_reward * 1.0
        + steer_penalty * 0.5
        + heading_reward * 0.5
        + progress_bonus
        + obstacle_bonus
    )
    return float(max(reward, 1e-3))


def _identity_reward(params):  # noqa: N802
    """Neutral pass-through — used by make_environment() in training mode.
    run.py computes its own shaped reward; the gym env just needs to return
    something non-zero so RecordEpisodeStatistics doesn't NaN."""
    return 1.0
