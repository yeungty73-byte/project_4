# A Gymnasium Wrapper for DeepRacer

## Setup
### Dependencies
- Docker or Apptainer.
- Python 3.10 or higher.
- Linux or Windows machine with Intel based CPU.

### Install
```bash
pip install -e ./
```

## Usage
### Start the simulation service
From the root of this repository, start the simulator container with the following command.
```bash
source scripts/start_deepracer.sh \
    [-C=MAX_CPU; default="3"] \
    [-M=MAX_MEMORY; default="6g"]

# example:
# source scripts/start_deepracer.sh -C "3" -M "6g"
```
You may also find other scripts under `scripts/` similarly useful to stop or restart the simulation service, etc.

To check if the container is rumming you can use the following commands.
```bash
docker ps -a            # if using Docker (local setup)
apptainer instance list # if using Apptainer (PACE ICE)
```
**Note** that the simulator is initialized by the `agent_params.json` and `environment_params.yaml` config files in the `configs/` directory. To change the simulation settings, restart it after changing these files under the `configs/` directory.

### Interact with the environment
```python
import gymnasium as gym
import deepracer_gym
from configs.reward_function import (
    # provide your own custom reward function if needed.
    # if not provided, gymnasium environment fetches it from configs.reward_function
    reward_function
)

env = gym.make(
    'deepracer-v0',
    reward_function = reward_function
)

observation, info = env.reset()

observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)

env.close()
```
The `terminated` flag is trigerred by the following in `info['episode_status']`.
```yaml
{
    "lap_complete": float,  # same as info['reward_params']['progress'] >= 100
    "crashed": boolean,     # same as info['reward_params']['is_crashed']
    "off_track": boolean,   # same as info['reward_params']['is_offtrack']
    "reversed": boolean,    # progress decreases for 15 consecutive steps. NOT accessible in info['reward_params'].
}
```
The `truncated` flag is trigerred by the following (not accessible in `info['reward_params']`).
```yaml
{
    "immobilized": boolean,     # move <= 0.0003 for 15 consecutive steps
    "time_up": boolean,         # 180 seconds max, or 100_000 steps max
}
```

For more details, see the [`gymnasium` API section](#gymnasium-API) below.

## Configuration
### Reward function
The `configs/reward_function.py` file defines the reward function which accepts varous [input parameters](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html). These are also accessible in `info` varibale of `gymnasium` as `info['reward_params']`.

To get motivation for designing reward functions for different types of races, please take a look at the [AWS DeepRacer reward function examples](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-examples.html).

### Agent parameters
The `configs/agent_params.json` configuration file defines the agent's action and observation space. The only settings of relevance are the following[^1]:.
| Parameter | Description |
|---|---|
| `action_space_type` | Can be `discrete` or `continuous`. |
| `action_space` | Defines the action space in terms of `speed` and `steering_angle`. See examples below. |
| `sensor` | Can be `FRONT_FACING_CAMERA` (a $160\times 120$ colored image), `STEREO_CAMERAS` (two $160\times 120$ greyscale images) and/or `LIDAR` ($64$ radial readings). Two camera sensors cannot be selected at once. Only LiDAR cannot be selected. For details please refer to the [AWS DeepRacer sensors page](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-choose-race-type.html). |

We provide two examples below:

#### Discrete actions with LiDAR + stereo camera 
```json
{
    "action_space": [
        {
            "steering_angle": 30,
            "speed": 0.6
        },
        {
            "steering_angle": 15,
            "speed": 0.6
        },
        {
            "steering_angle": 0,
            "speed": 0.6
        },
        {
            "steering_angle": -15,
            "speed": 0.6
        },
        {
            "steering_angle": -30,
            "speed": 0.6
        }
    ],
    "action_space_type": "discrete",
    "sensor": ["STEREO_CAMERAS", "LIDAR"],
    "neural_network": "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW",
    "version": "4"
}
```

#### Continuous actions with LiDAR + front-facing camera
```json
{
    "action_space": {
        "steering_angle": {
            "high": 30,
            "low": -30
        },
        "speed": {
            "high": 2,
            "low": 1
        }
    },
    "action_space_type": "continuous",
    "sensor": ["FRONT_FACING_CAMERA", "LIDAR"],
    "neural_network": "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW",
    "version": "4"
}
```

[^1]: Please donot change the `neural_network` and `version` variables.

### Environment parameters
The `configs/environment_params.yaml` configuration file is used to define the environment. See the [DeepRacer-for-cloud documentation](https://aws-deepracer-community.github.io/deepracer-for-cloud/reference.html) for the description of these parameters.
#### List of tracks
The following tracks can be selected by setting the `WORLD_NAME` parameter. You can find the layouts for these [here](https://github.com/aws-deepracer-community/deepracer-race-data/blob/main/raw_data/tracks/README.md).
```yaml
"Albert"
"AmericasGeneratedInclStart"
"Aragon"
"Austin"
"AWS_track"
"Belille"
"Bowtie_track"
"Canada_Race"
"Canada_Training"
"ChinaAlt_track"
"China_track"
"FS_June2020"
"hamption_open"
"hamption_pro"
"July_2020"
"jyllandsringen_open"
"jyllandsringen_pro"
"LGSWide"
"MexicoAlt_track"
"Mexico_track"
"Monaco"
"Monaco_building"
"New_YorkAlt_Track"
"New_York_Track"
"Oval_track"
"penbay_open"
"penbay_pro"
"reInvent2019_track"
"reInvent2019_wide"
"reInvent2019_wide_mirrored"
"reinvent_base"
"reinvent_base_jeremiah"
"reinvent_carpet"
"reinvent_concrete"
"reinvent_wood"
"Singapore"
"Singapore_building"
"Singapore_f1"
"Spain_track"
"Spain_track_f1"
"Straight_track"
"thunder_hill_open"
"thunder_hill_pro"
"Tokyo_Racing_track"
"Tokyo_Training_track"
"Virtual_Competition_1"
"Virtual_May19_Comp_track"
"Virtual_May19_Train_track"
"Vegas_track"
```

## `gymnasium` API
The DeepRacer environment follows the standard `gymnasium` API. Here are the key components:

### Environment Creation
```python
import gymnasium as gym
import deepracer_gym

env = gym.make('deepracer-v0')
```

### Observation Space
The observation space is a composotive space defined by a [`gymnasium.spaces.Dict`](https://gymnasium.farama.org/api/spaces/composite/) dictionary object containing the following keys and values depending on the sensors specified in `configs/agent_params.json`:
```python
{
    # two 8-bit greyscale (1 channel) images
    'STEREO_CAMERAS': Box(
        low=0, high=255, shape=(2, 120, 160)
    ),
    # one 8-bit colored (3 channel) image
    'FRONT_FACING_CAMERA': Box(
        low=0, high=255, shape=(3, 120, 160)
    ),
    'LIDAR': Box(
        low=0.15, high=float('inf'), shape=(64,)
    ),
}
```

### Action Space
Depending on the specification in `configs/agent_params.json`, the actions space can be the following. Note that for continuous action spaces, the input is normalized between -1 and 1 representing the `low` and `high` values of the respective quantity.

| Type | `gymnasium.spaces` object |
|---|---|
| Discrete | `Discrete(n)`, where `n` is 5 for [the example above](#discrete-actions-with-lidar--stereo-camera). |
| Continuous[^2] | `Box(-1, 1, shape=(n,))`, where `n` is 2 for [the example above](#continuous-actions-with-lidar--front-facing-camera). |

[^2]: For continuous action spaces, the `steering_angle` and `speed` occupy the 1st and 2nd indices of the 2D action vector/list as `[normalized_steering_angle, normalized_speed]`.

### Environment Step
```python
observation, reward, terminated, truncated, info = env.step(action)
```

The step function returns:
- `observation`: Dictionary of sensor readings
- `reward`: Float value from reward function
- `terminated`: Boolean indicating episode end due to:
  - Crash
  - Off-track
  - Reversed direction
  - Lap completion
- `truncated`: Boolean indicating episode end due to:
  - Time/Step limit
  - Immobilization
- `info`: Dictionary containing:
  - `reward_params`: Parameters used in reward calculation
  - `episode_status`: Current episode state

### Environment Reset
```python
observation, info = env.reset()
```

### Environment Close
```python
env.close()
```

### Rendering
```python
# Returns numpy array
env = gym.make('deepracer-v0', render_mode='rgb_array')
```

## Limitations, Problems and Troubleshooting
- Due to simulation limitations, the `deepracer-v0` environment **does NOT support** [environment vectorization](https://gymnasium.farama.org/api/vector/). This includes `gymnasium.vector.SyncVectorEnv`, which has to be run with a maximum of `num_envs=1`.
- If Docker does not work for you without `sudo`, please follow the instructions in [`README.md`](https://github.gatech.edu/rldm/P4_deepracer/blob/main/SETUP.md) to add it to `sudo` group.
- Please note that the first run of `scripts/start_deepracer.sh` or `scripts/restart_deepracer.sh` can be quite slow. This is because the simulator base image is downloaded (~12 GBs), built with P4 specific patches before being started. But this should be a one-time process and subsequent runs should be relatively quicker.
- We have tried to deligently test the simulator and various configurations for this project. However, it is entirely possible that some edge-cases may have gone overlooked due to limited time-constraints. Should you encounter such an edge case, please feel free to hop into an OH or reach out to a TA to get it fixed ASAP.
