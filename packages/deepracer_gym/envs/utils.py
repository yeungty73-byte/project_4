import json
import hashlib
import platform
import numpy as np
from loguru import logger
from gymnasium import spaces


LIDAR_SHAPE: tuple[int, ...]=(64,)
CAMERA_SHAPE: tuple[int, ...]=(120, 160)                        # H x W
STEREO_CAMERA_SHAPE: tuple[int, ...]=(2,)+CAMERA_SHAPE          # C x H x W
FRONT_FACING_CAMERA_SHAPE: tuple[int, ...]=(3,)+CAMERA_SHAPE    # C x H x W
SENSOR_SPACE: dict[str, spaces.Box]={
    'LIDAR': spaces.Box(
        low=0.15, high=float('inf'), shape=LIDAR_SHAPE, dtype=np.float64
    ),
    'STEREO_CAMERAS': spaces.Box(
        low=0, high=255, shape=STEREO_CAMERA_SHAPE, dtype=np.uint8
    ),
    'FRONT_FACING_CAMERA': spaces.Box(
        low=0, high=255, shape=FRONT_FACING_CAMERA_SHAPE, dtype=np.uint8
    ),
    # TODO: Look into implementing these!
    'SECTOR_LIDAR': None,
    'LEFT_CAMERA': None
}
AGENT_PARAMS_PATH: str='configs/agent_params.json'


def validate_action_space_config(config: dict, action_space_type: str):
    # make sure action_space defined correctly
    if action_space_type == 'discrete':
        assert isinstance(config['action_space'], list), \
                f'action_space_type is discrete but action_space is not a list.'
        
        for action in config['action_space']:
            assert isinstance(action, dict), \
                f'All actions should be defined as dictionaries in action_space.'

            assert all(
                (
                    key in action
                    and
                    isinstance(action[key], (int, float))
                ) for key in ('steering_angle', 'speed')
            ), f'steering_angle or speed incorrectly defined for action in action_space.'
    elif action_space_type == 'continuous':
        assert isinstance(config['action_space'], dict), \
                f'action_space_type is continuous but action_space is not a dictionary.'
        
        assert all(
            (
                key in config['action_space']
                and
                isinstance(config['action_space'][key], dict)
            ) for key in ('steering_angle', 'speed')
        ), f'steering_angle or speed incorrectly defined for action in action_space.'
        
        for action, bounds in config['action_space'].items():
            if not isinstance(bounds, dict):
                continue
            assert all(
                (
                    key in bounds
                    and
                    isinstance(bounds[key], (int, float))
                ) for key in ('low', 'high')
            ), f'Bounds incorrectly defined for {action} in action_space.'

            assert bounds['low'] < bounds['high'], \
                f'Lower bound should be lower than upper bound for {action} in action_space.'
    else:
        raise ValueError(
            f'action_space_type can only be continuous or discrete.'
        )
    
    return True


def action_space_type(config: dict):    
    if 'action_space_type' in config:
        if config['action_space_type'] not in ('discrete', 'continuous'):
            raise ValueError(
                f'Incorrectly defined action_space_type in config file.'
            )
        space_type = config['action_space_type']
    else:
        if isinstance(config['action_space'], list):
            # assuming discrete
            space_type = 'discrete'
        elif isinstance(config['action_space'], dict):
            # assuming continuous
            space_type = 'continuous'
        else:
            raise ValueError(
                f'Incorrectly defined action_space in config file.'
            )
    return space_type


def make_action_space(config_path: str=AGENT_PARAMS_PATH):
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    assert 'action_space' in config, \
        f'Action space not defined in config file {config_path}.'
    
    try:
        space_type = action_space_type(config)
    except Exception as e:
        logger.error(
            f'Incorrectly defined action_space in config file {config_path}.'
        )
        raise e
    
    assert validate_action_space_config(config, space_type)

    if space_type == 'discrete':
        size = len(config['action_space'])
        action_space = spaces.Discrete(
            size
        )
    elif space_type == 'continuous':
        action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float64
        )
    else:
        raise ValueError(
            f'Space type can only be discrete or continuous for actions. Got {space_type} instead.'
        )
    
    return action_space, config['action_space']


def make_observation_space(config_path: str=AGENT_PARAMS_PATH):
    with open(config_path, 'r') as file:
        config = json.load(file)
    sensors: list[str]=config['sensor']
    
    for sensor in sensors:
        assert (
            (sensor in SENSOR_SPACE) 
            and 
            (SENSOR_SPACE[sensor] is not None)
        ), f'Sensor {sensor} not supported!'

    return spaces.Dict({
        sensor: SENSOR_SPACE[sensor] for sensor in sensors
    }), sensors


def num_channels(measurement: np.ndarray):
    dimensions = len(measurement.shape)
    if dimensions == 2:
        channels = 1
    elif dimensions == 3:
        channels = measurement.shape[0]
    return channels


def string_to_port(string):
    hash_bytes = hashlib.sha256(string.encode()).digest()
    hash_int = int.from_bytes(
        hash_bytes[:4], byteorder='big'     # Only first 4 bytes
    )
    port = 1024 + (hash_int % (32767 - 1024 + 1))
    return int(port)


def get_host_name():
    try:
        return platform.node()
    except:
        return 'unknown'
