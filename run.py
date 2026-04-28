from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages")); import deepracer_gym
# REF: Balaji, B. et al. (2020). DeepRacer: Autonomous Racing Platform for Sim2Real RL. IEEE ICRA.
# REF: Salazar, J. et al. (2024). Deep RL for Autonomous Driving in AWS DeepRacer. Information, 15(2).
# REF: Samant, N. & Deshpande, A. (2020). How we broke into the top 1% of AWS DeepRacer. Building Fynd.
import math, socket, hashlib
import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import yaml
import time
import signal
import torch
import os
import json
import datetime
from collections import deque
import numpy as np
import gymnasium as gym
import harmonized_metrics as _hm
from loguru import logger
import subprocess
import csv as _csv
from munch import munchify
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math
import numpy as np
from collections import deque
from typing import Dict, Tuple