from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages")); import deepracer_gym
# REF: Balaji, B. et al. (2020). DeepRacer: Autonomous Racing Platform for Sim2Real RL. IEEE ICRA.
# REF: Salazar, J. et al. (2024). Deep RL for Autonomous Driving in AWS DeepRacer. Information, 15(2).
# REF: Samant, N. & Deshpande, A. (2020). How we broke into the top 1% of AWS DeepRacer. Building Fynd.
import math, socket, hashlib
import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import yaml