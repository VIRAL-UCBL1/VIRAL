import argparse
from logging import getLogger

import gymnasium as gym

from log.log_config import init_logger
from utils.ObjectivesMetrics import objective_metric_CartPole
from RLAlgo.DirectSearch import DirectSearch
from RLAlgo.Reinforce import Reinforce
from VIRAL import VIRAL

from utils.CustomRewardWrapper import CustomRewardWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from utils.Environments import Environments
from utils.Algo import Algo

def parse_logger():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    args = parser.parse_args()

    if args.verbose:
        init_logger("DEBUG")
        print("Verbose mode enabled")
    else:
        init_logger("INFO")
    

if __name__ == "__main__":
    logger = parse_logger()

    viral = VIRAL(Algo.PPO, Environments.CARTPOLE)
    res = viral.generate_reward_function(
        task_description="""Balance a pole on a cart, 
        Num Observation Min Max
        0 Cart Position -4.8 4.8
        1 Cart Velocity -Inf Inf
        2 Pole Angle ~ -0.418 rad (-24°) ~ 0.418 rad (24°)
        3 Pole Angular Velocity -Inf Inf
        Since the goal is to keep the pole upright for as long as possible.
        """,
        iterations=1,
    )
    for state in viral.memory:
        logger.info(state)
