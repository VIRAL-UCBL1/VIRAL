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

    return getLogger()
    

if __name__ == "__main__":
    logger = parse_logger()
    viral = VIRAL(Algo.PPO, Environments.CARTPOLE)
    res = viral.generate_reward_function(
        task_description=Environments.CARTPOLE.task_description,
        iterations=2,
    )
    for state in viral.memory:
        logger.info(state)
