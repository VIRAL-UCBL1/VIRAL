import argparse
from logging import getLogger

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from log.log_config import init_logger
from RLAlgo.DirectSearch import DirectSearch
from RLAlgo.Reinforce import Reinforce
from utils.Algo import Algo
from utils.CustomRewardWrapper import CustomRewardWrapper
from utils.Environments import Environments
from utils.ObjectivesMetrics import objective_metric_CartPole
from VIRAL import VIRAL
import multiprocessing as mp

def parse_logger():
    """
    Parses command-line arguments to configure the logger.

    Returns:
        Logger: Configured logger instance.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    args = parser.parse_args()

    if args.verbose:
        init_logger("DEBUG")
        print("Verbose mode enabled")
    else:
        init_logger("DEBUG")

    return getLogger()

def main():
    """
    Main entry point of the script.

    This block is executed when the script is run directly. It initializes the
    logger, and run VIRAL. It uses CLI interface.
    memory.
    """
    logger = parse_logger()
    viral = VIRAL(
        learning_algo=Algo.PPO,
        env_type=Environments.CARTPOLE, 
        success_function=Environments.CARTPOLE.task_function)
    res = viral.generate_reward_function(
        task_description=Environments.CARTPOLE.task_description,
        iterations=2,
    )
    for state in viral.memory:
        logger.info(state)

if __name__ == "__main__":
    main()