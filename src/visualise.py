import argparse
from logging import getLogger

from stable_baselines3 import PPO

from Environments import (Algo, CartPole, Highway, Hopper, LunarLander, Pacman,
                          Swimmer)
from LLM.LLMOptions import llm_options
from log.log_config import init_logger
from log.LoggerCSV import LoggerCSV
from PolicyTrainer.PolicyTrainer import PolicyTrainer
from VIRAL import VIRAL


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
    parse_logger()
    env_type = Swimmer(Algo.PPO)
    p = PolicyTrainer([], 0, env_type, 1, 2, False)
    #p.test_policy_hf("data/model/Hopper-v5_932454_1.pth", 5) # "model/Hopper-v5_1.pth"
    p.test_policy_video("./data/model/Swimmer-v5_770014_1.pth", 5)

if __name__ == "__main__":
    main()
