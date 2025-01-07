import argparse
from logging import getLogger

from Environments import Algo, CartPole, LunarLander, Pacman, Hopper
from LLM.LLMOptions import additional_options
from log.log_config import init_logger
from log.LoggerCSV import LoggerCSV
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
    env_type = CartPole(Algo.PPO)
    actor = 'falcon3:10b'
    critic = 'llama3.2-vision'
    human_feedback = False
    LoggerCSV(env_type, actor+critic)
    viral = VIRAL(
        env_type=env_type, model_actor=actor, model_critic=critic, hf=human_feedback, training_time=25_000,
        numenvs=2, options=additional_options)
    viral.generate_context()
    viral.generate_reward_function(n_init=1, n_refine=5)
    for state in viral.memory:
        viral.logger.info(state)

if __name__ == "__main__":
    for _ in range(10):
        main()
