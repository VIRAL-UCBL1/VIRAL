import argparse
from logging import getLogger

from Environments import Algo, CartPole, LunarLander, Pacman, Hopper, Highway
from LLM.LLMOptions import additional_options
from log.log_config import init_logger
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
    env_type = LunarLander(algo=Algo.DQN)
    model = 'phi4'
    viral = VIRAL(env_type=env_type, model=model, options=additional_options, training_time=50_000)
    viral.generate_context(env_type.prompt)
    viral.generate_reward_function(n_init=1, n_refine=0)

if __name__ == "__main__":
    for i in range(20):
        main()
