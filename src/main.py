import argparse
from logging import getLogger

from Environments import Algo, CartPole, LunarLander, Pacman, Hopper, Highway
from LLM.LLMOptions import llm_options
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
        init_logger()

    return getLogger()


def main():
    """
    Main entry point of the script.

    This block is executed when the script is run directly. It initializes the
    logger, and run VIRAL. It uses CLI interface.
    memory.
    """
    parse_logger()
    env_type = CartPole(
        prompt={
            "Goal": "Balance a pole on a cart",
            "Observation Space": """Num Observation Min Max
            0 Cart Position -4.8 4.8
            1 Cart Velocity -Inf Inf
            2 Pole Angle ~ -0.418 rad (-24°) ~ 0.418 rad (24°)
            3 Pole Angular Velocity -Inf Inf""",
            "Image": "Environments/img/CartPole.png",
        }
    )
    actor = "qwen2.5-coder"
    critic = "llama3.2-vision"
    human_feedback = False
    viral = VIRAL(
        env_type=env_type,
        model_actor=actor,
        model_critic=critic,
        options=llm_options,
        legacy_training=False,
        training_time=30_000,
    )
    viral.generate_context()
    viral.generate_reward_function(n_init=2, n_refine=0)
    for state in viral.memory:
        viral.logger.info(state)


if __name__ == "__main__":
    for i in range(50):
        main()
