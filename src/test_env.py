import argparse
from logging import getLogger

<<<<<<< HEAD
from Environments import Algo, CartPole, LunarLander, Pacman
from LLM.LLMOptions import additional_options
=======
from Environments import Algo, CartPole, LunarLander, Pacman, Highway
from LLM.LLMOptions import llm_options
>>>>>>> f343acce4ef7dd22555fda5e3d264f4b00f78360
from log.log_config import init_logger
from log.LoggerCSV import LoggerCSV
from RLAlgo.DirectSearch import DirectSearch
from RLAlgo.Reinforce import Reinforce
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
    # env_type = LunarLander(Algo.PPO)
    env_type = Highway(Algo.DQN)
    model = 'qwen2.5-coder'
    human_feedback = True
    LoggerCSV(env_type, model)
    viral = VIRAL(
        env_type=env_type, model=model, hf=human_feedback, training_time=int(20_000), numenvs=1, options=llm_options)
    are_worsts, are_betters, threshold = viral.policy_trainer.evaluate_policy([])
    viral.policy_trainer.test_policy_hf("model/highway-v0_0.pth", 5)
    for state in viral.memory:
        viral.logger.info(state)

if __name__ == "__main__":
    main()
