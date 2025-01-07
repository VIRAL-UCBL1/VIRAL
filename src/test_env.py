import argparse
from logging import getLogger

from Environments import Algo, CartPole, LunarLander, Pacman, Prompt
from LLM.LLMOptions import additional_options
from log.log_config import init_logger
from log.LoggerCSV import LoggerCSV
from RLAlgo.DirectSearch import DirectSearch
from RLAlgo.Reinforce import Reinforce
<<<<<<< HEAD
=======
from Environments import Algo, CartPole, LunarLander, Highway
>>>>>>> 2f1c72fb2b5738e7539e1ff4f6c68ba39c0da743
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
<<<<<<< HEAD
    # env_type = LunarLander(Algo.PPO)
    env_type = Pacman(Algo.PPO)
=======
    env_type = Highway(Algo.DQN)
>>>>>>> 2f1c72fb2b5738e7539e1ff4f6c68ba39c0da743
    model = 'qwen2.5-coder'
    human_feedback = True
    LoggerCSV(env_type, model)
    viral = VIRAL(
<<<<<<< HEAD
        env_type=env_type, model=model, hf=human_feedback, training_time=50_000, numenvs=4, options=additional_options)
=======
        env_type=env_type, model=model, hf=human_feedback, training_time=int(2e4), numenvs=1, options=additional_options)
    are_worsts, are_betters, threshold = viral.policy_trainer.evaluate_policy([])
    viral.policy_trainer.test_policy_hf("model/highway-v0_0.pth", 5)
>>>>>>> 2f1c72fb2b5738e7539e1ff4f6c68ba39c0da743
    for state in viral.memory:
        viral.logger.info(state)

if __name__ == "__main__":
    main()
