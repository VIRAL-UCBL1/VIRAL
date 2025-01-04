import argparse
from logging import getLogger

from log.log_config import init_logger
from log.LoggerCSV import LoggerCSV
from RLAlgo.DirectSearch import DirectSearch
from RLAlgo.Reinforce import Reinforce
from Environments import Prompt, Algo, CartPole, LunarLander
from VIRAL import VIRAL
from LLM.LLMOptions import additional_options

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

def main():
    """
    Main entry point of the script.

    This block is executed when the script is run directly. It initializes the
    logger, and run VIRAL. It uses CLI interface.
    memory.
    """
    parse_logger()
    env_type = LunarLander(Algo.PPO)
    model = 'qwen2.5-coder'
    human_feedback = True
    LoggerCSV(env_type, model)
    viral = VIRAL(
        env_type=env_type, model=model, hf=human_feedback, training_time=400_000, numenvs=2, options=additional_options)
    viral.test_reward_func("""
def reward_func(observations: np.ndarray, is_success: bool, is_failure: bool) -> float:
    x, y, v_x, v_y, theta, omega, leg_1, leg_2 = observations

    # Penalty for altitude and horizontal distance
    altitude_penalty = -abs(y) * 0.5  # Scaled down to emphasize landing stability over altitude
    distance_penalty = abs(x)

    # Reward for landing safely
    landing_reward = 100 if is_success else -50 if is_failure else 0

    # Penalty for angular deviation from vertical
    angular_penalty = abs(theta) * 1.0  # Scaled down to make it less significant

    # Penalize large angular velocity
    angular_velocity_penalty = abs(omega)

    # Reward for maintaining leg contact with the ground
    leg_contact_reward = 5 if leg_1 == 1 and leg_2 == 1 else -2  # Adjusted weights to be more punitive

    # Final reward calculation
    total_reward = landing_reward + altitude_penalty + distance_penalty - angular_penalty - angular_velocity_penalty + leg_contact_reward

    return max(total_reward, 0)""")
    viral.policy_trainer.test_policy_hf("model/LunarLander-v3_1.pth", 5)
    for state in viral.memory:
        viral.logger.info(state)

if __name__ == "__main__":
    main()
