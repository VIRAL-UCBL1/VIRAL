import argparse
from logging import getLogger

from log.log_config import init_logger
from log.LoggerCSV import LoggerCSV
from Environments import Algo, CartPole, LunarLander, Hopper
from VIRAL import VIRAL
from LLM.LLMOptions import llm_options

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
    env_type = CartPole(Algo.PPO)
    model = 'qwen2.5-coder'
    human_feedback = True
    LoggerCSV(env_type, model, 25_000)
    viral = VIRAL(
        env_type=env_type, model_actor=model, model_critic=model, hf=human_feedback, training_time=25_000, nb_vec_envs=2, options=llm_options)
    viral.test_reward_func("""
def reward_func(observations:np.ndarray, is_success:bool, is_failure:bool) -> float:    
    x, x_dot, theta, theta_dot = observations
    
    if is_success:
        return 10.0
    elif is_failure:
        return -10.0
    else:
        # Reward based on how close to vertical the pole is and how stable it is
        proximity_to_vertical = np.cos(theta)
        stability_factor = np.exp(-abs(theta_dot))
        
        reward = proximity_to_vertical * stability_factor
        
        return reward""")
    # viral.policy_trainer.test_policy_hf("./model/LunarLander-v3_1.pth", 2)
    # viral.policy_trainer.test_policy_video("./model/LunarLander-v3_1.pth")

    for state in viral.memory:
        viral.logger.info(state)
    
    return viral.memory[0].performances, viral.memory[1].performances

if __name__ == "__main__":
    fperf0 = []
    fperf1 = []
    for i in range(10):
        perf0, perf1 = main()
        fperf0.append(perf0)
        fperf1.append(perf1)
    with open('cartPole_legagy_rew.txt', 'w') as fichier:
        print("Final Perf 0", fperf0, file=fichier)
    with open('cartPole_best_rew.txt', 'w') as fichier:
        print("Final Perf 1", fperf1, file=fichier)
