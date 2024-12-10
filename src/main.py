from logging import getLogger

import gymnasium as gym
import argparse

from VIRAL import VIRAL
from log.log_config import init_logger
from RLalgo import PolitiqueDirectSearch
from ObjectivesMetrics import objective_metric_CartPole

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    args = parser.parse_args()

    if args.verbose:
        init_logger("DEBUG")
        print("Verbose mode enabled")
    else:
        init_logger("INFO")
    logger = getLogger("DREFUN")
    env = gym.make("CartPole-v1")
    learning_method = PolitiqueDirectSearch(env)
    viral = VIRAL(learning_method, env)

    reward_func = viral.generate_reward_function(
        task_description="""Balance a pole on a cart, 
        Num Observation Min Max
        0 Cart Position -4.8 4.8
        1 Cart Velocity -Inf Inf
        2 Pole Angle ~ -0.418 rad (-24°) ~ 0.418 rad (24°)
        3 Pole Angular Velocity -Inf Inf
        Since the goal is to keep the pole upright for as long as possible, by default, a reward of +1 is given for every step taken, including the termination step. The default reward threshold is 500 for v1
        """,
    )
    
    objective_metric = [objective_metric_CartPole]

    policy, reward_func, performance_metrics, perso_states = viral.evaluate_policy(objectives_metrics=objective_metric)
    viral.self_refine_reward(reward_func, performance_metrics, perso_states)
    policy, reward_func, performance_metrics, perso_states = viral.evaluate_policy(objectives_metrics=objective_metric)

