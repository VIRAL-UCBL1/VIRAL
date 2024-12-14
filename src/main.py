import argparse
from logging import getLogger

import gymnasium as gym

from log.log_config import init_logger
from ObjectivesMetrics import objective_metric_CartPole
from RLAlgo.DirectSearch import DirectSearch
from RLAlgo.Reinforce import Reinforce
from VIRAL import VIRAL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    args = parser.parse_args()

    if args.verbose:
        init_logger("DEBUG")
        print("Verbose mode enabled")
    else:
        init_logger("INFO")
    logger = getLogger("VIRAL")
    env = gym.make("CartPole-v1")
    # learning_method = DirectSearch(env)
    learning_method = Reinforce(env, couche_cachee=[64])
    objectives_metrics = [objective_metric_CartPole]
    viral = VIRAL(learning_method, env, objectives_metrics)
    res = viral.generate_reward_function(
        task_description="""Balance a pole on a cart, 
        Num Observation Min Max
        0 Cart Position -4.8 4.8
        1 Cart Velocity -Inf Inf
        2 Pole Angle ~ -0.418 rad (-24°) ~ 0.418 rad (24°)
        3 Pole Angular Velocity -Inf Inf
        Since the goal is to keep the pole upright for as long as possible.
        """,
        iterations=1,
    )
    for state in res:
        logger.info(state)
