import argparse
from logging import getLogger

import gymnasium as gym

from DirectSearchAlgo import PolitiqueDirectSearch
from log.log_config import init_logger
from VIRAL import VIRAL


def objective_metric_CartPole(states):
    """
    Objective metric for the CartPole environment.
    Calculates a score for the given state on a particular observation of the CartPole environment.
    
    :param state: The state of the CartPole environment.
    :return: a table of tuples containing the string name of the metric and the value of the metric.
    """
    
    # Calculate the difference between the pole angle and the median of the pole angle range
    pole_angle_diff = 0
    for state in states:
        pole_angle = state[2]
        pole_angle_diff += abs(pole_angle)
    pole_angle_diff = pole_angle_diff / len(states)
    
    # Calculate the difference between the pole position and the median of the pole position range
    pole_position_diff = 0
    for state in states:
       pole_position = state[0]
       pole_position_diff += abs(pole_position)
    pole_position_diff = pole_position_diff / len(states)
    
    result = [("pole_angle_diff", pole_angle_diff), ("pole_position_diff", pole_position_diff)]
    
    return result
    






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

    viral.evaluate_policy(objectives_metrics=objective_metric)
