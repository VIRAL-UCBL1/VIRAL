import gymnasium as gym

from Environments import Algo
from utils.utils import unwrap_env

from .EnvType import EnvType


class LunarLander(EnvType):
    """
    This class represents the Lunar Lander environment.
    """
    def __init__(
        self,
        algo: Algo = Algo.DQN,
        algo_param: dict = {
            "batch_size": 128,
            "buffer_size": 50000,
            "exploration_final_eps": 0.1,
            "exploration_fraction": 0.12,
            "gamma": 0.99,
            "gradient_steps": -1,
            "learning_rate": 0.00063,
            "learning_starts": 0,
            "policy": "MlpPolicy",
            "policy_kwargs": {"net_arch": [256, 256]},
            "target_update_interval": 250,
            "train_freq": 4,
            "tensorboard_log": "data/model/LunarLanderDQN/",
        },
        prompt: dict | str = {
            "Goal": "Land safely on the ground, but don't move if you touch the ground",
            "Observation Space": """Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
        The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.
    """,
        },
    ) -> None:
        """
        Constructor for the Lunar Lander environment.
        
        Args:
            algo (Algo, optional): The algorithm to be used for training. Defaults to Algo.DQN.
            algo_param (dict, optional): The parameters for the algorithm. Defaults to {"batch_size": 128, "buffer_size": 50000, "exploration_final_eps": 0.1, "exploration_fraction": 0.12, "gamma": 0.99, "gradient_steps": -1, "learning_rate": 0.00063, "learning_starts": 0, "policy": "MlpPolicy", "policy_kwargs": {"net_arch": [256, 256]}, "target_update_interval": 250, "train_freq": 4, "tensorboard_log": "data/model/LunarLanderDQN/"}.
            prompt (dict | str, optional): The prompt for the environment. Defaults to {"Goal": "Land safely on the ground, but don't move if you touch the ground", "Observation Space": [...].
        """
        super().__init__(algo, algo_param, prompt)

    def __repr__(self):
        """
        This function returns the name of the environment.
        
        Returns:
            str: The name of the environment.
        """
        return "LunarLander-v3"

    def success_func(self, env: gym.Env, info: dict) -> tuple[bool | bool]:
        """
        This function checks if the Lunar Lander has landed successfully or failed.
        
        Args:
            env (gym.Env): The environment.
            info (dict): The information about the environment.
            
        Returns:
            tuple[bool | bool]: A tuple of two booleans. The first boolean represents if the lander has landed successfully, and the second boolean represents if the lander has failed
        """
        # print("Lunar Lander Function")
        base_env = unwrap_env(env)  # unwrap the environment
        # print(base_env)  # print the environment
        # print(base_env.lander)  # print the lander object

        # check if the lander is awake
        # print('test', type(info["obs"]), info["obs"])
        if not base_env.lander.awake and abs(info["obs"][0]) <= 0.5:
            return True, False
        elif base_env.game_over or abs(info["obs"][0]) >= 1.0:
            return False, True
        else:
            return False, False

    def objective_metric(self, states) -> list[dict[str, float]]:
        """
        This function calculates the objective metric for the Lunar Lander environment.
        
        Args:
            states (list): The states of the environment.
        
        Returns:
            list[dict[str, float]]: The objective metric for the environment.
        """
        pass  # TODO
