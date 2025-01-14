import gymnasium as gym
from Environments import Algo
from .EnvType import EnvType
from utils.utils import unwrap_env


class LunarLander(EnvType):
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
            "tensorboard_log": "model/LunarLanderDQN/",
        },
        prompt: dict | str = {
            "Goal": "Land safely on the ground, but don't move if you touch the ground",
            "Observation Space": """Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
        The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.
    """,
        },
    ) -> None:
        super().__init__(algo, algo_param, prompt)

    def __repr__(self):
        return "LunarLander-v3"

    def success_func(self, env: gym.Env, info: dict) -> tuple[bool | bool]:
        """
        Cette fonction vérifie si le lander est "awake" et met à jour l'info.
        """
        # print("Lunar Lander Function")
        base_env = unwrap_env(env)  # unwrap the environment
        # print(base_env)  # print the environment
        # print(base_env.lander)  # print the lander object

        # check if the lander is awake
        if not base_env.lander.awake:
            return True, False
        else:
            return False, True

    def objective_metric(self, states) -> list[dict[str, float]]:
        pass  # TODO
