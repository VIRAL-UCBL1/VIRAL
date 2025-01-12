import gymnasium as gym
from Environments import Algo
from .EnvType import EnvType


class Highway(EnvType):
    def __init__(
        self,
        algo: Algo = Algo.DQN,
        algo_param: dict = {
            "policy": "MlpPolicy",
            "policy_kwargs": dict(net_arch=[256, 256]),
            "learning_rate": 5e-4,
            "buffer_size": 15000,
            "learning_starts": 200,
            "batch_size": 32,
            "gamma": 0.8,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 50,
            "verbose": 0,
            "tensorboard_log": "model/highway_dqn/",
        },
        prompt: dict | str = {
            "Goal": "Control the ego vehicle to reach a high speed without collision.",
            "Observation Space": """features include:
    - presence: Indicates whether a vehicle is present (1 if present, 0 otherwise).
    - x: Longitudinal position of the vehicle.
    - y: Lateral position of the vehicle.
    - vx: Longitudinal velocity of the vehicle.
    - vy: Lateral velocity of the vehicle.
Every vehicle has a line in the matrix. Number are normalized.
Observation looks like this for 4 vehicles with the ego vehicle.
Observation:
[[ 1.          0.8738351   0.33333334  0.3125      0.        ]
[ 1.          0.11421007  0.33333334 -0.04992021  0.        ]
[ 1.          0.24808374  0.         -0.02311555  0.        ]
[ 1.          0.35202843  0.33333334 -0.08297566  0.        ]
[ 1.          0.47324142  0.33333334 -0.03491976  0.        ]]
You have multiple action such as :
Each action is typically represented as an integer:
0: Keep Lane
1: Slow Down
2: Speed Up
3: Change Lane Left
4: Change Lane Right
""",
        },
    ):
        super().__init__(algo, algo_param, prompt)

    def __repr__(self):
        return "highway-v0"

    def success_func(self, env: gym.Env, info: dict) -> tuple[bool, bool]:
        """
        Vérifie si le véhicule a atteint une vitesse élevée sans collision.
        """
        speed = info.get("speed", 0)
        crashed = info.get("crashed", False)
        # print(f"info: {info}")
        print(f"speed: {speed}, crashed: {crashed}")
        # print(f"Truncated {info['TimeLimit.truncated']}")
        truncated = info["TimeLimit.truncated"]

        if truncated:
            return True, False
        else:
            return False, True

    def objective_metric(self, states) -> list[dict[str, float]]:
        """
        Calcule une métrique objective basée sur les états.
        """
        # Implémentation spécifique à votre cas d'utilisation
        pass  # TODO
