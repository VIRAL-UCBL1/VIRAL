import gymnasium as gym
from Environments import Algo
from .EnvType import EnvType
from utils.utils import unwrap_env

class LunarLander(EnvType):
	def __init__(self, algo: Algo):
        # Appel du constructeur de la classe mère
		algo_param = {
			"policy": "MlpPolicy",
			"verbose": 0,
			"device": "cpu",
			"ent_coef": 0.01,
			"gae_lambda": 0.98,
			"gamma": 0.999,
			"n_epochs": 4,
			"n_steps": 1024,
			"normalize_advantage": False
		}
		prompt = {
        "Goal": "Land safely and fast",
        "Observation Space": """Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
        The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.
    """,
    }
		super().__init__(algo, algo_param, prompt)

	def __repr__(self):
		return "LunarLander-v3"

	def success_func(self, env: gym.Env, info: dict) -> tuple[bool|bool]:
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

	def objective_metric(self, states)-> list[dict[str, float]]:
		pass # TODO