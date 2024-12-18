import gymnasium as gym
from Environments import Algo
from .EnvType import EnvType
from utils.utils import unwrap_env

class LunarLander(EnvType):
	def __init__(self, algo: Algo):
        # Appel du constructeur de la classe mère
		super().__init__(algo)

	def __repr__(self):
		return "LunarLander-v3"

	def success_func(self, env: gym.Env, info: dict) -> bool:
		"""
		Cette fonction vérifie si le lander est "awake" et met à jour l'info.
		"""
		# print("Lunar Lander Function")
		base_env = unwrap_env(env)  # unwrap the environment
		# print(base_env)  # print the environment
		# print(base_env.lander)  # print the lander object

		# check if the lander is awake
		if hasattr(base_env, "lander") and not base_env.lander.awake:
			return True
		else:
			return False

	def objective_metric(self, states)-> list[dict[str, float]]:
		pass # TODO