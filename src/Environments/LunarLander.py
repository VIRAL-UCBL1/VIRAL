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