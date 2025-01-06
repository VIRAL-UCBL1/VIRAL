import gymnasium as gym
from .EnvType import EnvType
from Environments import Algo

class Hopper(EnvType):
	def __init__(self, algo: Algo):
        # Appel du constructeur de la classe mère
		algo_param = {
			"policy": "MlpPolicy",
			"verbose": 0,
			"device": "cpu",
			}
		super().__init__(algo, algo_param)

	def __repr__(self):
		return "Hopper-v5"

	def success_func(self, env: gym.Env, info: dict) -> tuple[bool|bool]:
		"""Hopper success_fun

		Args:
			env (gym.Env): 
			info (dict): 

		Returns:
			tuple[bool|bool]: is_success, is_failure tuple
		"""
		
		if info["terminated"]:
			return False, True
		elif info["x_position"] > 5.0:
			return True, False
		else:
			return False, False

	def objective_metric(self, states)->  dict[str, float]:
		"""
		Objective metric for the CartPole environment.
		Calculates a score for the given state on a particular observation of the CartPole environment.

		:param state: The state of the CartPole environment.
		:return: a table of tuples containing the string name of the metric and the value of the metric.
		"""
		return {}
