import gymnasium as gym
from Environments import Algo
from .EnvType import EnvType

class CartPole(EnvType):
	def __init__(self, algo: Algo):
        # Appel du constructeur de la classe mère
		super().__init__(algo)

	def __repr__(self):
		return "CartPole-v1"

	def success_func(self, env: gym.Env, info: dict) -> tuple[bool|bool]:
		"""
		Cartpole Evaluation Function

		Args:
			env : gym.Env : Environment
			info : dict : Information from the environment

		Returns:
			bool : True if the episode is truncated, False otherwise
		"""
		if info["TimeLimit.truncated"]:
			return True, False
		else:
			return False, True

	def objective_metric(self, states)-> list[dict[str, float]]:
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

		result = [
			{"pole_angle_diff": pole_angle_diff},
			{"pole_position_diff": pole_position_diff},
		]

		return result