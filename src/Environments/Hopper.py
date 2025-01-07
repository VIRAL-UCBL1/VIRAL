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
		prompt = {
        "Goal": "Control the Hopper to move in the forward direction",
        "Observation Space": """Box(-inf, inf, (11,), float64)

The observation space consists of the following parts (in order):
qpos (5 elements by default): Position values of the robot’s body parts.
qvel (6 elements): The velocities of these individual body parts (their derivatives).
the x- and y-coordinates are returned in info with the keys "x_position" and "y_position", respectively.

| Num      | Observation                                      | Min   | Max  | Type                |
|----------|--------------------------------------------------|-------|------|---------------------|
| 0        | z-coordinate of the torso (height of hopper)     | -Inf  | Inf  | position (m)        |
| 1        | angle of the torso                               | -Inf  | Inf  | angle (rad)         |
| 2        | angle of the thigh joint                         | -Inf  | Inf  | angle (rad)         |
| 3        | angle of the leg joint                           | -Inf  | Inf  | angle (rad)         |
| 4        | angle of the foot joint                          | -Inf  | Inf  | angle (rad)         |
| 5        | velocity of the x-coordinate of the torso        | -Inf  | Inf  | velocity (m/s)      |
| 6        | velocity of the z-coordinate (height) of torso   | -Inf  | Inf  | velocity (m/s)      |
| 7        | angular velocity of the angle of the torso       | -Inf  | Inf  | angular velocity (rad/s) |
| 8        | angular velocity of the thigh hinge              | -Inf  | Inf  | angular velocity (rad/s) |
| 9        | angular velocity of the leg hinge                | -Inf  | Inf  | angular velocity (rad/s) |
| 10       | angular velocity of the foot hinge               | -Inf  | Inf  | angular velocity (rad/s) |
| excluded | x-coordinate of the torso                        | -Inf  | Inf  | position (m)        |
""",
		"Image": "./Environments/img/Hopper.png"
}
		super().__init__(algo, algo_param, prompt)

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
