import gymnasium as gym

from Environments import Algo

from .EnvType import EnvType


class Swimmer(EnvType):
    def __init__(
        self,
        algo: Algo = Algo.PPO,
        algo_param: dict = {
            "policy": "MlpPolicy",
            "verbose": 0,
            "device": "cpu",
        },
        prompt={
            "Goal": "Control the swimmer to move as fast as possible to the right by applying torques to the joints.",
            "Observation Space": """Box(-inf, inf, (8,), float64)

The observation space consists of the following elements (in order):
- qpos (3 elements by default): Position values of the robot’s body parts.
- qvel (5 elements): Velocities of these body parts (their derivatives).

By default, the observation does not include the x- and y-coordinates of the front end. These can be included by passing `exclude_current_positions_from_observation=False` during construction. In this case, the observation space will be `Box(-Inf, Inf, (10,), float64)`, where the first two observations are the x- and y-coordinates of the front end. Regardless of the value of `exclude_current_positions_from_observation`, the x- and y-coordinates are returned in `info` with the keys "x_position" and "y_position", respectively.

By default, the observation space is `Box(-Inf, Inf, (8,), float64)` with the following elements:

| Num | Observation                                | Min  | Max  | Type                   |
|-----|--------------------------------------------|------|------|------------------------|
| 0   | Angle of the front end                    | -Inf | Inf  | angle (rad)            |
| 1   | Angle of the first joint                  | -Inf | Inf  | angle (rad)            |
| 2   | Angle of the second joint                 | -Inf | Inf  | angle (rad)            |
| 3   | Velocity of the front end along the x-axis| -Inf | Inf  | velocity (m/s)         |
| 4   | Velocity of the front end along the y-axis| -Inf | Inf  | velocity (m/s)         |
| 5   | Angular velocity of the front end         | -Inf | Inf  | angular velocity (rad/s) |
| 6   | Angular velocity of the first joint       | -Inf | Inf  | angular velocity (rad/s) |
| 7   | Angular velocity of the second joint      | -Inf | Inf  | angular velocity (rad/s) |
""",
            "Action Space": """Box(-1, 1, (2,), float32)

The action space is a `Box(-1, 1, (2,), float32)`. An action represents the torques applied between the segments.

| Num | Action                              | Min | Max | Name (in corresponding XML file) | Joint   | Type (Unit) |
|-----|-------------------------------------|-----|-----|----------------------------------|---------|-------------|
| 0   | Torque applied on the first joint  | -1  | 1   | motor1_rot                      | hinge   | torque (N m) |
| 1   | Torque applied on the second joint | -1  | 1   | motor2_rot                      | hinge   | torque (N m) |
""",
        },
    ) -> None:
        super().__init__(algo, algo_param, prompt)

    def __repr__(self):
        return "Swimmer-v5"

    def success_func(self, env: gym.Env, info: dict) -> tuple[bool, bool]:
        """Success function for the Swimmer

        Args:
            env (gym.Env):
            info (dict):

        Returns:
            tuple[bool, bool]: is_success, is_failure tuple
        """
        if info.get("terminated", False):
            return False, True
        elif info.get("x_position", 0) > 5.0:
            return True, False
        else:
            return False, False

    def objective_metric(self, states) -> dict[str, float]:
        """
        Objective metric for the Swimmer environment.
        Calculates a score for the given state during a particular observation of the Swimmer environment.

        :param states: The state of the Swimmer environment.
        :return: A dictionary containing the name of the metric and its value.
        """
        return {}