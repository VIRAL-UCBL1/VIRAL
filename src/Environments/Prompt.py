from enum import Enum


class Prompt(Enum):
    CARTPOLE = {
        "Goal": "Balance a pole on a cart",
        "Observation Space": """Num Observation Min Max
0 Cart Position -4.8 4.8
1 Cart Velocity -Inf Inf
2 Pole Angle ~ -0.418 rad (-24°) ~ 0.418 rad (24°)
3 Pole Angular Velocity -Inf Inf""",
    }
    LUNAR_LANDER = {
        "Goal": "Land safely and fast",
        "Observation Space": """Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
        The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.
    """,
    }

    PACMAN = {
        "Goal": "Collect all food items and avoid ghosts unless a Power Pellet is consumed, enabling Pacman to eat ghosts.",
        "Observation Space": """
    Type              Shape               Description
    rgb               Box(0, 255, (210, 160, 3), uint8)  Full-color 3D representation of the environment
    grayscale         Box(0, 255, (210, 160), uint8)     Grayscale version of the visual environment
    ram               Box(0, 255, (128,), uint8)         RAM representation of the game state
    """,
    }
    HOPPER = {
        "Goal": "move forward as quickly as possible",
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
"""
}

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
