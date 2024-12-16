from enum import Enum

class Environments(Enum):
    CARTPOLE = ("CartPole-v1", """Balance a pole on a cart, 
    Num Observation Min Max
    0 Cart Position -4.8 4.8
    1 Cart Velocity -Inf Inf
    2 Pole Angle ~ -0.418 rad (-24°) ~ 0.418 rad (24°)
    3 Pole Angular Velocity -Inf Inf
    Since the goal is to keep the pole upright for as long as possible.
    """)
    LUNAR_LANDER = ("LunarLander-v3", """ The goal is  to land safely
    Action Space : Discrete(4) 
    Observation Space Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32),
    There are four discrete actions available:
    0: do nothing
    1: fire left orientation engine
    2: fire main engine
    3: fire right orientation engine
    The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.
    
    """)

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj

    @property
    def task_description(self):
        return self.description
