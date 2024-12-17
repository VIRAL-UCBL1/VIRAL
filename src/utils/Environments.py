from enum import Enum


def unwrap_env(env):
    """
    Fonction récursive pour déballer les wrappers Gym jusqu'à l'environnement de base.
    """
    while hasattr(env, "env"):  # check if env is a wrapper
        env = unwrap_env(env.env)
    return env


def lunar_lander_function(env, info,terminated = None,truncated = None):
    """
    Cette fonction vérifie si le lander est "awake" et met à jour l'info.
    """
    # print("Lunar Lander Function")
    base_env = unwrap_env(env)  # unwrap the environment
    # print(base_env)  # print the environment
    # print(base_env.lander)  # print the lander object

    # check if the lander is awake
    if hasattr(base_env, "lander") and not base_env.lander.awake:
        info["success"] = True
    else:
        info["success"] = False
        
def cartpole_function(env, info,terminated = None,truncated = None):
    """
    Cette fonction vérifie si le lander est "awake" et met à jour l'info.
    """
    # print("CartPole Function")
    if truncated:  # the episode was truncated (time limit reached successfully)
        info["success"] = True
    elif terminated:  # the episode is over
        info["success"] = False
    else:
        info["success"] = None  # the episode is still running




class Environments(Enum):
    CARTPOLE = ("CartPole-v1", """Balance a pole on a cart, 
    Num Observation Min Max
    0 Cart Position -4.8 4.8
    1 Cart Velocity -Inf Inf
    2 Pole Angle ~ -0.418 rad (-24°) ~ 0.418 rad (24°)
    3 Pole Angular Velocity -Inf Inf
    Since the goal is to keep the pole upright for as long as possible.
    """,cartpole_function)
    LUNAR_LANDER = ("LunarLander-v3", """ The goal is  to land safely
    Action Space : Discrete(4) 
    Observation Space Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32),
    There are four discrete actions available:
    0: do nothing
    1: fire left orientation engine
    2: fire main engine
    3: fire right orientation engine
    The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.
    
    """,lunar_lander_function)

    def __new__(cls, value, description, function):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        obj.function = function
        return obj

    @property
    def task_description(self):
        return self.description
    
    @property
    def task_function(self):
        return self.function
