from enum import Enum


class Algo(Enum):
    """
    Enum class for RL algorithms.
    
    Attributes:
        PPO (str): Proximal Policy Optimization.
        REINFORCE (str): REINFORCE algorithm.
        DQN (str): Deep Q-Network.
    
    """
    PPO = "PPO"
    REINFORCE = "REINFORCE"
    DQN = "DQN"
