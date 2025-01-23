from enum import Enum


class Algo(Enum):
    """
    Enum class for the algorithms.
    
    Attributes:
        PPO (str): Proximal Policy Optimization
        REINFORCE (str): REINFORCE
        DQN (str): Deep Q-Network
    """
    PPO = "PPO"
    REINFORCE = "REINFORCE"
    DQN = "DQN"
