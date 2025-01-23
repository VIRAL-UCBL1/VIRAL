from abc import abstractmethod

import gymnasium as gym

from Environments import Algo


class EnvType:
    """
    Abstract class for environment types.
    """
    def __init__(self, algo: Algo, algo_param: dict, prompt: dict) -> None:
        """
        Initializes the environment.
        
        Args:
            algo (Algo): The algorithm to use for training.
            algo_param (dict): The parameters for the algorithm.
            prompt (dict | str): The prompt for the environment.
        """
        self.algo: Algo = algo
        self.algo_param: dict = algo_param
        self.prompt: dict = prompt
    
    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of the environment.
  
        Returns:
            str: String representation of the environment.
        """
        return "not implemented"
    
    @abstractmethod
    def success_func(self, env: gym.Env, info: dict) -> bool:
        raise NotImplementedError("EnvType is an abstract class")
    
    @abstractmethod
    def objective_metric(self, states)-> list[dict[str, float]]:
        raise NotImplementedError("EnvType is an abstract class")