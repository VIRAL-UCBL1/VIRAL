from abc import abstractmethod
import gymnasium as gym
from Environments import Algo

class EnvType:
	def __init__(self, algo: Algo, algo_param: dict, prompt: dict) -> None:
		self.algo: Algo = algo
		self.algo_param: dict = algo_param
		self.prompt: dict = prompt
	
	@abstractmethod
	def __repr__(self) -> str:
		return "not implemented"
	
	@abstractmethod
	def success_func(self, env: gym.Env, info: dict) -> bool:
		raise NotImplementedError("EnvType is an abstract class")
	
	@abstractmethod
	def objective_metric(self, states)-> list[dict[str, float]]:
		raise NotImplementedError("EnvType is an abstract class")