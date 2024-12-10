from typing import Callable

class State:
	
	def __init__(self, idx, reward_func: Callable, reward_func_str: str, policy = None, perfomances: dict = None):
		self.idx = idx
		self.reward_func = reward_func
		self.reward_func_str = reward_func_str
		self.policy = policy
		self.performances = perfomances

	def __repr__(self):
		pass