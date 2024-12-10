from typing import Callable
from logging import getLogger

logger = getLogger('VIRAL')

class State:
	
	def __init__(self, idx, reward_func: Callable = None, reward_func_str: str = None, policy = None, perfomances: dict = None):
		self.idx = idx
		if self.idx == 0 and (reward_func is not None or reward_func_str is not None):
			logger.error("the inital state don't take reward function")
		elif self.idx != 0 and (reward_func is None or reward_func_str is None):
			logger.error("you need to give a reward function to the state")
		self.reward_func = reward_func
		self.reward_func_str = reward_func_str
		self.policy = policy
		self.performances = perfomances

	def set_policy(self, policy):
		self.policy = policy
	
	def set_performances(self, performances: dict):

		self.performances = performances

	def __repr__(self):
		if self.performances is None:
			repr = f"state {self.idx}: \nreward function: \n\n{self.reward_func_str}\n\n isn't trained yet"
		else:
			repr = f"state {self.idx}: \nreward function: \n\n{self.reward_func_str}\n\n have this performances: {self.performances} with the policy {self.policy}"
		return repr