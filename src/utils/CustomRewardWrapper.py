import gymnasium as gym
from typing import Callable


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, llm_reward_function: Callable = None):
        super().__init__(env)
        self.llm_reward_function = llm_reward_function

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.llm_reward_function is not None:
            reward = self.llm_reward_function(observation, terminated, truncated)

        return observation, reward, terminated, truncated, info
