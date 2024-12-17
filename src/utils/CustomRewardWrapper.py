from typing import Callable

import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, llm_reward_function: Callable = None, success_function: Callable = None):
        super().__init__(env)
        self.llm_reward_function = llm_reward_function
        self.success_function = success_function

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        # 5 env 
        if self.llm_reward_function is not None:
            reward = self.llm_reward_function(observation, terminated, truncated)
        else:
            reward = original_reward
        if self.success_function is not None:
            self.success_function(self.env, info)
        
        return observation, reward, terminated, truncated, info
