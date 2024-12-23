from typing import Callable

import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, success_func: Callable = None, llm_reward_function: Callable = None):
        super().__init__(env)
        self.success_func = success_func
        self.llm_reward_function = llm_reward_function

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(
            action
        )
        if self.llm_reward_function is not None and self.success_func is not None:
            info['TimeLimit.truncated'] = truncated
            info['terminated'] = terminated
            is_success = self.success_func(observation, info)
            reward = self.llm_reward_function(observation, is_success)
        else:
            reward = original_reward

        return observation, reward, terminated, truncated, info
