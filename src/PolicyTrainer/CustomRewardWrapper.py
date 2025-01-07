from typing import Callable

import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, success_func: Callable = None, llm_reward_function: Callable = None):
        """init the custom reward wrapper

        Args:
            env (gym.Env): the current environment
            success_func (Callable, optional): this function should return True if success. Defaults to None.
            llm_reward_function (Callable, optional): the generated reward function. Defaults to None.
        """
        super().__init__(env)
        self.success_func = success_func
        self.llm_reward_function = llm_reward_function

    def step(self, action):
        """override the step function to integrate our reward function

        Args:
            action (): the action to realise

        Returns:
            tuple: observation, reward, terminated, truncated, info
        """
        observation, original_reward, terminated, truncated, info = self.env.step(
            action
        )
        if self.llm_reward_function is not None and self.success_func is not None:
            info['TimeLimit.truncated'] = truncated
            info['terminated'] = terminated
            is_success = 0
            is_failure = 0
            if terminated or truncated:
                is_success, is_failure = self.success_func(self.env, info)
            reward = self.llm_reward_function(observation, is_success, is_failure)
        else:
            reward = original_reward
            #print(f"Observation: {observation}")
            # print("####"*50)
            # print(f"Reward: {reward}")
        return observation, reward, terminated, truncated, info
