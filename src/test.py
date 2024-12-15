import gymnasium as gym
import numpy as np

from CustomRewardWrapper import CustomRewardWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def reward_func(observations:np.ndarray, terminated: bool, truncated: bool) -> float:
    """Reward function for CartPole

    Args:
        observations (np.ndarray): observation on the current state
        terminated (bool): episode is terminated due a failure
        truncated (bool): episode is truncated due a success

    Returns:
        float: The reward for the current step
    """
    if terminated or truncated:
        return -1.0  # Penalize termination or truncation
    else:
        return 1.0  # Reward for every step taken


# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4, wrapper_class=CustomRewardWrapper, wrapper_kwargs={'llm_reward_function': reward_func})

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")