import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingInfoCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.training_metrics = {
            "episode_rewards": [], 
            "episode_observations": [], 
            "episode_lengths": [], 
        }

        self.current_episode_rewards = None
        self.current_episode_lengths = None
        self.num_envs = None

    def _on_training_start(self):
        """Init at the begin of the training"""
        self.current_episode_rewards = 0
        self.current_episode_lengths = 0

    def _on_step(self) -> bool:
        """call every steps"""
        obs = self.locals["new_obs"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self.current_episode_rewards += rewards[0]
        self.current_episode_lengths += 1
        self.training_metrics["episode_observations"].append(
                    obs[0]
                )
        if dones[0]:
            self.training_metrics["episode_rewards"].append(
                self.current_episode_rewards
            )
            self.training_metrics["episode_lengths"].append(
                self.current_episode_lengths
            )
            self.current_episode_rewards = 0
            self.current_episode_lengths = 0
        return True

    def _on_training_end(self) -> None:
        """call at the end of the training"""
        rewards = self.training_metrics["episode_rewards"]
        rewards /= np.linalg.norm(rewards)
        lengths = self.training_metrics["episode_lengths"]

        self.custom_metrics = {
            "observations": self.training_metrics["episode_observations"],
            "rewards": rewards,
            "mean_reward": np.mean(rewards) if rewards.all() else 0,
            "std_reward": np.std(rewards) if len(rewards) > 1 else 0,
        }

    def get_metrics(self):
        """for get metrics

        Returns:
            dict: contain metrics harvested
        """
        return self.custom_metrics
