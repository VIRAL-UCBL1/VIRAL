import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingInfoCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.training_metrics = {
            "episode_rewards": [],  
            "episode_lengths": [], 
        }

        self.current_episode_rewards = None
        self.current_episode_lengths = None
        self.num_envs = None

    def _on_training_start(self):
        """Initialisation au début de l'entraînement"""
        self.num_envs = self.training_env.num_envs
        self.current_episode_rewards = np.zeros(self.num_envs)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=int)

    def _on_step(self) -> bool:
        """Méthode appelée à chaque étape de l'entraînement."""
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self.current_episode_rewards += rewards
        self.current_episode_lengths += 1

        for i in range(self.num_envs):
            if dones[i]:
                self.training_metrics["episode_rewards"].append(
                    self.current_episode_rewards[i]
                )
                self.training_metrics["episode_lengths"].append(
                    self.current_episode_lengths[i]
                )

                self.current_episode_rewards[i] = 0
                self.current_episode_lengths[i] = 0

        return True

    def _on_training_end(self) -> None:
        """Méthode appelée à la fin de l'entraînement."""
        rewards = self.training_metrics["episode_rewards"]
        rewards /= np.linalg.norm(rewards)
        lengths = self.training_metrics["episode_lengths"]

        self.custom_metrics = {
            "mean_reward": np.mean(rewards) if rewards.all() else 0,
            "std_reward": np.std(rewards) if len(rewards) > 1 else 0,
            "mean_length": np.mean(lengths) if lengths else 0,
            "total_episodes": len(rewards),
        }

    def get_metrics(self):
        """
        Méthode pour récupérer les métriques calculées.
        :return: Dictionnaire des métriques
        """
        return self.custom_metrics
