import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TrainingInfoCallback(BaseCallback):
    # TODO refaire ces methodes, ne pas hésiter a tout delete :)
    def __init__(self):
        super().__init__()
        self.training_metrics = {
            "episode_rewards": [],  
            "episode_lengths": [], 
        }

        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.custom_metrics = {}

    def _on_step(self) -> bool:
        """Méthode appelée à chaque étape de l'entraînement."""
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        if self.locals["dones"][0]:
            self.training_metrics["episode_rewards"].append(self.current_episode_reward)
            self.training_metrics["episode_lengths"].append(self.current_episode_length)

            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def _on_training_end(self) -> None:
        """Méthode appelée à la fin de l'entraînement."""
        rewards = self.training_metrics["episode_rewards"]
        lengths = self.training_metrics["episode_lengths"]

        self.custom_metrics = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths)
        }

    def get_metrics(self):
        """
        Méthode pour récupérer les métriques calculées.
        :return: Dictionnaire des métriques
        """
        return self.custom_metrics
