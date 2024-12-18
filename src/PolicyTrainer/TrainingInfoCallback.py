import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingInfoCallback(BaseCallback):
    # TODO refaire ces methodes, ne pas hésiter a tout delete :)
    def __init__(self):
        super().__init__()
        self.training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_max_affile": 0,
        }

        self.current_episode_rewards = None
        self.current_episode_lengths = None
        self.num_envs = None
        self.max_sore = 0
        self.score_max_d_affile = 0
        self.premier_max = None

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
                self.training_metrics["episode_rewards"].append(self.current_episode_rewards[i])
                self.training_metrics["episode_lengths"].append(self.current_episode_lengths[i])
                
                if self.current_episode_rewards[i] > self.max_sore:
                    self.max_sore = self.current_episode_rewards[i]
                    self.score_max_d_affile = 0
                    self.training_metrics["episode_max_affile"] = self.score_max_d_affile
                    self.premier_max = i
                    
                if self.current_episode_rewards[i] == self.max_sore:
                    self.score_max_d_affile += 1
                    self.training_metrics["episode_max_affile"] = self.score_max_d_affile
                
                self.current_episode_rewards[i] = 0
                self.current_episode_lengths[i] = 0

        return True

    def _on_training_end(self) -> None:
        """Méthode appelée à la fin de l'entraînement."""
        rewards = self.training_metrics["episode_rewards"]
        lengths = self.training_metrics["episode_lengths"]
        
        self.custom_metrics = {
            "mean_reward": np.mean(rewards) if rewards else 0,
            "std_reward": np.std(rewards) if len(rewards) > 1 else 0,
            "mean_length": np.mean(lengths) if lengths else 0,
            "max_score": self.max_sore,
            "score_max_affile": self.score_max_d_affile,
            "premier_max": self.premier_max,
            "total_episodes": len(rewards)
        }

    def get_metrics(self):
        """
        Méthode pour récupérer les métriques calculées.
        :return: Dictionnaire des métriques
        """
        return self.custom_metrics