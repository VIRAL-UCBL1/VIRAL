import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TrainingInfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_metrics = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'mean_rewards': [],
            'mean_lengths': [],
            'terminated_count': 0,
            'truncated_count': 0,
            'total_episodes': 0
        }
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.custom_metrics = {}

    def _on_step(self) -> bool:
        for reward, done, truncated in zip(
            self.locals['rewards'], 
            self.locals['dones'], 
            self.locals.get('truncateds', [False] * len(self.locals['rewards']))
        ):
            self.current_episode_reward += reward
            self.current_episode_length += 1
            
            # Vérifier si l'épisode est terminé ou tronqué
            if done or truncated:
                self.training_metrics['total_episodes'] += 1
                
                if done:
                    self.training_metrics['terminated_count'] += 1
                
                if truncated:
                    self.training_metrics['truncated_count'] += 1
                
                # Stocker les métriques de l'épisode
                self.training_metrics['episode_rewards'].append(self.current_episode_reward)
                self.training_metrics['episode_lengths'].append(self.current_episode_length)
                
                # Calculer et stocker les moyennes glissantes
                if len(self.training_metrics['episode_rewards']) > 10:
                    mean_reward = np.mean(self.training_metrics['episode_rewards'][-10:])
                    mean_length = np.mean(self.training_metrics['episode_lengths'][-10:])
                    
                    self.training_metrics['mean_rewards'].append(mean_reward)
                    self.training_metrics['mean_lengths'].append(mean_length)
                
                # Réinitialiser pour le prochain épisode
                self.current_episode_reward = 0
                self.current_episode_length = 0
        
        # Stocker le nombre de timesteps
        self.training_metrics['timesteps'].append(self.num_timesteps)
        
        return True

    def _on_training_end(self) -> None:
        # Calculer le taux de succès à l'entraînement
        try:
            train_success_rate = (self.training_metrics['terminated_count'] / 
                                  self.training_metrics['total_episodes']) * 100
        except ZeroDivisionError:
            train_success_rate = 0

        # Préparer les métriques finales
        self.results = {
            "train_success_rate": train_success_rate,
            "train_episodes": self.training_metrics['total_episodes'],
            "custom_metrics": self.custom_metrics
        }

    def get_metrics(self):
        """
        Méthode pour récupérer les métriques dans le format spécifié
        """
        return self.results if hasattr(self, 'results') else {}
