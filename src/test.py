from typing import Callable
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from CustomRewardWrapper import CustomRewardWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
class TrainingInfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_metrics = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'mean_rewards': [],
            'mean_lengths': [],
            'terminated_count': 0,  # Nombre d'épisodes terminés normalement
            'truncated_count': 0,   # Nombre d'épisodes tronqués
            'total_episodes': 0     # Nombre total d'épisodes
        }
        
        self.current_episode_reward = 0
        self.current_episode_length = 0

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

    def get_metrics(self):
        """
        Méthode pour récupérer toutes les métriques collectées
        """
        return {
            'timesteps': self.training_metrics['timesteps'],
            'episode_rewards': self.training_metrics['episode_rewards'],
            'episode_lengths': self.training_metrics['episode_lengths'],
            'mean_rewards_10_episodes': self.training_metrics['mean_rewards'],
            'mean_lengths_10_episodes': self.training_metrics['mean_lengths'],
            'terminated_count': self.training_metrics['terminated_count'],
            'truncated_count': self.training_metrics['truncated_count'],
            'total_episodes': self.training_metrics['total_episodes']
        }

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

# Utilisation
training_callback = TrainingInfoCallback()

# Entraînement
model.learn(total_timesteps=60000, callback=training_callback)

# Récupération des métriques après l'entraînement
metrics = training_callback.get_metrics()

print(f"Nombre total d'épisodes : {metrics['total_episodes']}")
print(f"Épisodes terminés (terminated) : {metrics['terminated_count']}")
print(f"Épisodes tronqués (truncated) : {metrics['truncated_count']}")
# Exemple de visualisation des métriques
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Graphique des récompenses par épisode
plt.subplot(1, 2, 1)
plt.plot(metrics['episode_rewards'])
plt.title('Récompenses par Épisode')
plt.xlabel("Numéro d'Épisode")
plt.ylabel('Récompense')

# Graphique de la moyenne mobile des récompenses
plt.subplot(1, 2, 2)
plt.plot(metrics['mean_rewards_10_episodes'])
plt.title('Moyenne Mobile des Récompenses (10 derniers épisodes)')
plt.xlabel("Groupe d'Épisodes")
plt.ylabel('Récompense Moyenne')

plt.tight_layout()
plt.show()