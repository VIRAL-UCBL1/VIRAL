from copy import deepcopy

import numpy as np
import torch
from logging import getLogger

logger = getLogger("DREFUN")

class PolitiqueDirectSearch:
    def __init__(self, env, det=True):
        self.dim_entree = env.observation_space.shape[0]
        self.dim_sortie = env.action_space.n
        self.det = det
        self.env = env
        # Matrice entre * sortie
        self.poids = np.random.rand(self.dim_entree, self.dim_sortie)

    def __repr__(self):
        return "DirectSearch"

    def output(self, etat: np.ndarray) -> int:
        """Calcul de la sortie de la politique
        - si déterministe : argmax
        - si stochastique : probabilité de chaque action
        """
        # Nous utilisons une fonction d'activation soft max pour que les poids soient à la même échelle
        prob = torch.nn.functional.softmax(torch.tensor(etat.dot(self.poids)), dim=0)
        if self.det:
            return torch.argmax(prob).item()
        else:
            return torch.Categorical(probs=prob).sample().item()

    def set_poids(self, poids: np.ndarray):
        self.poids = poids

    def get_poids(self) -> np.ndarray:
        return self.poids

    def save(self, file):
        f = open(file, "w")
        f.write(f"{self.poids};{self.det}")

    def load(self, file):
        f = open(file, "r")
        param = f.read().split(";")
        self.poids = param[0]
        self.det = param[1]

    def rollout(self, reward_func, max_t=1000) -> int:
        """
        execute un episode sur l'environnement env avec la politique et renvoie la somme des recompenses obtenues sur l'épisode
        """
        total_rec = 0
        is_success = False
        state, _ = self.env.reset()
        for _ in range(1, max_t + 1):
            action = self.output(state)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            if reward_func is not None:
                reward = reward_func(next_observation, terminated, truncated)
            total_rec += reward
            state = next_observation
            if terminated:
                return total_rec, is_success
            if truncated:
                is_success = True
                return total_rec, is_success
        return total_rec, is_success

    def train(
        self, reward_func=None, nb_episodes=5000, max_t=1000, save_name=""
    ) -> tuple[list, np.ndarray]:
        original_state: PolitiqueDirectSearch = deepcopy(self)
        bruit_std = 1e-2
        meilleur_perf = 0
        meilleur_poid = self.get_poids()
        perf_by_episode = list()
        nb_best_perf = 0
        nb_success = 0
        for i_episode in range(1, nb_episodes + 1):
            perf, success = self.rollout(reward_func, max_t)
            nb_success += success
            perf_by_episode.append(perf)

            if perf == meilleur_perf:
                nb_best_perf += 1
            else:
                nb_best_perf = 0
            if nb_best_perf == 10:
                break
            if perf >= meilleur_perf:
                meilleur_perf = perf
                meilleur_poid = self.get_poids()
                # reduction de la variance du bruit
                bruit_std = max(1e-3, bruit_std / 2)
            else:
                # augmentation de la variance du bruit
                bruit_std = min(2, bruit_std * 2)
            # On calcule le bruit en fonction de la variance
            bruit = np.random.normal(0, bruit_std, self.dim_entree * self.dim_sortie)
            # Reshape le bruit pour qu'il ait la même taille que les poids
            bruit = bruit.reshape(self.dim_entree, self.dim_sortie)
            # if i_episode % 20 == 0:
            #     print(f"Episode {i_episode}, perf = {perf}, best perf = {meilleur_perf}, bruit = {bruit_std}")
            # On ajoute le bruit aux poids
            self.set_poids(self.get_poids() + bruit)
        if save_name is not None:
            self.save(save_name)
        trained_policy: PolitiqueDirectSearch = deepcopy(self)
        self.__dict__.update(original_state.__dict__)
        return trained_policy, perf_by_episode, (nb_success / i_episode), i_episode
