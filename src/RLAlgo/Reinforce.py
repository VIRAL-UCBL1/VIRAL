import random
from copy import deepcopy
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PolitiqueRenforce(nn.Module):
    def __init__(self, env, couche_cachee: list, gamma: float = 0.99):
        """
        env: Environnement Gymnasium
        couche_cachee: Liste des tailles des couches cachées (exemple [64, 32])
        gamma: Facteur de discount pour le calcul des retours cumulés
        det: Si la politique est déterministe ou stochastique
        """
        super(PolitiqueRenforce, self).__init__()
        self.dim_entree = env.observation_space.shape[0]
        self.dim_sortie = env.action_space.n
        self.gamma = gamma
        self.env = env

        # Créer dynamiquement les couches cachées
        self.couches_cachees = []
        input_size = self.dim_entree
        for hidden_size in couche_cachee:
            self.couches_cachees.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.couches_cachees = nn.ModuleList(self.couches_cachees)

        # La dernière couche qui produit la sortie
        self.fc_out = nn.Linear(input_size, self.dim_sortie)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
    def __repr__(self):
        return "PolitiqueRenforce"

    def forward(self, etat: torch.Tensor) -> torch.Tensor:
        """
        Etat: un tenseur d'état (entrée)
        return: Distribution de probabilité sur les actions
        """
        x = etat
        for layer in self.couches_cachees:
            x = F.relu(layer(x))
        x = self.fc_out(x)
        return F.softmax(x, dim=1)  # Softmax pour obtenir une distribution de probabilité

    def action(self, etat: np.ndarray) -> tuple[int, torch.Tensor]:
        """
        Renvoi l'action à exécuter dans l'état et la log-proba de cette action.
        etat: un état de l'environnement
        return: action à exécuter et la log-proba de cette action
        """
        if isinstance(etat, np.ndarray):
            etat = torch.tensor(etat, dtype=torch.float).unsqueeze(0)  # Ajouter une dimension pour le batch
        proba = self.forward(etat)
        m = Categorical(proba)
        action = m.sample()
        log_proba = m.log_prob(action)
        return action.item(), log_proba
    
    def output(self, etat: np.ndarray) -> int:
        """
        Calcul l'action à exécuter dans l'état.
        """
        # Nous utilisons une fonction d'activation soft max pour que les poids soient à la même échelle
        action, _ = self.action(etat)
        return action

    def trajectoire(self, env, reward_fonc, max_t: int = 1000) -> list[list, list, bool]:
        """
        Simule une trajectoire dans l'environnement en utilisant la politique.
        env: Environnement Gymnasium
        max_t: nombre max de pas de la trajectoire
        return: liste des récompenses et liste des log-probas des actions prises et si la trajectoire est tronquée
        """
        etat, _ = env.reset(seed=random.randint(0, 5000))
        recompenses = []
        log_probas = []
        is_success = False
        for t in range(max_t):
            action, log_proba = self.action(etat)
            etat_suivant, recompense, fini, truncated, _ = env.step(action)
            if reward_fonc is not None:
                recompense = reward_fonc(etat_suivant, fini, truncated)
            recompenses.append(recompense)
            log_probas.append(log_proba)
            if fini:
                return recompenses,log_probas, is_success
            if truncated:
                is_success = True
                return recompenses,log_probas, is_success
            etat = etat_suivant

        return recompenses, log_probas, is_success

    def calcul_retours_cumules(self, recompenses: list) -> list:
        """
        Calcule les retours cumulés pondérés par le facteur de discount gamma.
        recompenses: liste des récompenses obtenues lors d'une trajectoire
        return: liste des retours cumulés
        """
        retour_cum = []
        rec_cum_t = 0
        for recompense in reversed(recompenses):
            rec_cum_t = recompense + self.gamma * rec_cum_t
            retour_cum.insert(0, rec_cum_t)  # Insérer au début pour maintenir l'ordre
        return retour_cum

    def loss(self, log_probs: list, retours_cumules: list) -> torch.Tensor:
        """
        Calcule la perte de type REINFORCE.
        log_probs: liste des log-probas des actions prises lors d'une trajectoire
        retours_cumules: liste des retours cumulés pondérés
        return: perte
        """
        loss = [-log_prob * retour for log_prob, retour in zip(log_probs, retours_cumules)]
        return torch.cat(loss).sum()

    def train(
        self, reward_func=None, nb_episodes: int = 5000, max_t: int = 1000, save_name: str = "model/modelRenforce.pth", stop: threading.Event|None = None
    ) -> tuple[dict, list, float, int]:
        """
        Entraîne la politique en utilisant l'algorithme REINFORCE tout en restaurant les paramètres initiaux.
        
        Args:
            reward_func (callable, optional): Fonction de récompense personnalisée.
            nb_episodes (int): Nombre maximum d'épisodes.
            max_t (int): Nombre maximum de pas par épisode.
            save_name (str): Chemin pour sauvegarder le modèle.

        Returns:
            tuple: (Paramètres entraînés, Historique des récompenses, Taux de succès, Nombre d'épisodes exécutés).
        """
        # Sauvegarde des paramètres initiaux
        original = deepcopy(self.state_dict())

        # Initialisation des métriques
        recompenses = []
        a_la_suite = 0
        nb_success = 0
        score_max = 0

        for ep in range(nb_episodes):
            if stop is not None:
                if stop.is_set():
                    break
            self.optimizer.zero_grad()

            # Génération de trajectoire et calcul des récompenses
            recompense_ep, log_proba_ep, success = self.trajectoire(self.env, reward_func, max_t)
            nb_success += success

            # Calcul des retours cumulés
            retours_cum = self.calcul_retours_cumules(recompense_ep)
            recompenses.append(sum(recompense_ep))

            # Calcul et application du gradient
            loss = self.loss(log_proba_ep, retours_cum)
            loss.backward()
            self.optimizer.step()
    

            # Condition d'arrêt si le score maximum est atteint plusieurs fois consécutivement
            if recompenses[-1] == score_max:
                a_la_suite += 1
                score_max = recompenses[-1]
            else:
                a_la_suite = 0
            
            if recompenses[-1] > score_max:
                score_max = recompenses[-1]
                a_la_suite = 0

            if a_la_suite == 10:
                self.save(save_name)
                break

            # Affichage des progrès tous les 100 épisodes
            if ep % 100 == 0:
                print(f"Épisode: {ep}, Récompense: {sum(recompense_ep)}")

        # Sauvegarde des paramètres après entraînement
        trained_state = deepcopy(self)

        # Restauration des paramètres initiaux
        self.load_state_dict(original)

        # Calcul du taux de succès
        taux_success = nb_success / nb_episodes

        return trained_state, recompenses, taux_success, ep + 1


    def save(self, file: str):
        """
        Sauvegarde le modèle.
        """
        torch.save(self.state_dict(), file)

    def load(self, file: str):
        """
        Charge un modèle sauvegardé.
        """
        self.load_state_dict(torch.load(file))
