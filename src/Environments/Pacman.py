import os

import gymnasium as gym
import pkg_resources
from ale_py import ALEInterface

from Environments import Algo

from .EnvType import EnvType


class Pacman(EnvType):
    def __init__(self, algo: Algo):
        # Appel du constructeur de la classe mère
        algo_param = {
			"policy": "MlpPolicy",
			"verbose": 0,
			"device": "cpu",
			}
        super().__init__(algo, algo_param)

    def __repr__(self):
        # Chercher automatiquement le chemin d'installation des ROMs
        try:
            # Utiliser pkg_resources pour trouver le chemin du package AutoROM
            rom_path = pkg_resources.resource_filename('AutoROM', 'roms/pacman.bin')
            
            # Vérifier si le fichier existe
            if not os.path.exists(rom_path):
                raise FileNotFoundError(f"ROM file not found at {rom_path}")

            # Initialiser l'interface ALE
            ale = ALEInterface()

            # Charger la ROM spécifique à Pac-Man
            ale.loadROM(rom_path)

            # Retourner le nom de l'environnement
            return "ALE/Pacman-v5"
        
        except FileNotFoundError as e:
            return str(e)  # Renvoie l'erreur si la ROM n'est pas trouvée
        except Exception as e:
            return f"An error occurred: {e}"

    def success_func(self, env: gym.Env, info: dict) -> tuple[bool, bool]:
        """
        Fonction d'évaluation pour Pacman

        Args:
            env : gym.Env : Environnement
            info : dict : Informations provenant de l'environnement

        Returns:
            tuple : (bool, bool) indiquant si l'épisode est terminé et si c'est un succès
        """
        done = info.get("done", False)
        success = info.get("success", False)
        return done, success

    def objective_metric(self, states) -> list[dict[str, float]]:
        """
        Métrique objective pour l'environnement Pacman.
        Calcule un score basé sur l'état actuel de l'environnement.

        Args:
            states : Liste des états de l'environnement Pacman.

        Returns:
            list : Liste de dictionnaires contenant les noms et valeurs des métriques.
        """
        # Exemple de calcul de métriques basées sur les états
        score = 0
        for state in states:
            # Supposons que 'state' soit une observation de l'environnement
            # et que vous ayez une méthode pour extraire le score actuel
            score += self.extract_score_from_state(state)

        average_score = score / len(states) if states else 0

        result = [
            {"average_score": average_score},
            # Ajoutez d'autres métriques si nécessaire
        ]

        return result





