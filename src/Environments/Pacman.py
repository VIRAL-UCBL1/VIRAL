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
        prompt = {
        "Goal": "Collect all food items and avoid ghosts unless a Power Pellet is consumed, enabling Pacman to eat ghosts.",
        "Observation Space": """
    Type              Shape               Description
    rgb               Box(0, 255, (210, 160, 3), uint8)  Full-color 3D representation of the environment
    grayscale         Box(0, 255, (210, 160), uint8)     Grayscale version of the visual environment
    ram               Box(0, 255, (128,), uint8)         RAM representation of the game state
    """,
    }
        super().__init__(algo, algo_param, prompt)

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
        except FileNotFoundError as e:
            print(str(e)) # Renvoie l'erreur si la ROM n'est pas trouvée
        except Exception as e:
            print(f"An error occurred: {e}")
            
        super().__init__(algo, algo_param)

    def __repr__(self):
        # Retourner le nom de l'environnement
        return "ALE/Pacman-v5"
        


    def success_func(self, env: gym.Env, info: dict) -> tuple[bool, bool]:
        """
        Fonction d'évaluation pour Pacman

        Args:
            env : gym.Env : Environnement
            info : dict : Informations provenant de l'environnement

        Returns:
            tuple : (bool, bool) indiquant si l'épisode est terminé et si c'est un succès
        """
        nb_lives = info.get('lives')
        # print(f"info: {info}")
        # print(f"Number of lives: {nb_lives}")
        if nb_lives is not None and nb_lives == 0:
            done = True
            success = False
        else:
            done = False
            success = True
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
        pass #TODO





