import os

import gymnasium as gym
import pkg_resources
from ale_py import ALEInterface

from Environments import Algo

from .EnvType import EnvType


class Pacman(EnvType):
    """
    This class represents the Pacman environment.
    """
    def __init__(
        self,
        algo: Algo = Algo.DQN,
        algo_param: dict = {
            "policy": "MlpPolicy",
            "verbose": 0,
            "device": "cpu",
        },
        prompt: dict | str = {
            "Goal": "Collect all food items and avoid ghosts unless a Power Pellet is consumed, enabling Pacman to eat ghosts.",
            "Observation Space": """
    Type              Shape               Description
    rgb               Box(0, 255, (210, 160, 3), uint8)  Full-color 3D representation of the environment
    grayscale         Box(0, 255, (210, 160), uint8)     Grayscale version of the visual environment
    ram               Box(0, 255, (128,), uint8)         RAM representation of the game state
    """,
        },
    ) -> None:
        """
        Constructor for the Pacman environment
        
        Args:
            algo (Algo, optional): The algorithm to be used for training. Defaults to Algo.DQN.
            algo_param (dict, optional): The parameters for the algorithm. Defaults to {"policy": "MlpPolicy", "verbose": 0, "device": "cpu"}.
            prompt (dict | str, optional): The prompt for the environment. Defaults to {"Goal": "Collect all food items and avoid ghosts unless a Power Pellet is consumed, enabling Pacman to eat ghosts.", "Observation Space": [...].
        
        """
        try:
            rom_path = pkg_resources.resource_filename("AutoROM", "roms/pacman.bin")

            if not os.path.exists(rom_path):
                raise FileNotFoundError(f"ROM file not found at {rom_path}")
            ale = ALEInterface()
            ale.loadROM(rom_path)
        except FileNotFoundError as e:
            print(str(e))
        except Exception as e:
            print(f"An error occurred: {e}")
        super().__init__(algo, algo_param, prompt)

    def __repr__(self):
        """
        Function to return the name of the environment
        
        Returns:
            str: The name of the environment.
        
        """
        return "ALE/Pacman-v5"

    def success_func(self, env: gym.Env, info: dict) -> tuple[bool, bool]:
        """
        Function to check if the Pacman has successfully completed the game or failed.
        
        Args:
            env (gym.Env): The environment.
            info (dict): The information about the environment.
        
        Returns:
            tuple[bool, bool]: A tuple of two booleans. The first boolean represents if the Pacman has completed the game successfully, and the second boolean represents if the Pacman has failed.
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
        Method to calculate the objective metric for the Pacman environment
        
        Args:
            states: The states of the environment
        
        Returns:
            list[dict[str, float]]: A list of dictionaries containing the objective metric for the environment.
        """
        pass  # TODO
