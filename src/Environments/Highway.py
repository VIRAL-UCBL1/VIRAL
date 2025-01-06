import gymnasium as gym
import highway_env
from Environments import Algo
from .EnvType import EnvType
from utils.utils import unwrap_env

class Highway(EnvType):
    def __init__(self, algo: Algo):
        algo_param = {
            "policy": "MlpPolicy",
            "policy_kwargs": dict(net_arch=[256, 256]),
            "learning_rate": 5e-4,
            "buffer_size": 15000,
            "learning_starts": 200,
            "batch_size": 32,
            "gamma": 0.8,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 50,
            "verbose": 1,
            "tensorboard_log": "model/highway_dqn/",
        }
        prompt = {
        "",
    	}
        super().__init__(algo, algo_param, prompt)

    def __repr__(self):
        return "highway-v0"

    def success_func(self, env: gym.Env, info: dict) -> tuple[bool, bool]:
        """
        Vérifie si le véhicule a atteint une vitesse élevée sans collision.
        """
        base_env = unwrap_env(env)
        speed = info.get('speed', 0)
        crashed = info.get('crashed', False)
        print(f"speed: {speed}, crashed: {crashed}")
        if speed >= 30 and not crashed:
            return True, False
        else:
            return False, True

    def objective_metric(self, states) -> list[dict[str, float]]:
        """
        Calcule une métrique objective basée sur les états.
        """
        # Implémentation spécifique à votre cas d'utilisation
        pass  # TODO

    def get_prompt(self):
        pass # TODO