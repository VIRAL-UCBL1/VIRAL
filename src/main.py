import argparse
from logging import getLogger

from log.log_config import init_logger
from log.LoggerCSV import LoggerCSV
from RLAlgo.DirectSearch import DirectSearch
from RLAlgo.Reinforce import Reinforce
from Environments import Prompt, Algo, CartPole, LunarLander
from VIRAL import VIRAL

def parse_logger():
    """
    Parses command-line arguments to configure the logger.

    Returns:
        Logger: Configured logger instance.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    args = parser.parse_args()

    if args.verbose:
        init_logger("DEBUG")
        print("Verbose mode enabled")
    else:
        init_logger("DEBUG")

    return getLogger()

def main():
    """
    Main entry point of the script.

    This block is executed when the script is run directly. It initializes the
    logger, and run VIRAL. It uses CLI interface.
    memory.
    """
    logger = parse_logger()
    additional_options = {
            "temperature": 1,
            # "num_predict": 3, # l'impression que ça change rien a creuser
            # "mirostat" : 1,
            # "mirostat_eta" : 0.01, #gère la vitesse de réponses du model (0.1 par défaut) plus c'est petit plus c'est lent
            # "mirostat_tau" : 4.0, #gère la balance entre la diversité et la coherence des réponses (5.0 par défaut) plus c'est petit plus c'est focus et cohérent
            # num_ctx": 2048, # nombre de tokens contextuels (2048 par défaut peut être pas nécessaire de changer)
            # repeat_last_n": 64, # combien le model regarde en arrière pour éviter de répéter les réponses (64 par défaut large pour nous)
            # "repeat_penalty": 1.5, # pénalité pour éviter de répéter les réponses (1.1 par défaut au mac 1.5 intéressant a modificer je pense)
            # "stop": "stop you here" # pour stopper la génération de texte pas intéressant pour nous
            # "tfs_z": 1.2, #reduire l'impacte des token les moins "pertinents" (1.0 par défaut pour désactiver 2.0 max)
            # "top_k": 30, #reduit la probabilité de générer des non-sens (40 par défaut, 100 pour générer des réponses plus diverses, 10 pour des réponses plus "conservatrices")
            # "top_p": 0.95, #marche avec le top_k une forte valeur pour des texte plus diverses (0.9 par défaut)
            # "min_p": 0.05, #alternative au top_p, vise a s'aéssurer de la balance entre qualité et diversité (0.0 par défaut)
            # "seed": 42, # a utiliser pour la reproductibilité des résultats (important si publication)
        }
    env_type = CartPole(Algo.PPO)
    LoggerCSV(env_type, 'qwen2.5-coder')
    viral = VIRAL(
        env_type=env_type, options=additional_options)
    res = viral.generate_reward_function(
        task_description=Prompt.CARTPOLE,
        iterations=2,
    )
    for state in viral.memory:
        logger.info(state)

if __name__ == "__main__":
    main()