"""
Config file for the LLM model parameters.

Args:
    mirostat (int): Enables Mirostat sampling for controlling perplexity 
        (default: 0). 
        - 0: Disabled
        - 1: Mirostat
        - 2: Mirostat 2.0

    mirostat_eta (float): Adjusts the responsiveness of the Mirostat algorithm 
        (default: 0.1). 

    mirostat_tau (float): Balances coherence and diversity in the output 
        (default: 5.0). 

    num_ctx (int): Sets the size of the context window for token generation 
        (default: 2048). 

    repeat_last_n (int): Specifies the range to prevent repetition 
        (default: 64). 
        - 0: Disabled
        - -1: Uses num_ctx

    repeat_penalty (float): Penalizes repeated tokens in generation 
        (default: 1.1). 

    temperature (float): Adjusts creativity in the model's responses 
        (default: 0.8). 

    seed (int): Random seed for deterministic outputs 
        (default: 0). 

    stop (str): Stop sequences to terminate generation 
        (default: None). 

    tfs_z (float): Enables Tail Free Sampling to reduce less probable tokens 
        (default: 1.0). 

    num_predict (int): Maximum tokens to predict in generation 
        (default: -1). 

    top_k (int): Limits probability space for token selection 
        (default: 40). 

    top_p (float): Controls cumulative probability for token selection 
        (default: 0.9). 

    min_p (float): Ensures tokens meet a minimum probability threshold relative to the most likely token 
        (default: 0.0). 
"""

additional_options = {
        # "temperature": 1,
        # "num_predict": 3, # l'impression que ça change rien a creuser
        # "mirostat" : 1,
        # "mirostat_eta" : 0.01, #gère la vitesse de réponses du model (0.1 par défaut) plus c'est petit plus c'est lent
        # "mirostat_tau" : 4.0, #gère la balance entre la diversité et la coherence des réponses (5.0 par défaut) plus c'est petit plus c'est focus et cohérent
        # num_ctx": 2048, # nombre de tokens contextuels (2048 par défaut peut être pas nécessaire de changer)
        # "repeat_last_n": 128, # combien le model regarde en arrière pour éviter de répéter les réponses (64 par défaut large pour nous)
        # "repeat_penalty": 1.3, # pénalité pour éviter de répéter les réponses (1.1 par défaut au mac 1.5 intéressant a modificer je pense)
        # "stop": "stop you here" # pour stopper la génération de texte pas intéressant pour nous
        # "tfs_z": 1.2, #reduire l'impacte des token les moins "pertinents" (1.0 par défaut pour désactiver 2.0 max)
        # "top_k": 30, #reduit la probabilité de générer des non-sens (40 par défaut, 100 pour générer des réponses plus diverses, 10 pour des réponses plus "conservatrices")
        # "top_p": 0.95, #marche avec le top_k une forte valeur pour des texte plus diverses (0.9 par défaut)
        # "min_p": 0.05, #alternative au top_p, vise a s'aéssurer de la balance entre qualité et diversité (0.0 par défaut)
        # "seed": 211042, # a utiliser pour la reproductibilité des résultats (important si publication)
}