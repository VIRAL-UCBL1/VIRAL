def unwrap_env(env):
    """
    Fonction récursive pour déballer les wrappers Gym jusqu'à l'environnement de base.
    """
    while hasattr(env, "env"):  # check if env is a wrapper
        env = unwrap_env(env.env)
    return env