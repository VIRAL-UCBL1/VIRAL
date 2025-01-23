def unwrap_env(env):
    """
    Unwraps a gym environment to get the base env.
    """
    while hasattr(env, "env"):  # check if env is a wrapper
        env = unwrap_env(env.env)
    return env