def unwrap_env(env):
    """usefull  fonction for retrive private env attributes"""
    while hasattr(env, "env"):  # check if env is a wrapper
        env = unwrap_env(env.env)
    return env