from Environments import (CartPole, Highway, Hopper, LunarLander,
                          Swimmer)
from LLM.LLMOptions import llm_options
from log.log_config import init_logger
from VIRAL import VIRAL
init_logger("DEBUG")

def runs(
    total_timesteps: int,
    nb_vec_envs: int,
    nb_refined: int,
    human_feedback: bool,
    video_description: bool,
    legacy_training: bool,
    actor_model: str,
    critic_model: str,
    env: str,
    observation_space: str,
    goal: str,
    image: str,
    nb_gen: int,
    nb_runs: int,
    proxies: dict,
    focus: str = "",
):
    """help wrapper for launch several runs

    Args:
        total_timesteps (int): 
        nb_vec_envs (int): 
        nb_refined (int): 
        human_feedback (bool): 
        video_description (bool): 
        legacy_training (bool): 
        actor_model (str): 
        critic_model (str): 
        env (str): 
        observation_space (str): 
        goal (str): 
        image (str): 
        nb_gen (int): 
        nb_runs (int): 
        proxies (dict): 
        focus (str, optional): . Defaults to "".
    """
    switcher = {
        "Cartpole": CartPole,
        "LunarLander": LunarLander,
        "Highway": Highway,
        "Swimmer": Swimmer,
        "Hopper": Hopper,
    }
    instance = switcher[env]()
    if observation_space != "":
        instance.prompt["Observation Space"] = observation_space
    if goal is not None:
        instance.prompt["Goal"] = goal
    else:
        instance.prompt.pop("Goal", None)
    if image is not None:
        instance.prompt["Image"] = image
    else:
        instance.prompt.pop("Image", None)
    def run():
        viral = VIRAL(
            env_type=instance,
            model_actor=actor_model,
            model_critic=critic_model,
            hf=human_feedback,
            vd=video_description,
            nb_vec_envs=nb_vec_envs,
            options=llm_options,
            legacy_training=legacy_training,
            training_time=total_timesteps,
            proxies=proxies,
        )
        viral.generate_context()
        viral.generate_reward_function(nb_gen, nb_refined, focus)
        viral.policy_trainer.start_vd(viral.memory[1].policy, 1)

    for r in range(nb_runs):
        print(f"#######  {r}  ########")
        run()
        
proxies = { 
	"http"  : "socks5h://localhost:1080", 
	"https" : "socks5h://localhost:1080", 
}

obs_space = """Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], 
[ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
The state is an 8-dimensional vector: 
the coordinates of the lander in x & y, 
its linear velocities in x & y, 
its angle, its angular velocity, 
and two booleans that represent whether each leg is in contact with the ground or not.
"""
goal = "Land without crashing and using minimum fuel on the landing pad at coordinates (0,0)"

runs(
    total_timesteps=30_000,
    nb_vec_envs=1,
    nb_refined=2,
    human_feedback=False,
    video_description=True,
    legacy_training=False,
    actor_model="qwen2.5-coder:32b",
    critic_model="llama3.2-vision",
    env="LunarLander",
    observation_space=obs_space,
    goal=goal,
    image=None,
    nb_gen=1,
    nb_runs=10,
    proxies=proxies,
)
