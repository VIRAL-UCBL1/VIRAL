import gradio as gr


import argparse
import json
from logging import getLogger
from Environments import Algo, CartPole, Highway, Hopper, LunarLander, Swimmer
from LLM.LLMOptions import llm_options
from log.log_config import init_logger
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
        init_logger()

    return getLogger()


def runs(
    total_timesteps: int,
    nb_vec_envs: int,
    nb_refined: int,
    refined_ctrl: list[str],
    legacy_training: bool,
    actor_model: str,
    critic_model: str,
    env: str,
    observation_space: str,
    goal: str,
    image: None,
    nb_gen: int,
    nb_runs: int,
    proxies: str,
):
    parse_logger()
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
    if goal != "":
        instance.prompt["Goal"] = goal
    human_feedback = False
    video_description = False
    if 'human feedback' in refined_ctrl:
        human_feedback = True
    if 'Video Description' in refined_ctrl:
        video_description = True
    # TODO image
    proxies_dict = None
    if proxies != "":
        proxies_dict = json.loads(proxies)

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
            proxies=proxies_dict,
        )
        viral.generate_context()
        viral.generate_reward_function(nb_gen, nb_refined)  # TODO focus

    for _ in range(nb_runs):
        run()
    return 


demo = gr.Interface(
    runs,
    [
        gr.Slider(20000, 1000000, value=50000, label="total_timesteps"),
        gr.Slider(1, 8, value=1, step=1, label="number of vec envs"),
        gr.Slider(0, 3, value=1, step=1, label="number of refined loop"),
        gr.CheckboxGroup(
            ["human feedback", "Video Description"], label="Refined controls"
        ),
        gr.Checkbox(label="legacy training"),
        gr.Dropdown(
            ["qwen2.5-coder", "qwen2.5-coder:14b", "qwen2.5-coder:32b"],
            value="qwen2.5-coder:32b",
            label="Model Actor",
            info="The LLM witch produce the code",
        ),
        gr.Dropdown(
            [
                "qwen2.5-coder",
                "qwen2.5-coder:14b",
                "qwen2.5-coder:32b",
                "llama3.2-vision",
            ],
            value="llama3.2-vision",
            label="Model Critic",
            info="The LLM witch help to understand the request, for vision, please select llama3.2-vision",
        ),
        gr.Dropdown(
            ["Cartpole", "LunarLander", "Highway", "Swimmer", "Hopper"],
            label="Environment",
        ),
        gr.Textbox(label="Observation Space"),
        gr.Textbox(label="Goal Prompt"),
        gr.Image(label="Annotated Image of the Env"),
        gr.Slider(1, 4, value=1, step=1, label="number of gens per runs"),
        gr.Slider(1, 100, value=1, step=1, label="number of runs"),
        gr.Textbox(
            value="""{ 
        "http"  : "socks5h://localhost:1080", 
        "https" : "socks5h://localhost:1080", 
}""",
            label="optionnal proxies",
        ),
    ],
    "text",
    title="Vision-grounded Integration for Reward design And Learning",
)

if __name__ == "__main__":
    demo.launch()
