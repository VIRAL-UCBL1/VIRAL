import argparse
from logging import getLogger

from Environments import (Algo, CartPole, Highway, Hopper, LunarLander, Pacman,
                          Swimmer)
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


def main():
    """
    Main entry point of the script.

    This block is executed when the script is run directly. It initializes the
    logger, and run VIRAL. It uses CLI interface.
    memory.
    """
    parse_logger()
    # env_type = LunarLander(
    #     prompt={
    #         "Goal": "land the lander along the red trajectory shown in the image",
    #         "Observation Space": """Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
    #             The state is an 8-dimensional vector: 
    #             the coordinates of the lander in x & y,
    #             its linear velocities in x & y, 
    #             its angle, its angular velocity, 
    #             and two booleans that represent whether each leg is in contact with the ground or not.
    #         """,
    #         "Image": "Environments/img/LunarLander.png"
    #     }
    # )
     #env_type = Hopper(
    #    prompt={
    #        "Goal": "make a backflip",
    #        "Observation Space": """Box(-inf, inf, (11,), float64)
#
#            The observation space consists of the following parts (in order):
#            qpos (5 elements by default): Position values of the robot’s body parts.
 #           qvel (6 elements): The velocities of these individual body parts (their derivatives).
#            the x- and y-coordinates are returned in info with the keys "x_position" and "y_position", respectively.
#
#            | Num      | Observation                                      | Min   | Max  | Type                |
#            |----------|--------------------------------------------------|-------|------|---------------------|
#            | 0        | z-coordinate of the torso (height of hopper)     | -Inf  | Inf  | position (m)        |
#            | 1        | angle of the torso                               | -Inf  | Inf  | angle (rad)         |
#            | 2        | angle of the thigh joint                         | -Inf  | Inf  | angle (rad)         |
 #           | 3        | angle of the leg joint                           | -Inf  | Inf  | angle (rad)         |
#            | 4        | angle of the foot joint                          | -Inf  | Inf  | angle (rad)         |
#            | 5        | velocity of the x-coordinate of the torso        | -Inf  | Inf  | velocity (m/s)      |
#            | 6        | velocity of the z-coordinate (height) of torso   | -Inf  | Inf  | velocity (m/s)      |
#            | 7        | angular velocity of the angle of the torso       | -Inf  | Inf  | angular velocity (rad/s) |
#            | 8        | angular velocity of the thigh hinge              | -Inf  | Inf  | angular velocity (rad/s) |
#            | 9        | angular velocity of the leg hinge                | -Inf  | Inf  | angular velocity (rad/s) |
#            | 10       | angular velocity of the foot hinge               | -Inf  | Inf  | angular velocity (rad/s) |
#            | excluded | x-coordinate of the torso                        | -Inf  | Inf  | position (m)        |
#            """,
#            "Image": "Environments/img/Hopper_BackFlip.png"
#        })
    env_type = Swimmer()
    actor = "qwen2.5-coder:32b"
    critic = "llama3.2-vision"
    proxies = { 
        "http"  : "socks5h://localhost:1080", 
        "https" : "socks5h://localhost:1080", 
    }
    viral = VIRAL(
        env_type=env_type,
        model_actor=actor,
        model_critic=critic,
        hf=False,
        nb_vec_envs=2,
        options=llm_options,
        legacy_training=False,
        training_time=500_000,
        proxies=proxies
    )
    viral.generate_context()
    viral.generate_reward_function(n_init=2, n_refine=2)
    for state in viral.memory:
        viral.logger.info(state)


if __name__ == "__main__":
    for i in range(20):
        main()
