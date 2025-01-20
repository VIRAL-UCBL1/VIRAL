import random
import os
from logging import getLogger

from Environments import EnvType
from LLM.OllamaChat import OllamaChat
from log.LoggerCSV import LoggerCSV
from State.State import State
from LLM.GenCode import GenCode
from LLM.ClientVideoLVLM import ClienVideoLVLM
from PolicyTrainer.PolicyTrainer import PolicyTrainer


class VIRAL:
    def __init__(
        self,
        env_type: EnvType,
        model_actor: str,
        model_critic: str,
        hf: bool = False,
        vd: bool = False,
        seed: int = None,
        training_time: int = 25000,
        nb_vec_envs: int = 1,
        legacy_training: bool = True,
        options: dict = {},
        proxies: dict = None,
    ):
        """
        Initialize VIRAL architecture for dynamic reward function generation
            Args:
                env_type (EnvType): refer to parameter of an gym Env
                model_actor (str): Language model for reward generation
                hf (bool, optional): active the human feedback
                training_time (int, optional): timeout for model.learn()
                options (dict, optional): options for the llm
        """
        if seed is None:
            options["seed"] = random.randint(0, 1000000)
        model_name = model_actor if model_actor == model_critic or model_critic is None else model_actor+'_'+model_critic
        LoggerCSV(env_type, model_name, training_time)
        self.llm_actor = OllamaChat(
            model=model_actor,
            system_prompt="""
        You're a reinforcement learning expert specializing in the design of python reward functions.
        Strict criteria:
        1. Take care of Generate ALWAYS DIFFERENTS reward function per Response iteration
        2. Complete ONLY the reward function code
        3. Give no additional explanations
        4. STOP immediately your completion after the last return
        5. Assuming Numpy already imported as np
        6. Take into the observation of the state, the is_success boolean flag, the is_failure boolean flag
        """,
            options=options.copy(),
            proxies=proxies
        )
        self.llm_critic = OllamaChat(
            model=model_critic,
            system_prompt=f"""
        You're a reinforcement learning expert and assistant in rewarding for the {env_type} environment.
        As a critic, you're going to explains step by step, how to achieve the goal: {env_type.prompt['Goal']}.
        If you're reading an image, please use what you see, as a grounding, as a link to the state.
        The image contain red trajectory, the agent need to be identify and the trajectory needs to be precisely described.
        Every response you made, begin with the title '# HELP'
            """,
            options=options.copy(),
            proxies=proxies
        )
        self.hf = hf
        self.vd = vd
        if self.vd:
            self.client_video = ClienVideoLVLM(proxies)
        self.env_type: EnvType = env_type
        self.gen_code: GenCode = GenCode(self.env_type, self.llm_actor)
        self.logger = getLogger("VIRAL")
        self.memory: list[State] = [State(0)]
        self.policy_trainer: PolicyTrainer = PolicyTrainer(
            self.memory, options['seed'], self.env_type, timeout=training_time, nb_vec_envs=nb_vec_envs, legacy_training=legacy_training
        )

    def generate_context(self):
        """Generate more contexte for Step back prompting"""
        prompt = f"{self.env_type.prompt['Observation Space']}\n"
        prompt += f"Please, Describe the red trajectory an the observations for the following goal: \n{self.env_type.prompt['Goal']}."
        if "Image" in self.env_type.prompt.keys():
            self.llm_critic.add_message(prompt, images=[self.env_type.prompt["Image"]])
            self.llm_actor.add_message(prompt, images=[self.env_type.prompt["Image"]])
        else:
            self.llm_critic.add_message(prompt)
            self.llm_actor.add_message(prompt)
        response = self.llm_critic.generate_response(stream=True)
        response = self.llm_critic.print_Generator_and_return(response, -1)
        self.llm_actor.add_message(response)

    def generate_reward_function(
        self, n_init: int = 2, n_refine: int = 1, focus: str = ""
    ) -> list[State]:
        """
        Generate and iteratively improve a reward function using a Language Model (LLM).

        This method implements a sophisticated reward function generation process
        that involves multiple stages of creation, evaluation, and refinement.

        Key Stages:
            1. Initial Function Generation: Create two initial reward function candidates
            2. Evaluation: Compare and identify the best and worst performing functions
            3. Iterative Refinement: Progressively improve the worst-performing function

        Args:
            task_description (str): A detailed description of the task or environment
                                    for which the reward function is being generated.
            iterations (int, optional): Number of refinement iterations to perform.
                                        Defaults to 1.

        Returns:
            list[State]: A list of generated and refined reward function states,
                        containing information about each function's performance
                        and implementation.

        Process Overview:
            - Generates two initial reward functions using an LLM
            - Evaluates these functions using a policy evaluation method
            - Selects the worst-performing function for refinement
            - Iteratively refines the function through self-refinement
            - Tracks the evolution of reward functions in the memory

        Detailed Workflow:
            1. Generate two initial reward functions
                - Uses a predefined prompt template
                - Applies configurable LLM generation options
                - Compiles and tests each generated function
            2. Evaluates initial functions
                - Identifies best and worst performing functions
            3. Iterative Refinement
                - Applies self-refinement to the worst-performing function
                - Re-evaluates after each refinement
                - Repeats for specified number of iterations

        Note:
            - Uses dynamic LLM configuration options
            - Supports flexible environment types
            - Provides a systematic approach to reward function generation
            - Logging at various stages for debugging and tracking
        """
        ### INIT STAGE ###
        for i in range(1, n_init + 1):
            prompt = f"""Iteration {i}/{n_init},
            {focus}
        Complete this sentence using the HELP section as a guide:
        def reward_func(observations:np.ndarray, is_success:bool, is_failure:bool) -> float:
            \"\"\"Reward function for {self.env_type}

            Args:
                observations (np.ndarray): observation on the current state
                is_success (bool): True if the goal is achieved, False otherwise
                is_failure (bool): True if the episode ends unsuccessfully, False otherwise

            Returns:
                float: The reward for the current step
            \"\"\"
        """
            self.llm_actor.add_message(prompt)
            response = self.llm_actor.generate_response(stream=True)
            response = self.llm_actor.print_Generator_and_return(
                response, len(self.memory) - 1
            )
            state: State = self.gen_code.get(response) #TODO if  response doesn't work the chat is stuck and regenate the same response over and over
            self.memory.append(state)
            self.policy_trainer.start_learning(state.idx)

        are_worsts, are_betters, threshold = self.policy_trainer.evaluate_policy(
            range(1, n_init + 1)
        )
        ### SECOND STAGE ###
        for _ in range(n_refine):
            if are_worsts == []:
                break
            self.logger.debug(f"states to refines: {are_worsts}")
            news_idx: list[int] = []
            for worst_idx in are_worsts:
                # if self.memory[worst_idx].performances["sr"] < threshold - 0.2:
                    # news_idx.append(self.critical_refine_reward(worst_idx))
                # else:
                news_idx.append(self.self_refine_reward(worst_idx))
            are_worsts, are_betters, _ = self.policy_trainer.evaluate_policy(news_idx)
        return self.memory

    def critical_refine_reward(self, idx: int) -> int:
        self.logger.warning("critical refine reward")
        critic_prompt = f"""Given that the previous function \n({self.memory[idx].reward_func_str})\n
        written was subpar: Success Rate = {self.memory[idx].performances['sr']},

        Additionally, we have gathered stats during the training of this function:
        {self.memory[idx].performances}
        1. Identify why it did not work so that we do not repeat the same mistakes.

        Restart from scratch and assume that the previous assistance was not the right approach:

        2. Describe which observation can achive the Goal:\n{self.env_type.prompt['Goal']}.
        """
        actor_prompt = f"""Given that the previous function \n({self.memory[idx].reward_func_str})\n
        written was subpar: Success Rate = {self.memory[idx].performances['sr']},
        Complete this sentence using the HELP section as a guide:

        def reward_func(observations:np.ndarray, is_success:bool, is_failure:bool) -> float:
            \"\"\"Reward function for {self.env_type}

            Args:
                observations (np.ndarray): observation on the current state
                is_success (bool): True if the goal is achieved, False otherwise
                is_failure (bool): True if the episode ends unsuccessfully, False otherwise

            Returns:
                float: The reward for the current step
            \"\"\"
        """
        if self.hf:
            critic_prompt = self.human_feedback(critic_prompt, idx)
        self.llm_critic.add_message(critic_prompt)
        refined_critic = self.llm_critic.generate_response(stream=True)
        refined_critic = self.llm_critic.print_Generator_and_return(
            refined_critic, len(self.memory) - 1
        )
        self.llm_actor.add_message(refined_critic)
        self.llm_actor.add_message(actor_prompt)
        refined_response = self.llm_actor.generate_response(stream=True)
        refined_response = self.llm_actor.print_Generator_and_return(
            refined_response, len(self.memory) - 1
        )
        state = self.gen_code.get(refined_response)
        self.memory.append(state)
        self.policy_trainer.start_learning(state.idx)
        return state.idx

    def self_refine_reward(self, idx: int) -> int:
        """
        Iteratively improve a reward function using self-refinement techniques.

        This method implements an intelligent self-refinement process for reward functions
        by leveraging a Language Model (LLM) to analyze and improve the current function
        based on its previous performance.

        Key Objectives:
            - Analyze current reward function performance
            - Generate an improved version of the reward function
            - Maintain the core task objectives while optimizing the reward signal

        Args:
            idx (int): Index of the reward function in the memory to be refined.
                    Typically the worst-performing function from previous evaluation.

        Returns:
            int: Index of the newly created refined reward function in the memory.

        Refinement Process:
            1. Construct a refinement prompt with:
                - Current reward function code
                - Performance metrics
                - Explicit refinement goals
            2. Generate a new reward function using LLM
            3. Compile and validate the new function
            4. Append the new function to memory
            5. Return the index of the new function

        Refinement Goals:
            - Increase success rate of the policy
            - Optimize the reward signal for better learning
            - Preserve the original task objectives
            - Improve overall performance

        Notes:
            - Uses the existing memory to track function evolution
            - Leverages LLM for intelligent function refinement
            - Provides a systematic approach to reward function improvement
            - Maintains a history of function iterations
        """# based on the provided performance metrics {self.memory[idx].performances}. 
        refinement_prompt = f"""Analyze the shortcomings of the current reward function {self.memory[idx].reward_func_str}
        base on the goal {self.env_type.prompt['Goal']}, identify specific issues that may have led to suboptimal performance.
        and propose a new reward function that addresses these issues.
        Include comments in the code to explain your reasoning and how the new function improves upon the previous one.
        """
        self.logger.debug(self.memory[idx].performances)
        if self.hf:
            refinement_prompt = self.human_feedback(refinement_prompt, idx)
        if self.vd:
            refinement_prompt = self.video_description(refinement_prompt, idx)
        self.llm_actor.add_message(refinement_prompt)
        refined_response = self.llm_actor.generate_response(stream=True)
        refined_response = self.llm_actor.print_Generator_and_return(
            refined_response, len(self.memory) - 1
        )
        state = self.gen_code.get(refined_response)
        self.memory.append(state)
        self.policy_trainer.start_learning(state.idx)
        return state.idx

    def human_feedback(self, prompt: str, idx: int) -> str:
        """implement human feedback

        Args:
            prompt (str): user prompt
            idx (int): state.idx to refine

        Returns:
            str: return the modified prompt
        """
        self.logger.info(self.memory[idx])
        visualise = input("do you need to visualise policy ?\ny/n:")
        if visualise.lower() in ["y", "yes"]:
            self.policy_trainer.start_hf(self.memory[idx].policy, 2)
        feedback = input("add a comment, (press enter if you don't have one)\n:")
        if feedback:
            prompt = feedback + "\n" + prompt
        return prompt

    def video_description(self, prompt:str,  idx: str):
        if self.client_video is None:
            self.logger.error("client video not initialised")
            raise RuntimeError("client video not initialised")
        self.policy_trainer.start_vd(self.memory[idx].policy, 1)
        video_path = os.path.join("records", str(self.env_type), "rl-video-episode-0.mp4")
        self.logger.info(f"video safe at: {video_path}")
        video_prompt = """In this video, an object is in motion. 
        Describe only the movement of the object, focusing on the dynamics of its movement. 
        Specify the precise direction in which it is moving (e.g., forward, backward, diagonally, etc.) 
        and detail the characteristics of this movement 
        (speed, acceleration, trajectory, oscillation, rotation, etc.). 
        Also, mention any changes in direction or rhythm, 
        as well as any specific actions the object performs during its movement. 
        Avoid discussing the background or color"""
        response = self.client_video.generate_simple_response(video_prompt, video_path)
        self.logger.info(f"description of the video: \n {response}")
        if response:
            prompt = response + "\n" + prompt
        return prompt

    def test_reward_func(self, reward_func: str) -> None:
        state: State = self.gen_code.get(reward_func)
        self.memory.append(state)
        self.policy_trainer.start_learning(state.idx)
        are_worsts, are_betters, threshold = self.policy_trainer.evaluate_policy([state.idx])
        