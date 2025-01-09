import random
from logging import getLogger

from Environments import EnvType
from LLM.OllamaChat import OllamaChat
from log.LoggerCSV import LoggerCSV
from State.State import State
from LLM.GenCode import GenCode
from PolicyTrainer.PolicyTrainer import PolicyTrainer


class VIRAL:
    def __init__(
        self,
        env_type: EnvType,
        model: str,
        hf: bool = False,
        training_time: int = 25000,
        numenvs: int = 2,
        options: dict = {},
    ):
        """
        Initialize VIRAL architecture for dynamic reward function generation
            Args:
                env_type (EnvType): refer to parameter of an gym Env
                model (str): Language model for reward generation
                hf (bool, optional): active the human feedback
                training_time (int, optional): timeout for model.learn()
                options (dict, optional): options for the llm 
        """
        if options.get("seed") is None:
            options["seed"] = random.randint(0, 1000000)
        LoggerCSV(env_type, model)
        self.llm = OllamaChat(
            model=model,
            system_prompt=f"""
        You are an expert in Reinforcement Learning specialized in designing reward functions.
        Strict criteria:
        1. Take care of Generate ALWAYS DIFFERENTS reward function per Iterations
        2. Complete ONLY the reward function code
        3. Give no additional explanations
        4. STOP immediately your completion after the last return
        5. Focus on the {env_type} environment
        6. Assuming Numpy already imported as np
        7. Take into the observation of the state, the is_success boolean flag, the is_failure boolean flag
        """,
            options=options,
        )
        self.hf = hf
        self.env_type: EnvType = env_type
        self.gen_code: GenCode = GenCode(self.env_type, self.llm)
        self.logger = getLogger("VIRAL")
        self.memory: list[State] = [State(0)]
        self.policy_trainer: PolicyTrainer = PolicyTrainer(
            self.memory, self.env_type, timeout=training_time, numenvs=numenvs
        )

    def generate_context(self, prompt_info: dict):
        """Generate more contexte for Step back prompting

        Args:
            prompt_info (dict): contain a task, and observation space
        """
        prompt = f"{prompt_info}\nDescribe which observation can achive the Goal:\n{prompt_info['Goal']}."
        sys_prompt = (
            f"You're a physics expert, specializing in {self.env_type} motion analysis.\n"
            + "you can refer to some laws of physics \n"
            + "Don't explain obvious thinks, you talk to an expert \n"
            + "be Concise, short and begin your explaination with: CONTEXT:"
        )
        self.llm.add_message(prompt)
        response = self.llm.generate_simple_response(prompt, sys_prompt, stream=True)
        response = self.llm.print_Generator_and_return(response, -1)

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
        for i in range(1, n_init + 1):  # TODO make it work for 4_init
            prompt = f"""Iteration {i}/{n_init},
            {focus}
        Complete this sentence using the CONTEXT section as a guide:
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
            self.llm.add_message(prompt)
            response = self.llm.generate_response(stream=True)
            response = self.llm.print_Generator_and_return(response, len(self.memory)-1) #TODO if  response doesn't work the chat is stuck and regenate the same response over and over (je pensais qu'on avais fix mais apparament pas)
            state: State = self.gen_code.get(response)
            self.memory.append(state)

        are_worsts, are_betters, threshold = self.policy_trainer.evaluate_policy(range(1, n_init + 1))
        ### SECOND STAGE ###
        for _ in range(n_refine):
            if are_worsts == []:
                break
            self.logger.debug(f"states to refines: {are_worsts}")
            news_idx = self.self_refine_reward(are_worsts)
            are_worsts, are_betters, _ = self.policy_trainer.evaluate_policy(news_idx)
        return self.memory

    def self_refine_reward(self, list_idx: list[int]) -> list[int]:
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
        """
        news_idx: list[int] = []
        for idx in list_idx:
            refinement_prompt = f"""
            refine the reward function to:
            - Increase success rate
            - Optimize reward signal
            - Maintain task objectives

            previous performance:
            {self.memory[idx].performances}

            reward function to refine:
            {self.memory[idx].reward_func_str}
            """
            self.logger.debug(self.memory[idx].performances)
            if self.hf:
                refinement_prompt = self.human_feedback(refinement_prompt, idx)
            self.llm.add_message(refinement_prompt)
            refined_response = self.llm.generate_response(stream=True)
            refined_response = self.llm.print_Generator_and_return(refined_response, len(self.memory) - 1)
            state = self.gen_code.get(refined_response)
            self.memory.append(state)
            news_idx.append(len(self.memory) - 1)
        return news_idx

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
            self.policy_trainer.test_policy_hf(self.memory[idx].policy)
        feedback = input("add a comment, (press enter if you don't have one)\n:")
        if feedback:
            prompt = feedback + "\n" + prompt
        return prompt

    def test_reward_func(self, reward_func: str):
        state: State = self.gen_code.get(reward_func)
        self.memory.append(state)
        are_worsts, are_betters, threshold = self.policy_trainer.evaluate_policy([state.idx])

