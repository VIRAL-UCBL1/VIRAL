import random
from logging import getLogger
from typing import Callable

from Environments.Algo import Algo
from Environments.Environments import Environments
from LLM.OllamaChat import OllamaChat
from State.State import State
from LLM.GenCode import GenCode
from PolicyTrainer.PolicyTrainer import PolicyTrainer


class VIRAL:
    def __init__(
        self,
        learning_algo: Algo,
        env_type: Environments,
        success_function: Callable,
        objectives_metrics: list[callable] = [],
        model: str = "qwen2.5-coder",
        options: dict = {},
    ):
        """
        Initialize VIRAL architecture for dynamic reward function generation
            Args:
                model (str): Language model for reward generation
                learning_method (str): Reinforcement learning method
        """
        if options.get("seed") is None:
            options["seed"] = random.randint(0, 1000000)

        self.llm = OllamaChat(
            model=model,
            system_prompt="""
        You are an expert in Reinforcement Learning specialized in designing reward functions.
        Strict criteria:
        - Complete ONLY the reward function code
        - Use Python format
        - Give no additional explanations
        - Focus on the Gymnasium environment 
        - Take into the observation of the state, the terminated and truncated boolean
        - STOP immediately your completion after the last return
        """,
            options=options,
        )
        self.env_type: Environments = env_type
        self.gen_code: GenCode = GenCode(self.env_type, self.llm)
        self.success_function = success_function
        self.objectives_metrics = objectives_metrics
        self.learning_algo: Algo = learning_algo
        self.logger = getLogger("VIRAL")
        self.memory: list[State] = [State(0)]
        self.policy_trainer: PolicyTrainer = PolicyTrainer(
            self.memory, self.learning_algo, self.env_type,
            self.success_function
        )

    def generate_reward_function(
        self, task_description: str, iterations: int = 1
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
        # TODO Pourquoi additional_options ici et pas dans le constructeur ?
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
        ### INIT STAGE ###
        for i in [1, 2]:
            prompt = f"""
        Complete the reward function for a {self.env_type.value} environment.
        Task Description: {task_description} Iteration {i+1}/{2}

        complete this sentence:
        def reward_func(observations:np.ndarray, terminated: bool, truncated: bool) -> float:
            \"\"\"Reward function for {self.env_type.value}

            Args:
                observations (np.ndarray): observation on the current state
                terminated (bool): episode is terminated due a failure
                truncated (bool): episode is truncated due a success

            Returns:
                float: The reward for the current step
            \"\"\"
        """
            self.llm.add_message(prompt)
            response = self.llm.generate_response(
                stream=True, additional_options=additional_options
            )
            self.logger.info(f"additional options: {additional_options}")
            response = self.llm.print_Generator_and_return(response, i)
            state: State = self.gen_code.get(response)
            self.memory.append(state)

        best_idx, worst_idx = self.policy_trainer.evaluate_policy(1, 2)
        self.logger.debug(f"state to refine: {worst_idx}")
        ### SECOND STAGE ###
        for n in range(iterations - 1):
            new_idx = self.self_refine_reward(worst_idx)
            best_idx, worst_idx = self.policy_trainer.evaluate_policy(best_idx, new_idx)
            self.logger.debug(f"state to refine: {worst_idx}")
        return self.memory

    def self_refine_reward(self, idx: int) -> Callable:
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
        refinement_prompt = f"""
        improve the reward function to:
        - Increase success rate
        - Optimize reward signal
        - Maintain task objectives

        your best reward function:
        {self.memory[idx].reward_func_str}

        performance:
        {self.memory[idx].performances}
        """

        self.llm.add_message(refinement_prompt)
        refined_response = self.llm.generate_response(stream=True)
        refined_response = self.llm.print_Generator_and_return(refined_response)
        state = self.gen_code.get(refined_response)
        self.memory.append(state)

        return len(self.memory) - 1
