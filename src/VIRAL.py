import random
import signal
import sys
from multiprocessing import Process, Queue
from queue import Empty
from logging import getLogger
from time import sleep
from typing import Callable, Dict, List
from venv import logger

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils.Algo import Algo
from utils.CustomRewardWrapper import CustomRewardWrapper
from utils.Environments import Environments
from utils.OllamaChat import OllamaChat
from utils.State import State
from utils.TrainingInfoCallback import TrainingInfoCallback
import os


class VIRAL:
    def __init__(
        self,
        learning_algo: Algo,
        env_type : Environments,
        success_function: Callable,
        objectives_metrics: List[callable] = [],
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
        self.env_type : Environments = env_type
        self.success_function = success_function
        self.env = None
        self.objectives_metrics = objectives_metrics
        self.learning_algo: Algo = learning_algo
        self.learning_method = None
        self.logger = getLogger("VIRAL")
        # self._learning(self.memory[0])

        if os.name == "posix":
            self.queue = Queue()
            self.memory: List[State] = [State(0)]
            self.multi_process: list[Process] = []
            self.multi_process.append(
                Process(
                    target=self._learning,
                    args=(
                        self.memory[0],
                        self.queue,
                    ),
                )
            )
            self.multi_process[0].start()
        else:
            self.memory: List[State] = [State(0)]


    def generate_reward_function(
        self, task_description: str, iterations: int = 1
    ) -> List[State]:
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
            List[State]: A list of generated and refined reward function states, 
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
            reward_func, response = self.get_runnable_function(response)
            self.memory.append(State(i, reward_func, response))

        best_idx, worst_idx = self.evaluate_policy(1, 2)
        self.logger.debug(f"state to refine: {worst_idx}")
        ### SECOND STAGE ###
        for n in range(iterations - 1):
            new_idx = self.self_refine_reward(worst_idx)
            best_idx, worst_idx = self.evaluate_policy(best_idx, new_idx)
            self.logger.debug(f"state to refine: {worst_idx}")
        return self.memory

    def get_code(self, response: str) -> str:
        """
        Clean and validate a code response by removing code block markers and ensuring a function definition.

        This method is designed to process code responses, typically extracted from text or code blocks,
        by performing the following operations:\n
        1. Remove leading and trailing code block markers (```),
        2. Remove the 'python' language identifier,
        3. Strip any additional whitespace
        4. Validate that the response contains a function definition

        Args:
            response (str): The raw code response to be cleaned and validated.

        Returns:
            str: The cleaned code response containing a function definition.

        Raises:
            ValueError: If the response does not contain a valid function definition 
                        (i.e., if "def " is not present in the cleaned response).

        Logging:
            Logs the cleaned code at DEBUG level for debugging purposes.
    """
        cleaned_response = response.strip("```").replace("python", "").strip()
        if "def " not in cleaned_response:
            raise ValueError("The answer does not contain a valid function definition.")
        self.logger.debug("Code nettoyé pour compilation :\n" + cleaned_response)
        return cleaned_response

    def get_runnable_function(self, response: str, error: str = None) -> Callable:
        """
        Process and validate a reward function for a gym environment.

        This method attempts to generate and validate a reward function by:\n
        1. Handling potential previous errors
        2. Creating a gym environment
        3. Cleaning and compiling the code
        4. Testing the reward function with a sample action
        5. Recursively handling various potential errors

        Args:
            response (str): The code response containing the reward function definition.
            error (str, optional): Previous error message to be added to LLM context. 
                                    Defaults to None.

        Returns:
            tuple: A tuple containing:
                - Callable: The compiled and validated reward function
                - str: The original response code

        Raises:
            - ValueError: Invalid function definition
            - SyntaxError: Syntax issues in the function
            - RuntimeError: Execution problems during function testing
        
        Note:
            - Uses recursion to handle potential errors
            - Relies on get_code, compile_reward_function, and test_reward_function methods
            - Provides a robust mechanism for generating valid reward functions
    """
        if error is not None:
            self.llm.add_message(error)
            response = self.llm.generate_response(stream=True)
            response = self.llm.print_Generator_and_return(response)
        try:
            env = gym.make(self.env_type.value)
            response = self.get_code(response)
            reward_func = self.compile_reward_function(response)
            state, _ = env.reset()
            action = env.action_space.sample()
            next_observation, _, terminated, truncated, _ = env.step(action)
            self.test_reward_function(
                reward_func,
                observations=next_observation,
                terminated=terminated,
                truncated=truncated,
            )
        except ValueError as e:
            self.logger.warning(str(e))
            return self.get_runnable_function(response, str(e))
        except SyntaxError as e:
            self.logger.warning(f"Error syntax {e}")
            return self.get_runnable_function(response, str(e))
        except RuntimeError as e:
            self.logger.warning(f"Error execution {e}")
            return self.get_runnable_function(response, str(e))

        return reward_func, response

    def compile_reward_function(self, response: str) -> Callable:
        """
        Compile a reward function dynamically from a string response.

        This method takes a code string representing a reward function and dynamically 
        compiles it into an executable Python function. It provides a secure way to 
        generate reward functions for reinforcement learning environments.

        Key Features:
            - Dynamically executes code in an isolated global namespace
            - Provides access to NumPy functions
            - Extracts the compiled function by its name
            - Robust error handling for syntax issues

        Args:
            response (str): A string containing a complete Python function definition 
                            for a reward function.

        Returns:
            Callable: The compiled reward function that can be called with appropriate 
                    arguments in a gym environment.

        Raises:
            SyntaxError: If the provided code contains invalid Python syntax.
            ValueError: If the function cannot be extracted from the compiled namespace.

        Notes:
            - Uses `exec()` for dynamic code compilation
            - Provides NumPy (`np`) in the execution namespace
            - Assumes the last function defined in the response is the reward function
    """

        exec_globals = {}
        exec_globals["np"] = np
        try:
            exec(response, exec_globals)
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in the generated code : {e}")

        reward_function_name = response.split("(")[0].split()[
            -1
        ]  # récup le nom de la fonction
        reward_function = exec_globals.get(reward_function_name)

        return reward_function

    def test_reward_function(self, reward_function: Callable, *args, **kwargs):
        """
        Test the compiled reward function with provided inputs to validate its execution.

        This method serves as a crucial validation step in the reward function generation 
        process. It attempts to execute the reward function with the given arguments and 
        logs the output or raises an error if execution fails.

        Purpose:
            - Verify the reward function can be executed without errors
            - Log the reward function's output for debugging
            - Ensure the function returns a valid result in the context of a gym environment

        Args:
            reward_function (Callable): The compiled reward function to be tested.
            *args: Variable length argument list to pass to the reward function.
                Typically includes observations, actions, or environment states.
            **kwargs: Arbitrary keyword arguments to pass to the reward function.
                May include additional context like 'terminated' or 'truncated' flags.

        Raises:
            RuntimeError: If the reward function fails to execute successfully.
                This includes any exceptions that occur during function invocation.

        Logging:
            - Logs the reward function's output at DEBUG level when successful
            - Provides detailed error information if execution fails

        Notes:
            - Designed to be flexible with varying function signatures
            - Critical for validating dynamically generated reward functions
            - Part of the reward function generation quality control process
        """
        try:
            reward = reward_function(*args, **kwargs)
            self.logger.debug(f"Reward function output: {reward}")
        except Exception as e:
            raise RuntimeError(f"Error during reward function execution: {e}")

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
        reward_func, refined_response = self.get_runnable_function(refined_response)
        self.memory.append(State(len(self.memory), reward_func, refined_response))

        return len(self.memory) - 1

    def _learning(self, state: State, queue: Queue = None) -> None:
        """train a policy on an environment"""
        self.logger.debug(f"state {state.idx} begin is learning with reward function: {state.reward_func_str}")
        vec_env, model, numvenv = self._generate_env_model(state.reward_func)
        training_callback = TrainingInfoCallback()
        policy = model.learn(total_timesteps=60000, callback=training_callback)
        policy.save(f"model/policy{state.idx}.model")
        metrics = training_callback.get_metrics()
        self.logger.debug(f"TRAINING METRICS: {metrics}")
        sr_test = self.test_policy(vec_env, policy, numvenv)
        # ajoute au dict metrics les performances sans ecraser les anciennes
        metrics["test_success_rate"] = sr_test
        self.logger.debug(
            f"state {state.idx} as finished is learning with performances: {metrics}"
        )
        if os.name == "posix":
            queue.put([state.idx, f"model/policy{state.idx}.model", metrics])
        else:
            state.set_performances(metrics)
            self.memory[state.idx].set_policy(policy)

    def evaluate_policy(self, idx1: int, idx2: int) -> int:
        """
        Evaluate policy performance for multiple reward functions

        Args:
            objectives_metrics (List[callable]): Custom objective metrics
            num_episodes (int): Number of evaluation episodes

        Returns:
            Dict: Performance metrics for multiple reward functions
        """
        if os.name == "posix":
            if len(self.memory) < 2:
                self.logger.error("At least two reward functions are required.")
            to_get: int = 0
            to_join: list = []
            for i in [idx1, idx2]:
                if self.memory[i].performances is None:
                    self.multi_process.append(
                        Process(target=self._learning, args=(self.memory[i], self.queue))
                    )
                    self.multi_process[-1].start()
                    to_get += 1
                    to_join.append(len(self.multi_process)-1)

            while to_get != 0:
                try:
                    get = self.queue.get(block=False)
                    self.memory[get[0]].set_policy(get[1])
                    self.memory[get[0]].set_performances(get[2])
                    self.logger.debug(
                        f"state {get[0]} has finished learning with performances: {get[2]}"
                    )
                    to_get -= 1
                except Empty:
                    sleep(0.1)

            for p in to_join:
                self.multi_process[p].join()
            if (
                self.memory[idx1].performances["test_success_rate"]
                > self.memory[idx2].performances["test_success_rate"]
            ):
                return idx1, idx2
            else:
                return idx2, idx1
        else:
            if len(self.memory) < 2:
                self.logger.error("At least two reward functions are required.")
            for i in [idx1, idx2]:
                if self.memory[i].performances is None:
                    self._learning(self.memory[i])
            # TODO comparaison sur le success rate pour l'instant
            if (
                self.memory[idx1].performances["test_success_rate"]
                > self.memory[idx2].performances["test_success_rate"]
            ):
                return idx1, idx2
            else:
                return idx2, idx1

    def test_policy(
        self,
        env,
        policy,
        numvenv,
        nb_episodes=100,
    ) -> float:
        all_rewards = []
        nb_success = 0

        obs = env.reset()

        for _ in range(nb_episodes // numvenv):
            episode_rewards = np.zeros(numvenv)
            dones = [False] * numvenv
            
            while not all(dones):
                actions, _ = policy.predict(obs)
                obs, rewards, new_dones, infos = env.step(actions)
                episode_rewards += np.array(rewards)
                for i, (done, info) in enumerate(zip(new_dones, infos)):
                    if done:
                        dones[i] = True
                        if self.success_function(env.envs[i], info):
                            nb_success += 1
            
            all_rewards.extend(episode_rewards)

        success_rate = nb_success / nb_episodes
        return success_rate

    def _generate_env_model(self, reward_func):
        """
        Generate the environment model
        """
        numenvs = 2
        # SubprocVecEnv sauf on utilisera cuda derrière
        vec_env = make_vec_env(
            self.env_type.value, 
            n_envs=numenvs, 
            wrapper_class=CustomRewardWrapper, 
            wrapper_kwargs={"llm_reward_function": reward_func})
        if self.learning_algo == Algo.PPO:
            model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu")
        else:
            raise ValueError("The learning algorithm is not implemented.")
        
        return vec_env, model, numenvs
        
