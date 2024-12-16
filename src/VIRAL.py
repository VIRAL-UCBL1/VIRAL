import random
import signal
import sys
from logging import getLogger
from typing import Callable, Dict, List

import numpy as np

from utils.OllamaChat import OllamaChat
from utils.State import State
from utils.Algo import Algo
from utils.Environments import Environments
from utils.CustomRewardWrapper import CustomRewardWrapper
from utils.TrainingInfoCallback import TrainingInfoCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import torch

class VIRAL:
    def __init__(
        self,
        learning_algo: Algo,
        env_type : Environments,
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
        if (options.get("seed") is None):
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
        self.env = None
        self.objectives_metrics = objectives_metrics
        self.learning_algo : Algo = learning_algo
        self.learning_method = None
        self.memory: List[State] = [State(0)]
        self.logger = getLogger("VIRAL")
        #self._learning(self.memory[0])


    def generate_reward_function(
        self, task_description: str, iterations: int = 1
    ) -> List[State]:
        """
        Generate reward function using LLM

        Args:
            task_description (str): Detailed description of the task
            environment_type (str): Type of environment (2D/3D, robotics, etc.)

        Returns:
            Callable: Generated reward function
        """
        # TODO Pourquoi additional_options ici et pas dans le constructeur ?
        additional_options = {
            "temperature": 1,
            #"num_predict": 3, # l'impression que ça change rien a creuser
            
            #"mirostat" : 1,
            #"mirostat_eta" : 0.01, #gère la vitesse de réponses du model (0.1 par défaut) plus c'est petit plus c'est lent
            #"mirostat_tau" : 4.0, #gère la balance entre la diversité et la coherence des réponses (5.0 par défaut) plus c'est petit plus c'est focus et cohérent
            
            #num_ctx": 2048, # nombre de tokens contextuels (2048 par défaut peut être pas nécessaire de changer)
            
            #repeat_last_n": 64, # combien le model regarde en arrière pour éviter de répéter les réponses (64 par défaut large pour nous)
            
            #"repeat_penalty": 1.5, # pénalité pour éviter de répéter les réponses (1.1 par défaut au mac 1.5 intéressant a modificer je pense)
            
            #"stop": "stop you here" # pour stopper la génération de texte pas intéressant pour nous
            
            #"tfs_z": 1.2, #reduire l'impacte des token les moins "pertinents" (1.0 par défaut pour désactiver 2.0 max)
            
            #"top_k": 30, #reduit la probabilité de générer des non-sens (40 par défaut, 100 pour générer des réponses plus diverses, 10 pour des réponses plus "conservatrices")
            #"top_p": 0.95, #marche avec le top_k une forte valeur pour des texte plus diverses (0.9 par défaut)
            #"min_p": 0.05, #alternative au top_p, vise a s'aéssurer de la balance entre qualité et diversité (0.0 par défaut)
            
            #"seed": 42, # a utiliser pour la reproductibilité des résultats (important si publication)
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
            reward_func, response = self._get_runnable_function(response)
            self.memory.append(State(i, reward_func, response))
            
        best_idx, worst_idx = self.evaluate_policy(1, 2) 
        self.logger.debug(f"state to refine: {worst_idx}")
        ### SECOND STAGE ###
        for n in range(iterations - 1):
            new_idx = self.self_refine_reward(worst_idx)
            best_idx, worst_idx = self.evaluate_policy(best_idx, new_idx)
            self.logger.debug(f"state to refine: {worst_idx}")
        return self.memory

    def _get_code(self, response: str) -> str:
        cleaned_response = response.strip("```").replace("python", "").strip()
        if "def " not in cleaned_response:
            raise ValueError("The answer does not contain a valid function definition.")
        self.logger.debug("Code nettoyé pour compilation :\n" + cleaned_response)
        return cleaned_response

    def _get_runnable_function(self, response: str, error: str = None) -> Callable:
        if error is not None:
            self.llm.add_message(error)
            response = self.llm.generate_response(stream=True)
            response = self.llm.print_Generator_and_return(response)
        try:
            env = gym.make(self.env_type.value)
            response = self._get_code(response)
            reward_func = self._compile_reward_function(response)
            state, _ = env.reset()
            action = env.action_space.sample()
            next_observation, _, terminated, truncated, _ = env.step(action)
            self._test_reward_function(
                reward_func,
                observations=next_observation,
                terminated=terminated,
                truncated=truncated,
            )
        except ValueError as e:
            self.logger.warning(str(e))
            return self._get_runnable_function(response, str(e))
        except SyntaxError as e:
            self.logger.warning(f"Error syntax {e}")
            return self._get_runnable_function(response, str(e))
        except RuntimeError as e:
            self.logger.warning(f"Error execution {e}")
            return self._get_runnable_function(response, str(e))

        return reward_func, response

    def _compile_reward_function(self, response: str) -> Callable:
        """
        Compile the reward function from the LLM response.
        Args:
            response (str): LLM generated reward function.

        Returns:
            Callable: Compiled reward function.
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

    def _test_reward_function(self, reward_function: Callable, *args, **kwargs):
        """
        Test the compiled reward function with example inputs.

        Args:
            reward_function (Callable): The reward function to test.
            *args: Positional arguments for the reward function.
            **kwargs: Keyword arguments for the reward function.
        """
        try:
            reward = reward_function(*args, **kwargs)
            self.logger.debug(f"Reward function output: {reward}")
        except Exception as e:
            raise RuntimeError(f"Error during reward function execution: {e}")

    def self_refine_reward(self, idx: int) -> Callable:
        """
        Self-refinement of reward function based on performance

        Args:
            current_reward_func (Callable): Current reward function
            performance_metrics (Dict): Performance evaluation metrics

        Returns:
            Callable: Refined reward function
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
        reward_func, refined_response = self._get_runnable_function(refined_response)
        self.memory.append(State(len(self.memory), reward_func, refined_response))

        return len(self.memory) - 1

    def _learning(self, state: State) -> None:
        """train a policy on an environment"""
        self.logger.debug(f"state {state.idx} begin is learning with reward function: {state.reward_func_str}")
        vec_env, model = self._generate_env_model(state.reward_func)
        training_callback = TrainingInfoCallback()
        policy = model.learn(total_timesteps=60000, callback=training_callback)
        metrics = training_callback.get_metrics()
        self.logger.debug(f"TRAINING METRICS: {metrics}")
        self.memory[state.idx].set_policy(policy)
        sr_test = self.test_policy(vec_env, policy)
        # ajoute au dict metrics les performances sans ecraser les anciennes
        metrics["test_success_rate"] = sr_test
        self.memory[state.idx].set_performances(metrics)
        self.logger.debug(f"state {state.idx} as finished is learning with performances: {metrics}")

    def evaluate_policy(self, idx1: int, idx2: int) -> int:
        """
        Evaluate policy performance for multiple reward functions

        Args:
            objectives_metrics (List[callable]): Custom objective metrics
            num_episodes (int): Number of evaluation episodes

        Returns:
            Dict: Performance metrics for multiple reward functions
        """
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
        reward_func=None,
        nb_episodes=100,
        max_t=1000
    ) -> list:
        all_rewards = []
        all_states = []
        nb_success = 0
        for epi in range(1, nb_episodes + 1):
            obs = env.reset()
            epi_rewards = 0
            while True:
                action, _states = policy.predict(obs)
                obs, reward, dones, info = env.step(action)
                epi_rewards += reward.item()
                if dones[0]:
                    if info[0]["TimeLimit.truncated"]:
                        print("truncated")
                        nb_success += 1
                    break
            all_rewards.append(epi_rewards)
        return (nb_success / nb_episodes)

    def _generate_env_model(self, reward_func):
        """
        Generate the environment model
        """
        vec_env = make_vec_env(self.env_type.value, n_envs=1, wrapper_class=CustomRewardWrapper, wrapper_kwargs={'llm_reward_function': reward_func})
        if self.learning_algo == Algo.PPO:
            model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
        else:
            raise ValueError("The learning algorithm is not implemented.")
        
        return vec_env, model
        