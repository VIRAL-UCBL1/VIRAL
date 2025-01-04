from re import S
from typing import Callable
from gymnasium import make
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import Process, Queue
from logging import getLogger
from time import sleep
from queue import Empty

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from PolicyTrainer.CustomRewardWrapper import CustomRewardWrapper
from PolicyTrainer.TrainingInfoCallback import TrainingInfoCallback
from State.State import State
from Environments.Algo import Algo
from Environments.EnvType import EnvType

import numpy as np
import os


class PolicyTrainer:
    def __init__(self, memory: list[State], env_type: EnvType, timeout: int, numenvs):
        """initialise the policy trainer

        Args:
            memory (list[State]): 
            env_type (EnvType): parameter of the env
            timeout (int): for the model.learn()
        """
        self.logger = getLogger("VIRAL")
        self.memory = memory
        self.timeout = timeout
        self.numenvs = numenvs
        self.algo = env_type.algo
        self.algo_param = env_type.algo_param
        self.env_name = str(env_type)
        self.success_func = env_type.success_func
        if os.name == "posix":
            self.queue = Queue()
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
            self.to_get = 1
        else:
            self._learning(self.memory[0])

    def _learning(self, state: State, queue: Queue = None) -> None:
        """train a policy on an environment

        Args:
            state (State): 
            queue (Queue, optional): handle modification to return. Defaults to None.
        """
        self.logger.info(
            f"state {state.idx} begin is learning"
        )
        vec_env, model, numvenv = self._generate_env_model(state.reward_func, self.numenvs)
        training_callback = TrainingInfoCallback()
        policy = model.learn(total_timesteps=self.timeout, callback=training_callback, progress_bar=True) # , progress_bar=True
        policy.save(f"model/policy{state.idx}.model")
        metrics = training_callback.get_metrics()
        #self.logger.debug(f"{state.idx} TRAINING METRICS: {metrics}")
        sr_test = self.test_policy(policy)
        # ajoute au dict metrics les performances sans ecraser les anciennes
        metrics["sr"] = sr_test
        if os.name == "posix":
            queue.put([state.idx, f"model/policy{state.idx}.model", metrics])
        else:
            self.memory[state.idx].set_performances(metrics)
            self.memory[state.idx].set_policy(policy)
            self.logger.debug(
                f"state {state.idx} has finished learning with performances: {metrics}"
            )

    def evaluate_policy(self, list_idx: list[int]) -> tuple[list[int], list[int], float]:
        """
        Evaluate policy performance for multiple reward functions

        Args:
            objectives_metrics (list[callable]): Custom objective metrics
            num_episodes (int): Number of evaluation episodes

        Returns:
            Dict: Performance metrics for multiple reward functions
        """
        if os.name == "posix":
            if len(self.memory) < 2:
                self.logger.warning("At least two reward functions are required.")
            to_join: list = []
            for i in list_idx:
                if self.memory[i].performances is None:
                    self.multi_process.append(
                        Process(
                            target=self._learning, args=(self.memory[i], self.queue)
                        )
                    )
                    self.multi_process[-1].start()
                    self.to_get += 1
                    to_join.append(len(self.multi_process) - 1)

            while self.to_get != 0:
                try:
                    get = self.queue.get(block=False)
                    self.memory[get[0]].set_policy(get[1])
                    self.memory[get[0]].set_performances(get[2])
                    self.logger.debug(
                        f"state {get[0]} has finished learning with performances: {get[2]['sr']}"
                    )
                    self.to_get -= 1
                except Empty:
                    sleep(0.1)

            for p in to_join:
                self.multi_process[p].join()

            are_worsts: list[int] = []
            are_betters: list[int] = []
            threshold: float = self.memory[0].performances["sr"]
            self.logger.info(f"the threshold is {threshold}")
            for i in list_idx:
                if threshold > self.memory[i].performances["sr"]:
                    are_worsts.append(i)
                else:
                    are_betters.append(i)
            return are_worsts, are_betters, threshold

        else:
            if len(self.memory) < 2:
                self.logger.error("At least two reward functions are required.")
            for i in list_idx:
                if self.memory[i].performances is None:
                    self._learning(self.memory[i])
            
            are_worsts: list[int] = []
            are_betters: list[int] = []
            threshold: float = self.memory[0].performances["sr"]
            self.logger.info(f"the threshold is {threshold}")
            for i in list_idx:
                if threshold > self.memory[i].performances["sr"]:
                    are_worsts.append(i)
                else:
                    are_betters.append(i)
            return are_worsts, are_betters, threshold

    def test_policy(
        self,
        policy,
        nb_episodes: int = 100,
    ) -> float:
        """test a policy already train

        Args:
            env (VecEnv): envs
            policy (): can be PPO or other RLAlgo
            numvenv (int): number of env in the vec
            nb_episodes (int, optional): . Defaults to 100.

        Returns:
            float: _description_
        """
        all_rewards = []
        nb_success = 0
        env = make(self.env_name)
        for _ in range(nb_episodes):
            obs, _ = env.reset()
            episode_rewards = 0
            done = False
            while not done:
                actions, _ = policy.predict(obs)
                obs, reward, term, trunc, info = env.step(actions)
                episode_rewards += reward
                done = trunc or term

                if done:
                    info["TimeLimit.truncated"] = trunc
                    info["terminated"] = term
                    is_success, _ = self.success_func(env, info)
                    if is_success:
                        nb_success += 1
            all_rewards.append(episode_rewards)

        success_rate = nb_success / nb_episodes
        return success_rate

    def test_policy_hf(self, policy_path: str, nb_episodes: int = 5):
        """visualise a policy

        Args:
            policy_path (str): the path of the policy to load
            nb_episodes (int, optional): . Defaults to 100.
        """
        env = make(self.env_name, render_mode='human')
        if self.algo == Algo.PPO:
            policy = PPO.load(policy_path)
        nb_success = 1
        for _ in range(nb_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                actions, _ = policy.predict(obs)
                obs, _, term, trunc, info = env.step(actions)
                if term or trunc:
                    is_success, _ = self.success_func(env, info)
                    if is_success:
                        nb_success += 1
                done = term or trunc
        env.close()

    def _generate_env_model(self, reward_func, numenvs = 2) -> tuple[VecEnv, PPO, int]:
        """Generate the environment model

        Args:
            reward_func (Callable): the generated reward function

        Raises:
            ValueError: if algo not implemented

        Returns:
            tuple[VecEnv, PPO, int]: the envs, the model, the number of envs
        """
        vec_env = make_vec_env(
            self.env_name,
            n_envs=numenvs,
            wrapper_class=CustomRewardWrapper,
            wrapper_kwargs={"success_func": self.success_func, "llm_reward_function": reward_func},
        )
        if self.algo == Algo.PPO:
            model = PPO(env=vec_env, **self.algo_param)
        else:
            raise ValueError("The learning algorithm is not implemented.")

        return vec_env, model, numenvs
