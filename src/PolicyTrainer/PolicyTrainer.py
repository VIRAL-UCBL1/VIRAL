from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv #TODO maybe emilien want to use this
from multiprocessing import Process, Queue
from logging import getLogger
from time import sleep
from queue import Empty

from PolicyTrainer.CustomRewardWrapper import CustomRewardWrapper
from PolicyTrainer.TrainingInfoCallback import TrainingInfoCallback
from State.State import State
from Environments.Algo import Algo
from Environments.EnvType import EnvType

import numpy as np
import os


class PolicyTrainer:
    def __init__(self, memory: list[State], env_type: EnvType):
        self.logger = getLogger("VIRAL")
        self.memory = memory
        self.algo = env_type.algo
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
            self._learning(self.memory[0], self.queue)

    def _learning(self, state: State, queue: Queue = None) -> None:
        """train a policy on an environment"""
        self.logger.debug(f"state {state.idx} begin is learning with reward function: {state.reward_func_str}")
        vec_env, model, numvenv = self._generate_env_model(state.reward_func)
        training_callback = TrainingInfoCallback()
        policy = model.learn(total_timesteps=25000, callback=training_callback)
        policy.save(f"model/policy{state.idx}.model")
        metrics = training_callback.get_metrics()
        self.logger.debug(f"{state.idx} TRAINING METRICS: {metrics}")
        sr_test = self.test_policy(vec_env, policy, numvenv)
        # ajoute au dict metrics les performances sans ecraser les anciennes
        metrics["test_success_rate"] = sr_test
        if os.name == "posix":
            queue.put([state.idx, f"model/policy{state.idx}.model", metrics])
        else:
            self.memory[state.idx].set_performances(metrics)
            self.memory[state.idx].set_policy(policy)
            self.logger.debug(f"state {state.idx} has finished learning with performances: {metrics}")

    def evaluate_policy(self, idx1: int, idx2: int) -> int:
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
                self.logger.error("At least two reward functions are required.")
            to_join: list = []
            for i in [idx1, idx2]:
                if self.memory[i].performances is None:
                    self.multi_process.append(
                        Process(target=self._learning, args=(self.memory[i], self.queue))
                    )
                    self.multi_process[-1].start()
                    self.to_get += 1
                    to_join.append(len(self.multi_process)-1)

            while self.to_get != 0:
                try:
                    get = self.queue.get(block=False)
                    self.memory[get[0]].set_policy(get[1])
                    self.memory[get[0]].set_performances(get[2])
                    self.logger.debug(
                        f"state {get[0]} has finished learning with performances: {get[2]}"
                    )
                    self.to_get -= 1
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
                        if self.success_func(env.envs[i], info):
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
            self.env_name, 
            n_envs=numenvs, 
            wrapper_class=CustomRewardWrapper, 
            wrapper_kwargs={"llm_reward_function": reward_func})
        if self.algo == Algo.PPO:
            model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
        else:
            raise ValueError("The learning algorithm is not implemented.")

        return vec_env, model, numenvs

