import os
from logging import getLogger
from multiprocessing import Process, Queue
from queue import Empty
from time import sleep

import gymnasium as gym
from gymnasium import make
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from Environments.Algo import Algo
from Environments.EnvType import EnvType
from log.log_config import get_log_level
from PolicyTrainer.CustomRewardWrapper import CustomRewardWrapper
from PolicyTrainer.TrainingInfoCallback import TrainingInfoCallback
from State.State import State


class PolicyTrainer:
    def __init__(self, memory: list[State], seed: int, env_type: EnvType, timeout: int, nb_vec_envs: int, legacy_training: bool):
        """
        Initialize the PolicyTrainer instance.

        Args:
            memory (list[State]): list of states
            env_type (EnvType): the type of environment
            timeout (int): the maximum number of timesteps
        """
        self.logger = getLogger("VIRAL")
        self.progress_bar = True if get_log_level() == "DEBUG" else False
        self.memory = memory
        self.timeout = timeout
        self.nb_vec_envs = nb_vec_envs
        self.algo = env_type.algo
        self.seed = seed
        env_type.algo_param.update({'seed': seed})
        self.algo_param = env_type.algo_param
        self.objective_metric = env_type.objective_metric
        self.env_name = str(env_type)
        self.success_func = env_type.success_func
        self.to_join: list = []
        self.queue = Queue()
        self.multi_process: list[Process] = []
        self.to_get = 0
        self.legacy_training = legacy_training
        if len(self.memory) > 0 and self.legacy_training:
            self.start_learning(0)

    def _learning(self, state: State, queue: Queue = None) -> None:
        """
        Train a policy for a given state

        Args:
            state (State): the state to train
            queue (Queue, optional): handle modification to return. Defaults to None.
        """
        self.logger.info(
            f"state {state.idx} begin is learning"
        )
        model = self._generate_env_model(state.reward_func)
        training_callback = TrainingInfoCallback()
        policy = model.learn(total_timesteps=self.timeout, callback=training_callback, progress_bar=self.progress_bar)
        path = f"data/model/{self.env_name}_{self.seed}_{state.idx}.pth"
        policy.save(path)
        metrics = training_callback.get_metrics()
        #self.logger.debug(f"{state.idx} TRAINING METRICS: {metrics}")
        sr_test = self.test_policy(policy)
        objective_metric = self.objective_metric(metrics.pop('observations'))
        if objective_metric is not None:
            metrics.update(objective_metric)
        metrics["sr"] = sr_test
        if os.name == "posix":
            queue.put([state.idx, path, metrics])
        else:
            self.memory[state.idx].set_performances(metrics)
            self.memory[state.idx].set_policy(path)
        self.logger.info(f"state {state.idx} has finished learning with performances: {sr_test}")


    def start_learning(self, idx: int) -> None:
        """
        Start the learning process for a given state
        
        Args:
            idx (int): the index of the state to train
        """
        if os.name == "posix":
            self._start_proccess_learning(idx)
        else:
            self._learning(self.memory[idx])

    def _start_proccess_learning(self, idx: int) -> None:
        """
        Start the learning process for a given state in a new process
        
        Args:
            idx (int): the index of the state to train
        """
        assert (os.name == "posix"), "multi-proccess features only available on LINUX system..."
        if self.memory[idx].policy is None:
            self.multi_process.append(
                Process(
                    target=self._learning, args=(self.memory[idx], self.queue)
                )
            )
            self.to_join.append(idx)
            self.multi_process[-1].start()
            self.to_get += 1

    def evaluate_policy(self, list_idx: list[int]) -> tuple[list[int], list[int], float]:
        """
        Evaluate policy performance for multiple reward functions

        Args:
            objectives_metrics (list[callable]): Custom objective metrics
            num_episodes (int): Number of evaluation episodes

        Returns:
            Dict: Performance metrics for multiple reward functions
        """
        if len(self.memory) < 2:
            self.logger.warning("At least two reward functions are required.")

        for i in list_idx:
            if self.memory[i].policy is None and i not in self.to_join:
                self.logger.error("Need to start_learning before evaluate him")
                raise RuntimeError

        if os.name == "posix": # waiting proccess
            while self.to_get != 0:
                try:
                    get = self.queue.get(block=False)
                    self.memory[get[0]].set_policy(get[1])
                    self.memory[get[0]].set_performances(get[2])
                    self.to_get -= 1
                except Empty:
                    sleep(0.5)

            for p in self.to_join:
                p = p if self.legacy_training else p-1 # Corresponding idx if not inital training
                self.multi_process[p].join()

        are_worsts: list[int] = []
        are_betters: list[int] = []
        if self.legacy_training:
            threshold: float = self.memory[0].performances["sr"]
        else:
            threshold: float = 0.9
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
        """
        Test a policy on the environment

        Args:
            env (VecEnv): the environment
            policy (BasePolicy): the policy to test
            numvenv (int): the number of environments
            nb_episodes (int, optional): the number of episodes to test. Defaults to 100.

        Returns:
            float: the success rate of the policy
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
                #print(f"info: {info}")
                
                episode_rewards += reward
                done = trunc or term

                if done:
                    info["TimeLimit.truncated"] = trunc
                    info["terminated"] = term
                    # print(f"infodsqdqsdqsds: {info}")
                    is_success, _ = self.success_func(env, info)
                    if is_success:
                        nb_success += 1
            all_rewards.append(episode_rewards)

        success_rate = nb_success / nb_episodes
        return success_rate


    def start_hf(self, policy_path: str, nb_episodes: int = 10):
        """
        Start the test of a policy to evaluate its performances
        
        Args:
            policy_path (str): the path of the policy to test
            nb_episodes (int, optional): the number of episodes to test. Defaults to 100.
            
        """
        if os.name == "posix":
            self.multi_process.append(
                Process(
                    target=self.test_policy_hf, args=(policy_path, nb_episodes)
                )
            )
            self.multi_process[-1].start()
            self.multi_process[-1].join()
        else:
            self.test_policy_hf(policy_path, nb_episodes)


    def test_policy_hf(self, policy_path: str, nb_episodes: int = 10):
        """
        Visualize the test of a policy to evaluate its performances

        Args:
            policy_path (str): the path of the policy to test
            nb_episodes (int, optional): the number of episodes to test. Defaults to 10.
        """
        env = make(self.env_name, render_mode='human') # TODO pass env param
        if self.algo == Algo.PPO:
            policy = PPO.load(policy_path)
        elif self.algo == Algo.DQN:
            policy = DQN.load(policy_path)
        nb_success = 0
        for _ in range(nb_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                actions, _ = policy.predict(obs)
                obs, _, term, trunc, info = env.step(actions)
                if term or trunc:
                    self.logger.info(info)
                    print(f"terminated: {term}, truncated: {trunc}")
                    info["TimeLimit.truncated"] = trunc
                    info["terminated"] = term
                    print(f"info: {info}")
                    is_success, _ = self.success_func(env, info)
                    if is_success:
                        nb_success += 1
                done = term or trunc
        print(f"nb_success: {nb_success/nb_episodes}")
        env.close()

    def start_vd(self, policy_path: str, nb_episodes: int = 3):
        """
        Start the test of a policy to evaluate its performances
        
        Args:
            policy_path (str): the path of the policy to test
            nb_episodes (int, optional): the number of episodes to test. Defaults to 3.
            
        """
        if os.name == "posix":
            self.multi_process.append(
                Process(
                    target=self.test_policy_video, args=(policy_path, nb_episodes)
                )
            )
            self.multi_process[-1].start()
            self.multi_process[-1].join()
        else:
            self.test_policy_video(policy_path, nb_episodes)

    def test_policy_video(self, policy_path: str, nb_episodes: int = 3):
        """
        Visualize the test of a policy to evaluate its performances
        
        Args:
            policy_path (str): the path of the policy to test
            nb_episodes (int, optional): the number of episodes to test. Defaults to 3.
            
        """
        env = make(self.env_name, render_mode='rgb_array')
        env = RecordVideo(
            env, video_folder=f"records/{self.env_name}", name_prefix=f"{self.env_name}_{self.seed}", episode_trigger=lambda e: True
        )
        if self.algo == Algo.PPO:
            policy = PPO.load(policy_path)
        elif self.algo == Algo.DQN:
            policy = DQN.load(policy_path)
        for videos in range(nb_episodes):
            done = truncated = False
            obs, info = env.reset()
            while not (done or truncated):
                action, _states = policy.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
        env.close()

    def _generate_env_model(self, reward_func) -> tuple[VecEnv, PPO, int]:
        """
        Generate the environment model

        Args:
            reward_func (Callable): the generated reward function

        Raises:
            ValueError: if algo not implemented

        Returns:
            tuple[VecEnv, PPO, int]: the envs, the model, the number of envs
        """
        if self.nb_vec_envs == 1:
            self.logger.debug("simple env")
            env = gym.make(self.env_name) # , terminate_when_unhealthy=False
            env = CustomRewardWrapper(env, self.success_func, reward_func)
        else:
            env = make_vec_env(
                self.env_name,
                n_envs=self.nb_vec_envs,
                wrapper_class=CustomRewardWrapper,
                wrapper_kwargs={"success_func": self.success_func, "llm_reward_function": reward_func},
                # env_kwargs={'terminate_when_unhealthy': False}
            )
        if self.algo == Algo.PPO:
            model = PPO(env=env, **self.algo_param)
        elif self.algo == Algo.DQN:
            # use gym.make instead of make_vec_env for DQN. gym 10min / vec_env 2h
            model = DQN(env=env, **self.algo_param)
        else:
            raise ValueError("The learning algorithm is not implemented.")

        return model
