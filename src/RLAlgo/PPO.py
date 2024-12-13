from copy import deepcopy
from logging import getLogger

import threading
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset

logger = getLogger("VIRAL")


class PPO(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        hidden_size: int = 256,
        std: float = 0.0,
        batch_size: int = 5,
        ppo_epoch: int = 4,
        lr: float = 3e-4,
        nb_episodes: int = 2000,
        max_t: int = 1000,
    ):
        """ """
        super(PPO, self).__init__()
        self.num_inputs = env.observation_space.shape[0]
        self.num_outputs = env.action_space.n
        self.critic = nn.Sequential(
            nn.Linear(self.num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )  # predict the reward from a state

        self.actor = nn.Sequential(
            nn.Linear(self.num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_outputs),
        )  # predict the action todo
        self.log_std = nn.Parameter(
            torch.ones(1, self.num_outputs) * std
        )  # compute the std

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.1)

        self.apply(init_weights)  # help to converge

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.nb_episodes = nb_episodes
        self.max_t = max_t
        self.ppo_epoch = ppo_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.env = env

    def forward(self, x) -> tuple[Normal, float]:
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        value = self.critic(x)  # predict the reward
        mu = self.actor(x)  # predict the action
        std = self.log_std.exp()  # compute std
        dist = Normal(mu, std)  # make a normal distribution of proba
        return dist, value

    def __repr__(self) -> str:
        return "PPO"

    def output(self, state: np.ndarray) -> int:
        """ """
        dist, _ = self.forward(state)
        action = dist.sample()
        return action.item()

    def save(self, file: str):
        """
        save the model.
        """
        torch.save(self.state_dict(), file)

    def load(self, file: str):
        """
        load the model form file
        """
        self.load_state_dict(torch.load(file))

    def train(
        self,
        reward_func=None,
        save_name: str = "",
        stop: threading.Event | None = None,
    ) -> tuple[dict, list, float, int]:
        # Sauvegarde des paramètres initiaux
        cp_policy = deepcopy(self)

        # Initialisation des métriques
        recompenses = []
        a_la_suite = 0
        nb_success = 0
        score_max = 0

        state, _ = cp_policy.env.reset()
        for i_episode in range(cp_policy.nb_episodes):
            if stop is not None:
                if stop.is_set():
                    break
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            # Run policy T times
            for t in range(cp_policy.max_t):
                dist, value = cp_policy.forward(state)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.item()
                next_state, reward, terminated, truncated, _ = cp_policy.env.step(action)
                log_probs.append(log_prob)
                values.append(value)
                actions.append(action)
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                state = next_state
                if terminated:
                    break
                if truncated:
                    nb_success +=1
                    break
            # compute Â_1 ... Â_t
            _, value = cp_policy.forward(state)
            values.append(value)
            advantage_estimates, returns = cp_policy._global_advantage_estimates(
                rewards, values
            )
            # optimise
            dataset = cp_policy.PPODataset(
                states, actions, log_probs, returns, advantage_estimates
            )
            data_loader = DataLoader(dataset, batch_size=cp_policy.batch_size, shuffle=True)
            cp_policy._ppo_update(data_loader)
        if save_name is not None:
            cp_policy.save(save_name)
        return cp_policy, rewards, (nb_success / i_episode), i_episode

    def _global_advantage_estimates(self, rewards, values, gamma=0.99, tau=0.95):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] - values[step]
            gae = delta + gamma * tau * gae
            returns.insert(0, gae + values[step])
        return returns - values, returns

    def _ppo_update(
        self, dataloader: DataLoader):
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in dataloader:
                dist, value = self.forward(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 0.8, 1.2) * advantage
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    class PPODataset(Dataset):
        def __init__(self, states, actions, log_probs, returns, advantages):
            self.states = states
            self.actions = actions
            self.log_probs = log_probs
            self.returns = returns
            self.advantages = advantages

        def __len__(self):
            return self.states.size(0)

        def __getitem__(self, idx):
            return (
                self.states[idx, :],
                self.actions[idx, :],
                self.log_probs[idx, :],
                self.returns[idx, :],
                self.advantages[idx, :],
            )
