"""
Actor / Critic / CostCritic networks for MAPPO.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from typing import List


def mlp(sizes: List[int], activation=nn.Tanh, output_activation=None):
    """Build an MLP with orthogonal initialization."""
    layers = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        # Orthogonal init
        gain = 0.01 if i == len(sizes) - 2 else np.sqrt(2)
        nn.init.orthogonal_(linear.weight, gain=gain)
        nn.init.constant_(linear.bias, 0.0)
        layers.append(linear)
        if i < len(sizes) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Gaussian policy with learnable log_std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int]):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_sizes + [act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_actions(self, obs, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.rsample()
        log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        return actions, log_probs

    def evaluate_actions(self, obs, actions):
        dist = self.forward(obs)
        log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()
        return log_probs, entropy


class Critic(nn.Module):
    """State value function V(s)."""

    def __init__(self, obs_dim: int, hidden_sizes: List[int]):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_sizes + [1])

    def forward(self, obs):
        return self.net(obs)


class CostCritic(nn.Module):
    """Cost value function V_c(s) for logging."""

    def __init__(self, obs_dim: int, hidden_sizes: List[int]):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_sizes + [1])

    def forward(self, obs):
        return self.net(obs)
