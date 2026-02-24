"""
QMIX: Monotonic Value Function Factorisation

QMIX learns a centralized action-value function Q_tot as a monotonic
mixing of per-agent utilities. Uses value decomposition for CTDE.

Reference:
    Rashid et al., "QMIX: Monotonic Value Function Factorisation for
    Deep Multi-Agent Reinforcement Learning", ICML 2018
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from cosmos.registry import ALGO_REGISTRY
from cosmos.algos.base import BaseMARLAlgo, AlgoConfig, OffPolicyAlgo


class RNNAgent(nn.Module):
    """
    Agent network with GRU for partial observability.

    Each agent uses an RNN to process observation history
    and output Q-values for each action.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(1, self.rnn_hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observations, shape (batch, obs_dim)
            hidden_state: RNN hidden state, shape (batch, rnn_hidden_dim)

        Returns:
            q_values: Q-values for each action, shape (batch, n_actions)
            hidden_state: Updated hidden state
        """
        x = F.relu(self.fc1(obs))
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q, h


class QMixer(nn.Module):
    """
    Mixing network that combines agent Q-values into Q_tot.

    Uses hypernetworks conditioned on global state to produce
    mixing weights with non-negative constraints (monotonicity).
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_embed_dim: int = 64
    ):
        super().__init__()

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        # Hypernetworks for mixing weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, num_agents * mixing_embed_dim)
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim)
        )

        # Hypernetworks for biases
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(
        self,
        agent_qs: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix agent Q-values into Q_tot.

        Args:
            agent_qs: Per-agent Q-values, shape (batch, num_agents)
            state: Global state, shape (batch, state_dim)

        Returns:
            q_tot: Total Q-value, shape (batch, 1)
        """
        batch_size = agent_qs.shape[0]

        # Reshape agent Qs
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)

        # Generate mixing weights (abs ensures monotonicity)
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)

        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)

        # Generate biases
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_embed_dim)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # Two-layer mixing
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.view(batch_size, 1)


@ALGO_REGISTRY.register("qmix")
class QMIX(OffPolicyAlgo):
    """
    QMIX algorithm implementation.

    Uses value decomposition with monotonic mixing for cooperative
    multi-agent tasks. Each agent learns individual Q-functions,
    combined via a mixing network for centralized training.

    Config options:
        actor_lr: Learning rate (default: 5e-4)
        gamma: Discount factor (default: 0.99)
        tau: Target network update rate (default: 0.005)
        hidden_sizes: Hidden layer sizes (default: (64, 64))
        buffer_size: Replay buffer capacity (default: 5000 episodes)
        batch_size: Training batch size (default: 32 episodes)
        epsilon_start: Initial exploration epsilon (default: 1.0)
        epsilon_end: Final exploration epsilon (default: 0.05)
        epsilon_decay: Epsilon decay steps (default: 50000)
        target_update_interval: Steps between target updates (default: 200)
    """

    def __init__(
        self,
        obs_dim: int,
        share_obs_dim: int,
        act_dim: int,
        num_agents: int,
        n_actions: int = None,
        cfg: Optional[Union[AlgoConfig, Dict[str, Any]]] = None,
        device: str = "cpu"
    ):
        """
        Args:
            obs_dim: Individual observation dimension.
            share_obs_dim: Shared/global state dimension.
            act_dim: Action dimension (for continuous) or ignored for discrete.
            num_agents: Number of agents.
            n_actions: Number of discrete actions.
            cfg: Configuration dict or AlgoConfig.
            device: Torch device.
        """
        super().__init__(obs_dim, share_obs_dim, act_dim, num_agents, cfg, device)

        # QMIX uses discrete actions
        self.n_actions = n_actions if n_actions is not None else act_dim

        # Get config values
        self.lr = getattr(self.cfg, 'actor_lr', 5e-4)
        self.gamma = getattr(self.cfg, 'gamma', 0.99)
        self.tau = getattr(self.cfg, 'tau', 0.005)
        hidden_sizes = getattr(self.cfg, 'hidden_sizes', (64, 64))
        self.hidden_dim = hidden_sizes[0] if hidden_sizes else 64
        self.rnn_hidden_dim = hidden_sizes[1] if len(hidden_sizes) > 1 else 64

        # Exploration parameters
        self.epsilon = getattr(cfg, 'epsilon_start', 1.0) if isinstance(cfg, dict) else 1.0
        self.epsilon_end = getattr(cfg, 'epsilon_end', 0.05) if isinstance(cfg, dict) else 0.05
        self.epsilon_decay = getattr(cfg, 'epsilon_decay', 50000) if isinstance(cfg, dict) else 50000
        self.target_update_interval = getattr(cfg, 'target_update_interval', 200) if isinstance(cfg, dict) else 200

        # Networks
        self.agent = RNNAgent(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            hidden_dim=self.hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim
        ).to(device)

        self.mixer = QMixer(
            num_agents=num_agents,
            state_dim=share_obs_dim,
            mixing_embed_dim=32,
            hypernet_embed_dim=64
        ).to(device)

        # Target networks
        self.target_agent = copy.deepcopy(self.agent)
        self.target_mixer = copy.deepcopy(self.mixer)

        # Optimizer
        self.params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        # Hidden states for episode
        self.hidden_states = None
        self.target_hidden_states = None

        # Training step counter
        self.train_steps = 0

    def init_hidden(self, batch_size: int = 1):
        """Initialize agent hidden states."""
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.num_agents, -1
        ).to(self.device)

    def get_actions(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        avail_actions: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select actions using epsilon-greedy.

        Args:
            obs: Observations, shape (num_agents, obs_dim)
            deterministic: If True, always select greedy action
            avail_actions: Available actions mask (optional)

        Returns:
            actions: Selected actions, shape (num_agents, 1)
            q_values: Q-values for selected actions
        """
        if self.hidden_states is None:
            self.init_hidden(1)

        obs_tensor = torch.FloatTensor(obs).to(self.device)

        with torch.no_grad():
            # Get Q-values for each agent
            q_values_list = []
            new_hidden_list = []

            for i in range(self.num_agents):
                agent_obs = obs_tensor[i:i+1]
                agent_hidden = self.hidden_states[0, i:i+1]
                q, h = self.agent(agent_obs, agent_hidden)
                q_values_list.append(q)
                new_hidden_list.append(h)

            q_values = torch.stack(q_values_list, dim=0).squeeze(1)  # (num_agents, n_actions)
            self.hidden_states = torch.stack(new_hidden_list, dim=0).unsqueeze(0)

        # Mask unavailable actions
        if avail_actions is not None:
            avail_mask = torch.FloatTensor(avail_actions).to(self.device)
            q_values[avail_mask == 0] = -float('inf')

        # Epsilon-greedy action selection
        if deterministic or np.random.random() > self.epsilon:
            actions = q_values.argmax(dim=-1).cpu().numpy()
        else:
            if avail_actions is not None:
                actions = np.array([
                    np.random.choice(np.where(avail_actions[i] == 1)[0])
                    for i in range(self.num_agents)
                ])
            else:
                actions = np.random.randint(0, self.n_actions, size=self.num_agents)

        # Return actions as (num_agents, 1) for compatibility
        selected_q = q_values.gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device))

        return actions.reshape(-1, 1), selected_q.cpu().numpy()

    def get_values(self, share_obs: np.ndarray) -> np.ndarray:
        """
        Get Q_tot value estimate.

        Note: QMIX doesn't have a separate value function.
        Returns placeholder for interface compatibility.
        """
        return np.zeros((self.num_agents, 1))

    def update(self, buffer: Any) -> Dict[str, float]:
        """
        Update QMIX from experience buffer.

        Args:
            buffer: EpisodeReplayBuffer with stored episodes

        Returns:
            Dict with loss values
        """
        if not buffer.can_sample(getattr(self.cfg, 'batch_size', 32)):
            return {"loss": 0.0}

        batch = buffer.sample(getattr(self.cfg, 'batch_size', 32))

        # Unpack batch
        obs = batch["obs"]  # (batch, seq_len, num_agents, obs_dim)
        actions = batch["actions"]  # (batch, seq_len, num_agents, 1)
        rewards = batch["rewards"]  # (batch, seq_len, num_agents)
        dones = batch["dones"]  # (batch, seq_len, num_agents)
        filled = batch["filled"]  # (batch, seq_len)
        state = batch["share_obs"][:, :, 0, :]  # Use first agent's share_obs as state

        batch_size, seq_len = obs.shape[:2]

        # Get available actions if present
        avail_actions = batch.get("avail_actions")  # (batch, seq_len, num_agents, n_actions)

        # Initialize hidden states
        hidden = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.num_agents, -1
        ).to(self.device)
        target_hidden = self.target_agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.num_agents, -1
        ).to(self.device)

        # Compute Q-values for all timesteps
        q_evals = []
        q_targets = []

        for t in range(seq_len):
            # Current Q-values
            q_eval_list = []
            q_target_list = []
            new_hidden_list = []
            new_target_hidden_list = []

            for i in range(self.num_agents):
                agent_obs = obs[:, t, i, :]
                q, h = self.agent(agent_obs, hidden[:, i, :])
                q_eval_list.append(q)
                new_hidden_list.append(h)

                # Target Q-values
                with torch.no_grad():
                    q_t, h_t = self.target_agent(agent_obs, target_hidden[:, i, :])
                    q_target_list.append(q_t)
                    new_target_hidden_list.append(h_t)

            q_evals.append(torch.stack(q_eval_list, dim=1))  # (batch, num_agents, n_actions)
            q_targets.append(torch.stack(q_target_list, dim=1))

            hidden = torch.stack(new_hidden_list, dim=1)
            target_hidden = torch.stack(new_target_hidden_list, dim=1)

        q_evals = torch.stack(q_evals, dim=1)  # (batch, seq_len, num_agents, n_actions)
        q_targets = torch.stack(q_targets, dim=1)

        # Get chosen action Q-values
        actions_long = actions.long().squeeze(-1)  # (batch, seq_len, num_agents)
        chosen_q = q_evals.gather(3, actions_long.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len, num_agents)

        # Get max target Q-values
        if avail_actions is not None:
            q_targets[avail_actions == 0] = -float('inf')
        max_target_q = q_targets.max(dim=-1)[0]  # (batch, seq_len, num_agents)

        # Mix Q-values
        q_tot_eval = []
        q_tot_target = []

        for t in range(seq_len):
            q_tot_eval.append(self.mixer(chosen_q[:, t, :], state[:, t, :]))
            with torch.no_grad():
                q_tot_target.append(self.target_mixer(max_target_q[:, t, :], state[:, t, :]))

        q_tot_eval = torch.stack(q_tot_eval, dim=1).squeeze(-1)  # (batch, seq_len)
        q_tot_target = torch.stack(q_tot_target, dim=1).squeeze(-1)

        # Compute targets
        team_rewards = rewards.mean(dim=-1)  # (batch, seq_len)
        team_dones = dones.any(dim=-1).float()  # (batch, seq_len)

        targets = team_rewards[:, :-1] + self.gamma * (1 - team_dones[:, :-1]) * q_tot_target[:, 1:]

        # TD error
        td_error = q_tot_eval[:, :-1] - targets

        # Mask padding
        mask = filled[:, :-1]
        masked_td_error = td_error * mask

        # Loss
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimizer.step()

        # Update target networks
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self._soft_update_target()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end,
                          self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay)

        return {
            "loss": loss.item(),
            "q_tot_mean": q_tot_eval.mean().item(),
            "epsilon": self.epsilon
        }

    def _soft_update_target(self):
        """Soft update target networks."""
        for param, target_param in zip(self.agent.parameters(), self.target_agent.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "agent": self.agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_agent.load_state_dict(checkpoint["target_agent"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", 0.05)
        self.train_steps = checkpoint.get("train_steps", 0)

    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.agent.eval()
        self.mixer.eval()

    def train_mode(self):
        """Set networks to training mode."""
        self.agent.train()
        self.mixer.train()

    def to(self, device: str) -> "QMIX":
        """Move model to device."""
        self.device = device
        self.agent.to(device)
        self.mixer.to(device)
        self.target_agent.to(device)
        self.target_mixer.to(device)
        return self

    def reset(self):
        """Reset hidden states for new episode."""
        self.hidden_states = None
        self.target_hidden_states = None
