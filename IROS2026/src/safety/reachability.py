"""
Hamilton-Jacobi Reachability Analysis (Section III-B)

Offline computation of environment-aware safe regions using
value function approximation with neural networks.

Key equations from paper:
    - Feasibility value: V*_c(s) = min_pi max_t c(s_t)
    - Bellman backup: T* Q_c = (1-gamma) c(s) + gamma max{c(s), V_c(s')}
    - Expectile regression for offline learning
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy classes for type hints
    class nn:
        class Module:
            pass


@dataclass
class ReachabilityConfig:
    """Configuration for reachability pretraining."""
    hidden_dim: int = 256
    n_layers: int = 3
    gamma: float = 0.99
    tau: float = 0.9  # Expectile for IQL
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_updates: int = 10000


class FeasibilityValueNet(nn.Module if TORCH_AVAILABLE else object):
    """
    Neural network for feasibility value function V*_c(s).

    V*_c(s) <= 0 means state s is feasible (can satisfy constraints).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for FeasibilityValueNet")

        super().__init__()

        layers = []
        in_dim = state_dim

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute V*_c(s)."""
        return self.net(state).squeeze(-1)


class FeasibilityQNet(nn.Module if TORCH_AVAILABLE else object):
    """
    Neural network for feasibility Q-function Q_c(s, a).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for FeasibilityQNet")

        super().__init__()

        layers = []
        in_dim = state_dim + action_dim

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q_c(s, a)."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


class ReachabilityPretrainer:
    """
    Offline reachability pretrainer using IQL-style learning.

    Learns V*_c(s) from offline data to identify feasible regions
    without online interaction.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: ReachabilityConfig = None,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ReachabilityPretrainer")

        self.config = config or ReachabilityConfig()
        self.device = torch.device(device)

        # Networks
        self.V_net = FeasibilityValueNet(
            state_dim,
            self.config.hidden_dim,
            self.config.n_layers,
        ).to(self.device)

        self.Q_net = FeasibilityQNet(
            state_dim,
            action_dim,
            self.config.hidden_dim,
            self.config.n_layers,
        ).to(self.device)

        # Target networks
        self.V_target = FeasibilityValueNet(
            state_dim,
            self.config.hidden_dim,
            self.config.n_layers,
        ).to(self.device)
        self.V_target.load_state_dict(self.V_net.state_dict())

        # Optimizers
        self.V_optimizer = torch.optim.Adam(
            self.V_net.parameters(),
            lr=self.config.learning_rate,
        )
        self.Q_optimizer = torch.optim.Adam(
            self.Q_net.parameters(),
            lr=self.config.learning_rate,
        )

        self.tau = self.config.tau
        self.gamma = self.config.gamma

    def expectile_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """
        Asymmetric expectile loss for IQL.

        L^tau(u) = |tau - 1(u < 0)| * u^2
        """
        diff = target - pred
        weight = torch.where(diff > 0, tau, 1 - tau)
        return (weight * diff ** 2).mean()

    def bellman_backup(
        self,
        states: torch.Tensor,
        costs: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feasible Bellman backup (Eq. 13).

        T* Q_c = (1 - gamma) * c(s) + gamma * max{c(s), V_c(s')}
        """
        with torch.no_grad():
            V_next = self.V_target(next_states)
            max_term = torch.maximum(costs, V_next)

        return (1 - self.gamma) * costs + self.gamma * max_term

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        costs: np.ndarray,
        next_states: np.ndarray,
    ) -> Dict[str, float]:
        """
        Single update step.

        Args:
            states: [batch, state_dim]
            actions: [batch, action_dim]
            costs: [batch] constraint values
            next_states: [batch, state_dim]

        Returns:
            Dictionary of loss values
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        costs = torch.FloatTensor(costs).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # Update Q-network
        Q_pred = self.Q_net(states, actions)
        Q_target = self.bellman_backup(states, costs, next_states)
        Q_loss = F.mse_loss(Q_pred, Q_target)

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update V-network with expectile regression
        with torch.no_grad():
            Q_values = self.Q_net(states, actions)

        V_pred = self.V_net(states)
        V_loss = self.expectile_loss(V_pred, Q_values, self.tau)

        self.V_optimizer.zero_grad()
        V_loss.backward()
        self.V_optimizer.step()

        # Soft update target network
        with torch.no_grad():
            for p, p_target in zip(
                self.V_net.parameters(),
                self.V_target.parameters(),
            ):
                p_target.data.mul_(0.995)
                p_target.data.add_(0.005 * p.data)

        return {
            "Q_loss": Q_loss.item(),
            "V_loss": V_loss.item(),
            "V_mean": V_pred.mean().item(),
        }

    def train_offline(
        self,
        dataset: Dict[str, np.ndarray],
        n_updates: int = None,
    ) -> List[Dict[str, float]]:
        """
        Train from offline dataset.

        Args:
            dataset: Dictionary with keys 'states', 'actions', 'costs', 'next_states'
            n_updates: Number of update steps

        Returns:
            List of loss dictionaries
        """
        n_updates = n_updates or self.config.n_updates
        batch_size = self.config.batch_size
        n_samples = len(dataset['states'])

        history = []

        for i in range(n_updates):
            # Sample batch
            idx = np.random.randint(0, n_samples, batch_size)

            losses = self.update(
                dataset['states'][idx],
                dataset['actions'][idx],
                dataset['costs'][idx],
                dataset['next_states'][idx],
            )

            history.append(losses)

            if (i + 1) % 1000 == 0:
                print(f"Update {i+1}/{n_updates}: "
                      f"Q_loss={losses['Q_loss']:.4f}, "
                      f"V_loss={losses['V_loss']:.4f}, "
                      f"V_mean={losses['V_mean']:.4f}")

        return history

    def is_feasible(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
    ) -> bool:
        """
        Check if state is in feasible region.

        Args:
            state: State to check
            threshold: V*_c threshold (default 0)

        Returns:
            True if V*_c(s) <= threshold
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            V = self.V_net(state).item()

        return V <= threshold

    def get_feasibility_value(self, state: np.ndarray) -> float:
        """Get V*_c(s) for a state."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.V_net(state).item()

    def save(self, path: str):
        """Save trained networks."""
        torch.save({
            'V_net': self.V_net.state_dict(),
            'Q_net': self.Q_net.state_dict(),
            'V_target': self.V_target.state_dict(),
        }, path)

    def load(self, path: str):
        """Load trained networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.V_net.load_state_dict(checkpoint['V_net'])
        self.Q_net.load_state_dict(checkpoint['Q_net'])
        self.V_target.load_state_dict(checkpoint['V_target'])


def collect_offline_data(
    env,
    n_episodes: int = 100,
    policy=None,
) -> Dict[str, np.ndarray]:
    """
    Collect offline dataset for reachability pretraining.

    Args:
        env: Environment instance
        n_episodes: Number of episodes to collect
        policy: Optional policy (random if None)

    Returns:
        Dataset dictionary
    """
    states = []
    actions = []
    costs = []
    next_states = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            if policy is not None:
                action = policy(obs)
            else:
                action = env.action_space.sample()

            next_obs, reward, cost, term, trunc, next_info = env.step(action)

            states.append(obs)
            actions.append(action)
            costs.append(cost)
            next_states.append(next_obs)

            obs = next_obs
            done = term or trunc

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'costs': np.array(costs),
        'next_states': np.array(next_states),
    }
