"""
CBF-based action corrector for multi-agent formation tasks.

This module provides a lightweight, training-time safety filter that adjusts
velocity actions using control barrier inspired repulsive terms.
"""

from __future__ import annotations

import numpy as np
import torch


class CBFCorrector:
    """Apply collision-avoidance correction to actions in batched MA envs."""

    def __init__(self, num_agents: int, num_envs: int, device: str, config: dict | None = None):
        self.num_agents = int(num_agents)
        self.num_envs = int(num_envs)
        self.device = device
        self.config = config or {}

        self.cbf_enabled = bool(self.config.get("use_cbf", True))
        # Compatibility with existing mappo_rmp Runner, which checks
        # `self.rmp_corrector.rmp_enabled` before applying correction.
        self.rmp_enabled = self.cbf_enabled
        self.safety_radius = float(self.config.get("cbf_safety_radius", 0.35))
        self.cbf_gain = float(self.config.get("cbf_gain", 0.8))
        self.max_correction = float(self.config.get("cbf_max_correction", 0.6))
        self.cbf_weight = float(self.config.get("cbf_weight", 1.0))

    def get_agent_positions_from_env(self, envs, n_envs: int):
        agent_positions = []
        agent_velocities = []

        if hasattr(envs, "envs") and len(envs.envs) > 0:
            for env_idx in range(min(n_envs, len(envs.envs))):
                env = envs.envs[env_idx]
                if hasattr(env, "env") and hasattr(env.env, "task") and hasattr(env.env.task, "agent"):
                    agent = env.env.task.agent
                    env_positions = []
                    env_velocities = []
                    for i in range(self.num_agents):
                        pos_attr = getattr(agent, f"pos_{i}", None)
                        vel_attr = getattr(agent, f"vel_{i}", None)
                        pos = pos_attr[:2] if pos_attr is not None else np.zeros(2)
                        vel = vel_attr[:2] if vel_attr is not None else np.zeros(2)
                        env_positions.append(pos)
                        env_velocities.append(vel)
                    if env_idx == 0:
                        agent_positions = [[pos] for pos in env_positions]
                        agent_velocities = [[vel] for vel in env_velocities]
                    else:
                        for i in range(self.num_agents):
                            agent_positions[i].append(env_positions[i])
                            agent_velocities[i].append(env_velocities[i])

            for i in range(self.num_agents):
                agent_positions[i] = np.array(agent_positions[i], dtype=np.float64)
                agent_velocities[i] = np.array(agent_velocities[i], dtype=np.float64)
        else:
            for _ in range(self.num_agents):
                agent_positions.append(np.zeros((n_envs, 2), dtype=np.float64))
                agent_velocities.append(np.zeros((n_envs, 2), dtype=np.float64))

        return agent_positions, agent_velocities

    def _pairwise_barrier_correction(self, positions: np.ndarray):
        """Compute per-agent correction in one env from pairwise barrier terms."""
        corr = np.zeros_like(positions, dtype=np.float64)
        r = self.safety_radius
        eps = 1e-8
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    continue
                d_vec = positions[i] - positions[j]
                dist = np.linalg.norm(d_vec) + eps
                if dist >= r:
                    continue
                # Barrier-inspired repulsive field: stronger near boundary crossing.
                strength = self.cbf_gain * (1.0 / dist - 1.0 / r) / (dist * dist)
                corr[i] += strength * (d_vec / dist)

        norm = np.linalg.norm(corr, axis=1, keepdims=True) + eps
        scale = np.minimum(1.0, self.max_correction / norm)
        corr = corr * scale
        return corr

    def apply_correction(self, actions: list, agent_positions: list, agent_velocities: list):
        del agent_velocities  # current CBF corrector only uses positions

        if not self.cbf_enabled:
            return actions

        corrected_actions = [[] for _ in range(self.num_agents)]
        for env_idx in range(self.num_envs):
            pos_env = np.zeros((self.num_agents, 2), dtype=np.float64)
            for i in range(self.num_agents):
                pos_env[i] = agent_positions[i][env_idx][:2]

            corr_env = self._pairwise_barrier_correction(pos_env)
            for i in range(self.num_agents):
                act = actions[i][env_idx]
                act_np = act.detach().cpu().numpy() if isinstance(act, torch.Tensor) else np.asarray(act)
                corrected = act_np.copy()
                usable_dim = min(2, corrected.shape[0])
                corrected[:usable_dim] = corrected[:usable_dim] + self.cbf_weight * corr_env[i][:usable_dim]
                corrected_actions[i].append(
                    torch.tensor(corrected, dtype=torch.float32, device=self.device)
                )

        return [torch.stack(agent_acts) for agent_acts in corrected_actions]
