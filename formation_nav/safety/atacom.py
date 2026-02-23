"""
ATACOM safety filter with RMPflow formation force blending.

For each agent i, builds an independent ConstraintsSet:
  - Inter-agent collision avoidance: c = -||q_i - q_j|| + safety_radius  (N-1 constraints)
  - Obstacle avoidance: c = -||q_i - obs_k|| + (obs_r + margin)         (K constraints)
  - Boundary constraints: |q| - (arena_size - margin)                    (4 constraints)

Core projection:
  Jc = [J_q | J_s]
  Jc_inv = Jc^T @ (Jc @ Jc^T + eps*I)^{-1}
  Nc = (I - Jc_inv @ Jc)[:, :dim_null] @ diag(dq_max)
  dq = Nc @ alpha + (-K_c * Jc_inv @ c(q))

RMPflow blending: dq_final = dq_safe + beta * rmp_formation_force
"""

import numpy as np
from typing import Optional

from formation_nav.safety.constraints import StateConstraint, ConstraintsSet
from formation_nav.safety.rmp_policies import MultiRobotRMPForest
from formation_nav.config import SafetyConfig, EnvConfig


class AtacomSafetyFilter:
    """Per-agent ATACOM safety filter with RMPflow formation guidance."""

    def __init__(self, env_cfg: EnvConfig, safety_cfg: SafetyConfig,
                 desired_distances: np.ndarray, topology_edges: list,
                 obstacle_positions: np.ndarray = None):
        self.num_agents = env_cfg.num_agents
        self.safety_radius = safety_cfg.safety_radius
        self.K_c = safety_cfg.K_c
        self.dq_max = safety_cfg.dq_max
        self.eps_pinv = safety_cfg.eps_pinv
        self.rmp_blend = safety_cfg.rmp_formation_blend
        self.arena_size = env_cfg.arena_size
        self.boundary_margin = safety_cfg.boundary_margin

        self.obstacle_positions = obstacle_positions  # (K, 3): x, y, radius
        self.num_obstacles = 0 if obstacle_positions is None else len(obstacle_positions)

        # Build per-agent constraint sets
        self.per_agent_constraints = []
        for i in range(self.num_agents):
            cs = self._build_constraints_for_agent(i, env_cfg, safety_cfg)
            self.per_agent_constraints.append(cs)

        # RMPflow forest for formation forces
        obs_pos = obstacle_positions if obstacle_positions is not None else []
        self.rmp_forest = MultiRobotRMPForest(
            num_agents=self.num_agents,
            desired_distances=desired_distances,
            topology_edges=topology_edges,
            obstacle_positions=obs_pos,
            safety_radius=self.safety_radius,
        )

        # Cache for other agent positions
        self._positions = np.zeros((self.num_agents, 2))
        self._velocities = np.zeros((self.num_agents, 2))

    def _build_constraints_for_agent(self, agent_idx, env_cfg, safety_cfg):
        """Build ConstraintsSet for a single agent."""
        dim_q = 2
        cs = ConstraintsSet(dim_q)

        # Inter-agent collision avoidance
        n_others = self.num_agents - 1
        if n_others > 0:
            c_agent = StateConstraint(
                dim_q=dim_q,
                dim_out=n_others,
                fun=self._make_inter_agent_f(agent_idx),
                jac_q=self._make_inter_agent_J(agent_idx),
                slack_type=safety_cfg.slack_type,
                slack_beta=safety_cfg.slack_beta,
                threshold=safety_cfg.slack_threshold,
            )
            cs.add_constraint(c_agent)

        # Obstacle avoidance
        if self.num_obstacles > 0:
            c_obs = StateConstraint(
                dim_q=dim_q,
                dim_out=self.num_obstacles,
                fun=self._make_obstacle_f(agent_idx),
                jac_q=self._make_obstacle_J(agent_idx),
                slack_type=safety_cfg.slack_type,
                slack_beta=safety_cfg.slack_beta,
                threshold=safety_cfg.slack_threshold,
            )
            cs.add_constraint(c_obs)

        # Boundary constraints (4: +x, -x, +y, -y)
        c_boundary = StateConstraint(
            dim_q=dim_q,
            dim_out=4,
            fun=self._make_boundary_f(),
            jac_q=self._make_boundary_J(),
            slack_type=safety_cfg.slack_type,
            slack_beta=safety_cfg.slack_beta,
            threshold=safety_cfg.slack_threshold,
        )
        cs.add_constraint(c_boundary)

        return cs

    # ---- Constraint function factories ----

    def _make_inter_agent_f(self, agent_idx):
        """c_j(q_i) = -||q_i - q_j|| + safety_radius  for j != i."""
        def f(q):
            results = []
            for j in range(self.num_agents):
                if j == agent_idx:
                    continue
                d = np.linalg.norm(q - self._positions[j])
                results.append(-d + self.safety_radius)
            return np.array(results)
        return f

    def _make_inter_agent_J(self, agent_idx):
        """Jacobian of inter-agent constraints w.r.t. q_i."""
        def J(q):
            rows = []
            for j in range(self.num_agents):
                if j == agent_idx:
                    continue
                diff = q - self._positions[j]
                d = max(np.linalg.norm(diff), 1e-8)
                rows.append(-diff / d)
            return np.array(rows)
        return J

    def _make_obstacle_f(self, agent_idx):
        """c_k(q_i) = -||q_i - obs_k|| + (obs_r + margin)."""
        def f(q):
            results = []
            for k in range(self.num_obstacles):
                obs = self.obstacle_positions[k]
                d = np.linalg.norm(q - obs[:2])
                results.append(-d + obs[2])  # obs[2] is radius
            return np.array(results)
        return f

    def _make_obstacle_J(self, agent_idx):
        def J(q):
            rows = []
            for k in range(self.num_obstacles):
                obs = self.obstacle_positions[k]
                diff = q - obs[:2]
                d = max(np.linalg.norm(diff), 1e-8)
                rows.append(-diff / d)
            return np.array(rows)
        return J

    def _make_boundary_f(self):
        """4 boundary constraints: q_x - bound, -q_x - bound, q_y - bound, -q_y - bound."""
        bound = self.arena_size - self.boundary_margin

        def f(q):
            return np.array([
                q[0] - bound,     # +x boundary
                -q[0] - bound,    # -x boundary
                q[1] - bound,     # +y boundary
                -q[1] - bound,    # -y boundary
            ])
        return f

    def _make_boundary_J(self):
        bound = self.arena_size - self.boundary_margin

        def J(q):
            return np.array([
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ])
        return J

    # ---- Core projection ----

    def reset(self, positions):
        """Reset slack variables for all agents."""
        self._positions = positions.copy()
        for i in range(self.num_agents):
            self.per_agent_constraints[i].reset_slack(positions[i])

    def update_obstacles(self, obstacle_positions):
        """Update obstacle positions."""
        self.obstacle_positions = obstacle_positions
        self.num_obstacles = len(obstacle_positions) if obstacle_positions is not None else 0

    def project(self, alphas, positions, velocities):
        """
        Project RL actions through ATACOM safety filter.

        Args:
            alphas: (num_agents, 2) raw RL policy outputs in [-1, 1]
            positions: (num_agents, 2) current positions
            velocities: (num_agents, 2) current velocities

        Returns:
            safe_actions: (num_agents, 2) safe accelerations
        """
        self._positions = positions.copy()
        self._velocities = velocities.copy()

        # Get RMPflow formation forces for blending
        if self.rmp_blend > 0:
            formation_forces = self.rmp_forest.get_formation_forces(positions, velocities)
        else:
            formation_forces = np.zeros_like(positions)

        safe_actions = np.zeros((self.num_agents, 2))

        for i in range(self.num_agents):
            alpha_i = np.array(alphas[i]).flatten()
            q_i = positions[i]

            cs = self.per_agent_constraints[i]
            cs.reset_slack(q_i)

            # Construct augmented Jacobian [J_q | J_s]
            J_q, J_s = cs.get_jacobians(q_i)
            Jc = np.hstack([J_q, J_s])  # (dim_out, dim_q + dim_slack)

            # Damped pseudoinverse (clip Jc to avoid overflow)
            Jc = np.clip(Jc, -1e6, 1e6)
            JJT = Jc @ Jc.T + self.eps_pinv * np.eye(Jc.shape[0])
            Jc_inv = Jc.T @ np.linalg.inv(JJT)  # (dim_q + dim_slack, dim_out)

            # Null-space projector
            dim_null = cs.dim_null  # should be 2
            bases = np.diag(np.ones(dim_null) * self.dq_max)
            Nc = (np.eye(Jc.shape[1]) - Jc_inv @ Jc)[:, :dim_null] @ bases

            # ATACOM projection
            dq_null = Nc @ alpha_i[:dim_null]
            dq_err = -self.K_c * (Jc_inv @ cs.c(q_i, origin_constr=False))
            dq = dq_null + dq_err

            # Extract position-space action (first dim_q components)
            dq_pos = dq[:2]

            # Blend with RMPflow formation force
            dq_final = dq_pos + self.rmp_blend * formation_forces[i]

            safe_actions[i] = dq_final

        return safe_actions
