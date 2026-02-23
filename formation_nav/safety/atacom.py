"""
ATACOM safety filter with CBF-based safety correction and RMPflow formation force blending.

This module provides backward-compatible AtacomSafetyFilter class.
For the full multi-agent safety implementation with centralized/decentralized modes,
deadlock detection, coupling constraints, and priority systems, see:
    formation_nav.safety.cosmos.COSMOS (COordinated Safety On Manifold for multi-agent Systems)

For each agent i, builds an independent ConstraintsSet:
  - Inter-agent collision avoidance: c = -||q_i - q_j|| + safety_radius  (N-1 constraints)
  - Obstacle avoidance: c = -||q_i - obs_k|| + (obs_r + margin)         (K constraints)
  - Boundary constraints: |q| - (arena_size - margin)                    (4 constraints)

Core projection:
  Jc = [J_q | J_s]
  Jc_inv = Jc^T @ (Jc @ Jc^T + eps*I)^{-1}
  Nc = (I - Jc_inv @ Jc)[:, :dim_null] @ diag(dq_max)
  dq = Nc @ alpha + (-K_c * Jc_inv @ c(q))

Additional CBF safety:
  For each constraint h(q) = -c(q) >= 0 (safe set), ensure:
    ḣ = -∇c · q̇ >= -α * h
  This prevents velocity that would violate constraints after integration.

RMPflow blending: dq_final = dq_safe + beta * rmp_formation_force

See docs/THEORY.md for detailed theoretical background on ATACOM, RMPflow, and MA-ATACOM.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.optimize import minimize

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

    # ---- CBF safety correction ----

    def _cbf_safety_correction(self, agent_idx: int, q: np.ndarray, v: np.ndarray,
                                dq_desired: np.ndarray, dt: float = 0.05) -> np.ndarray:
        """
        Apply CBF-based safety correction to ensure constraints remain satisfied.

        For each constraint h(q) = -c(q) >= 0, we need:
            ḣ = -∇c · q̇ >= -α * h

        This is equivalent to:
            ∇c · q̇ <= α * h = α * (-c(q))

        Args:
            agent_idx: agent index
            q: current position (2,)
            v: current velocity (2,) - not used currently
            dq_desired: desired velocity from ATACOM (2,)
            dt: time step

        Returns:
            dq_safe: safe velocity (2,)
        """
        alpha_cbf = 5.0  # CBF class-K function parameter

        # Get constraint values and Jacobians for this agent
        c_vals = []
        J_rows = []

        # Inter-agent constraints
        for j in range(self.num_agents):
            if j == agent_idx:
                continue
            diff = q - self._positions[j]
            d = max(np.linalg.norm(diff), 1e-8)
            c = -d + self.safety_radius  # c <= 0 is safe
            h = -c  # h >= 0 is safe
            grad_c = -diff / d  # gradient of c w.r.t. q

            # CBF condition: grad_c · dq <= alpha * h
            c_vals.append(c)
            J_rows.append(grad_c)

        # Obstacle constraints
        if self.obstacle_positions is not None:
            for k in range(self.num_obstacles):
                obs = self.obstacle_positions[k]
                diff = q - obs[:2]
                d = max(np.linalg.norm(diff), 1e-8)
                c = -d + obs[2]  # c <= 0 is safe
                grad_c = -diff / d

                c_vals.append(c)
                J_rows.append(grad_c)

        # Boundary constraints
        bound = self.arena_size - self.boundary_margin
        # +x: q[0] - bound <= 0
        c_vals.append(q[0] - bound)
        J_rows.append(np.array([1.0, 0.0]))
        # -x: -q[0] - bound <= 0
        c_vals.append(-q[0] - bound)
        J_rows.append(np.array([-1.0, 0.0]))
        # +y: q[1] - bound <= 0
        c_vals.append(q[1] - bound)
        J_rows.append(np.array([0.0, 1.0]))
        # -y: -q[1] - bound <= 0
        c_vals.append(-q[1] - bound)
        J_rows.append(np.array([0.0, -1.0]))

        c_vals = np.array(c_vals)
        J = np.array(J_rows)  # (num_constraints, 2)
        h_vals = -c_vals  # safe set h >= 0

        # Check CBF conditions: J @ dq <= alpha * h
        cbf_rhs = alpha_cbf * h_vals

        # If all CBF conditions are satisfied, return desired velocity
        cbf_lhs = J @ dq_desired
        if np.all(cbf_lhs <= cbf_rhs + 1e-6):
            return dq_desired

        # Otherwise, solve QP to find closest safe velocity
        # min ||dq - dq_desired||^2 s.t. J @ dq <= alpha * h
        # Using simple iterative projection for efficiency

        dq = dq_desired.copy()
        for _ in range(10):  # iteration limit
            cbf_lhs = J @ dq
            violations = cbf_lhs - cbf_rhs
            violated_idx = np.where(violations > 1e-6)[0]

            if len(violated_idx) == 0:
                break

            # Project out the most violated constraint
            worst_idx = violated_idx[np.argmax(violations[violated_idx])]
            grad = J[worst_idx]
            violation = violations[worst_idx]

            # Project dq to satisfy the constraint
            grad_norm_sq = np.dot(grad, grad)
            if grad_norm_sq > 1e-10:
                dq = dq - (violation + 0.01) * grad / grad_norm_sq

        return dq

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

    def project(self, alphas, positions, velocities, dt: float = 0.05):
        """
        Project RL actions through ATACOM safety filter.

        For a double-integrator system (v += a*dt, p += v*dt), we compute:
        1. Desired velocity from ATACOM null-space projection
        2. CBF-based safety correction on desired velocity
        3. Convert safe velocity to acceleration: a = (v_safe - v_current) / dt

        Args:
            alphas: (num_agents, 2) raw RL policy outputs in [-1, 1]
            positions: (num_agents, 2) current positions
            velocities: (num_agents, 2) current velocities
            dt: time step

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
            v_i = velocities[i]

            cs = self.per_agent_constraints[i]
            # Note: Do NOT reset slack every step - only in reset()

            # Evaluate original constraint values to check safety
            c_origin = cs.c(q_i, origin_constr=True)

            # Construct augmented Jacobian [J_q | J_s]
            J_q, J_s = cs.get_jacobians(q_i)
            Jc = np.hstack([J_q, J_s])  # (dim_out, dim_q + dim_slack)

            # Damped pseudoinverse
            Jc = np.clip(Jc, -1e6, 1e6)
            JJT = Jc @ Jc.T + self.eps_pinv * np.eye(Jc.shape[0])
            Jc_inv = Jc.T @ np.linalg.inv(JJT)  # (dim_q + dim_slack, dim_out)

            # Null-space projector
            dim_null = cs.dim_null  # should be 2
            bases = np.diag(np.ones(dim_null) * self.dq_max)
            Nc = (np.eye(Jc.shape[1]) - Jc_inv @ Jc)[:, :dim_null] @ bases

            # ATACOM projection (outputs desired velocity)
            c_augmented = cs.c(q_i, origin_constr=False)  # with slack
            dq_null = Nc @ alpha_i[:dim_null]
            dq_err = -self.K_c * (Jc_inv @ c_augmented)
            dq = dq_null + dq_err

            # Extract position-space velocity command (first dim_q components)
            dq_pos = dq[:2]

            # Additional safety: if constraint is violated, prioritize correction
            max_violation = np.max(c_origin)
            if max_violation > 0:
                violation_scale = min(1.0, np.exp(-max_violation * 10))
                dq_pos = violation_scale * dq_null[:2] + dq_err[:2]

            # Blend with RMPflow formation force (only if safe)
            if max_violation < -0.1:
                dq_final = dq_pos + self.rmp_blend * formation_forces[i]
            else:
                blend_scale = max(0, min(1, (-max_violation) / 0.1))
                dq_final = dq_pos + blend_scale * self.rmp_blend * formation_forces[i]

            # Apply CBF safety correction on velocity
            dq_safe = self._cbf_safety_correction(i, q_i, v_i, dq_final)

            # Convert safe velocity to acceleration
            # For double integrator: v_next = v + a*dt, so a = (v_desired - v) / dt
            # But we need to be careful: dq_safe is the desired velocity
            # We use a proportional controller to track the desired velocity
            k_vel = 2.0  # velocity tracking gain
            a_desired = k_vel * (dq_safe - v_i)

            # Clip acceleration
            a_max = self.dq_max * 2  # max acceleration
            norm = np.linalg.norm(a_desired)
            if norm > a_max:
                a_desired = a_desired / norm * a_max

            safe_actions[i] = a_desired

        return safe_actions
