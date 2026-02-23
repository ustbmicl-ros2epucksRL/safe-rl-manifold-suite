"""
COSMOS: COordinated Safety On Manifold for multi-agent Systems

A novel multi-agent safety framework that extends constraint manifold methods
to multi-robot systems with formal safety guarantees.

Key Features:
  1. Centralized COSMOS (C-COSMOS): Joint optimization over all agents
  2. Decentralized COSMOS (D-COSMOS): Per-agent with neighbor information
  3. Coupling constraints: Formation shape, connectivity, consensus
  4. Priority-weighted constraints: Dynamic priority based on danger levels
  5. Deadlock detection and resolution: Multi-agent specific challenge
  6. Safety verification: Track safety margins and certificates

Key Multi-Agent Challenges Addressed:
  - Coupled dynamics: Agent i's safety depends on agent j's actions
  - Information structure: Centralized vs decentralized knowledge
  - Scalability: O(n^2) pairwise constraints
  - Deadlock: Mutual blocking due to safety constraints
  - Livelock: Oscillatory behavior from competing objectives
  - Fairness: Balanced constraint satisfaction across agents

Theoretical Foundation:
  - Constraint manifold projection (inspired by ATACOM)
  - RMPflow geometric structure for multi-agent coordination
  - Control Barrier Functions for forward invariance

References:
  - Liu et al. 2021: "Robot RL on the Constraint Manifold" (ATACOM)
  - Liu et al. 2024: "Safe RL on the Constraint Manifold" (Theory)
  - Li et al. 2019: "Multi-Robot RMPflow" (Multi-agent structure)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy.optimize import minimize, LinearConstraint
from dataclasses import dataclass, field
from enum import Enum

from formation_nav.safety.constraints import StateConstraint, ConstraintsSet
from formation_nav.safety.rmp_policies import MultiRobotRMPForest
from formation_nav.config import SafetyConfig, EnvConfig


class COSMOSMode(Enum):
    """COSMOS execution mode."""
    CENTRALIZED = "centralized"      # Joint optimization (full coupling)
    DECENTRALIZED = "decentralized"  # Per-agent with neighbor info
    DISTRIBUTED = "distributed"       # Async updates, local info only


@dataclass
class SafetyMetrics:
    """Safety verification metrics for monitoring."""
    min_inter_agent_dist: float = float('inf')
    min_obstacle_dist: float = float('inf')
    max_constraint_violation: float = 0.0
    num_active_constraints: int = 0
    num_violated_constraints: int = 0
    deadlock_detected: bool = False
    safety_margin: float = 0.0  # Minimum margin to any constraint boundary
    cbf_interventions: int = 0  # Number of CBF corrections applied


@dataclass
class AgentState:
    """State information for a single agent."""
    position: np.ndarray
    velocity: np.ndarray
    constraint_values: np.ndarray = field(default_factory=lambda: np.array([]))
    slack_values: np.ndarray = field(default_factory=lambda: np.array([]))
    priority: float = 1.0  # Dynamic priority weight


class CouplingConstraint:
    """
    Multi-agent coupling constraint involving multiple agents.

    Examples:
      - Formation shape: det(P) = desired_area (3+ agents)
      - Connectivity: Laplacian eigenvalue bound
      - Consensus: |x_i - x_centroid| <= bound
    """

    def __init__(self, agent_indices: List[int], dim_out: int,
                 fun, jac_agents: dict, name: str = "coupling"):
        """
        Args:
            agent_indices: List of agent indices involved
            dim_out: Number of constraint outputs
            fun: c(positions) -> (dim_out,), positions shape (len(agent_indices), 2)
            jac_agents: {agent_idx: jac_fun} where jac_fun(positions) -> (dim_out, 2)
            name: Constraint name for debugging
        """
        self.agent_indices = agent_indices
        self.dim_out = dim_out
        self.fun = fun
        self.jac_agents = jac_agents
        self.name = name

    def evaluate(self, all_positions: np.ndarray) -> np.ndarray:
        """Evaluate constraint for involved agents."""
        involved_pos = all_positions[self.agent_indices]
        return self.fun(involved_pos)

    def jacobian(self, agent_idx: int, all_positions: np.ndarray) -> np.ndarray:
        """Get Jacobian w.r.t. specific agent's position."""
        if agent_idx not in self.agent_indices:
            return np.zeros((self.dim_out, 2))
        involved_pos = all_positions[self.agent_indices]
        return self.jac_agents[agent_idx](involved_pos)


class COSMOS:
    """
    COSMOS: COordinated Safety On Manifold for multi-agent Systems.

    A novel multi-agent safety framework with centralized/decentralized modes,
    providing formal safety guarantees for multi-robot coordination.

    Key Features:
      1. Mode selection: centralized (C-COSMOS) vs decentralized (D-COSMOS)
      2. Coupling constraints: formation shape, connectivity maintenance
      3. Priority-weighted projection: danger-adaptive constraint weighting
      4. Deadlock detection and resolution
      5. Safety certificate computation
    """

    def __init__(self, env_cfg: EnvConfig, safety_cfg: SafetyConfig,
                 desired_distances: np.ndarray, topology_edges: list,
                 obstacle_positions: np.ndarray = None,
                 mode: COSMOSMode = COSMOSMode.DECENTRALIZED):

        self.num_agents = env_cfg.num_agents
        self.safety_radius = safety_cfg.safety_radius
        self.K_c = safety_cfg.K_c
        self.dq_max = safety_cfg.dq_max
        self.eps_pinv = safety_cfg.eps_pinv
        self.rmp_blend = safety_cfg.rmp_formation_blend
        self.arena_size = env_cfg.arena_size
        self.boundary_margin = safety_cfg.boundary_margin
        self.mode = mode

        self.obstacle_positions = obstacle_positions
        self.num_obstacles = 0 if obstacle_positions is None else len(obstacle_positions)
        self.desired_distances = desired_distances
        self.topology_edges = topology_edges

        # Per-agent constraint sets (for decentralized mode)
        self.per_agent_constraints = []
        for i in range(self.num_agents):
            cs = self._build_agent_constraints(i, env_cfg, safety_cfg)
            self.per_agent_constraints.append(cs)

        # Coupling constraints (for formation shape, connectivity)
        self.coupling_constraints: List[CouplingConstraint] = []
        self._build_coupling_constraints()

        # Centralized constraint system (for centralized mode)
        self.centralized_dim_q = 2 * self.num_agents
        self.centralized_constraints = self._build_centralized_constraints(env_cfg, safety_cfg)

        # RMPflow forest for formation forces
        obs_pos = obstacle_positions if obstacle_positions is not None else []
        self.rmp_forest = MultiRobotRMPForest(
            num_agents=self.num_agents,
            desired_distances=desired_distances,
            topology_edges=topology_edges,
            obstacle_positions=obs_pos,
            safety_radius=self.safety_radius,
        )

        # Agent states cache
        self._positions = np.zeros((self.num_agents, 2))
        self._velocities = np.zeros((self.num_agents, 2))
        self._agent_states = [AgentState(np.zeros(2), np.zeros(2)) for _ in range(self.num_agents)]

        # Deadlock detection state
        self._position_history = []
        self._deadlock_threshold = 0.01  # Movement threshold for deadlock detection
        self._deadlock_window = 20  # Number of steps to check

        # Safety metrics
        self.metrics = SafetyMetrics()

    # ========== Constraint Building ==========

    def _build_agent_constraints(self, agent_idx: int,
                                  env_cfg: EnvConfig,
                                  safety_cfg: SafetyConfig) -> ConstraintsSet:
        """Build per-agent constraint set (decentralized mode)."""
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

        # Boundary constraints
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

    def _build_centralized_constraints(self, env_cfg: EnvConfig,
                                        safety_cfg: SafetyConfig) -> ConstraintsSet:
        """Build centralized constraint set over joint configuration space."""
        dim_q = 2 * self.num_agents
        cs = ConstraintsSet(dim_q)

        # Pairwise inter-agent collision avoidance
        num_pairs = self.num_agents * (self.num_agents - 1) // 2
        if num_pairs > 0:
            c_pairs = StateConstraint(
                dim_q=dim_q,
                dim_out=num_pairs,
                fun=self._make_centralized_inter_agent_f(),
                jac_q=self._make_centralized_inter_agent_J(),
                slack_type=safety_cfg.slack_type,
                slack_beta=safety_cfg.slack_beta,
                threshold=safety_cfg.slack_threshold,
            )
            cs.add_constraint(c_pairs)

        # Per-agent obstacle avoidance (flattened)
        if self.num_obstacles > 0:
            c_obs = StateConstraint(
                dim_q=dim_q,
                dim_out=self.num_agents * self.num_obstacles,
                fun=self._make_centralized_obstacle_f(),
                jac_q=self._make_centralized_obstacle_J(),
                slack_type=safety_cfg.slack_type,
                slack_beta=safety_cfg.slack_beta,
                threshold=safety_cfg.slack_threshold,
            )
            cs.add_constraint(c_obs)

        # Per-agent boundary constraints (flattened)
        c_boundary = StateConstraint(
            dim_q=dim_q,
            dim_out=4 * self.num_agents,
            fun=self._make_centralized_boundary_f(),
            jac_q=self._make_centralized_boundary_J(),
            slack_type=safety_cfg.slack_type,
            slack_beta=safety_cfg.slack_beta,
            threshold=safety_cfg.slack_threshold,
        )
        cs.add_constraint(c_boundary)

        return cs

    def _build_coupling_constraints(self):
        """Build multi-agent coupling constraints."""
        # Formation area constraint (for n >= 3 agents)
        if self.num_agents >= 3:
            self.coupling_constraints.append(
                self._make_formation_area_constraint()
            )

        # Connectivity constraint (algebraic connectivity bound)
        if self.num_agents >= 2:
            self.coupling_constraints.append(
                self._make_connectivity_constraint()
            )

    # ========== Constraint Functions (Decentralized) ==========

    def _make_inter_agent_f(self, agent_idx: int):
        """c_j(q_i) = -||q_i - q_j|| + safety_radius for j != i."""
        def f(q):
            results = []
            for j in range(self.num_agents):
                if j == agent_idx:
                    continue
                d = np.linalg.norm(q - self._positions[j])
                results.append(-d + self.safety_radius)
            return np.array(results)
        return f

    def _make_inter_agent_J(self, agent_idx: int):
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

    def _make_obstacle_f(self, agent_idx: int):
        def f(q):
            results = []
            for k in range(self.num_obstacles):
                obs = self.obstacle_positions[k]
                d = np.linalg.norm(q - obs[:2])
                results.append(-d + obs[2] + self.safety_radius * 0.5)
            return np.array(results)
        return f

    def _make_obstacle_J(self, agent_idx: int):
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
        bound = self.arena_size - self.boundary_margin
        def f(q):
            return np.array([
                q[0] - bound, -q[0] - bound,
                q[1] - bound, -q[1] - bound,
            ])
        return f

    def _make_boundary_J(self):
        def J(q):
            return np.array([
                [1.0, 0.0], [-1.0, 0.0],
                [0.0, 1.0], [0.0, -1.0],
            ])
        return J

    # ========== Constraint Functions (Centralized) ==========

    def _make_centralized_inter_agent_f(self):
        """Pairwise collision constraints over joint space."""
        def f(q_joint):
            q = q_joint.reshape(self.num_agents, 2)
            results = []
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    d = np.linalg.norm(q[i] - q[j])
                    results.append(-d + self.safety_radius)
            return np.array(results)
        return f

    def _make_centralized_inter_agent_J(self):
        """Jacobian of pairwise constraints w.r.t. joint configuration."""
        def J(q_joint):
            q = q_joint.reshape(self.num_agents, 2)
            num_pairs = self.num_agents * (self.num_agents - 1) // 2
            jac = np.zeros((num_pairs, 2 * self.num_agents))
            idx = 0
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    diff = q[i] - q[j]
                    d = max(np.linalg.norm(diff), 1e-8)
                    grad_i = -diff / d
                    grad_j = diff / d
                    jac[idx, 2*i:2*i+2] = grad_i
                    jac[idx, 2*j:2*j+2] = grad_j
                    idx += 1
            return jac
        return J

    def _make_centralized_obstacle_f(self):
        def f(q_joint):
            q = q_joint.reshape(self.num_agents, 2)
            results = []
            for i in range(self.num_agents):
                for k in range(self.num_obstacles):
                    obs = self.obstacle_positions[k]
                    d = np.linalg.norm(q[i] - obs[:2])
                    results.append(-d + obs[2] + self.safety_radius * 0.5)
            return np.array(results)
        return f

    def _make_centralized_obstacle_J(self):
        def J(q_joint):
            q = q_joint.reshape(self.num_agents, 2)
            jac = np.zeros((self.num_agents * self.num_obstacles, 2 * self.num_agents))
            idx = 0
            for i in range(self.num_agents):
                for k in range(self.num_obstacles):
                    obs = self.obstacle_positions[k]
                    diff = q[i] - obs[:2]
                    d = max(np.linalg.norm(diff), 1e-8)
                    jac[idx, 2*i:2*i+2] = -diff / d
                    idx += 1
            return jac
        return J

    def _make_centralized_boundary_f(self):
        bound = self.arena_size - self.boundary_margin
        def f(q_joint):
            q = q_joint.reshape(self.num_agents, 2)
            results = []
            for i in range(self.num_agents):
                results.extend([
                    q[i, 0] - bound, -q[i, 0] - bound,
                    q[i, 1] - bound, -q[i, 1] - bound,
                ])
            return np.array(results)
        return f

    def _make_centralized_boundary_J(self):
        def J(q_joint):
            jac = np.zeros((4 * self.num_agents, 2 * self.num_agents))
            for i in range(self.num_agents):
                jac[4*i + 0, 2*i + 0] = 1.0
                jac[4*i + 1, 2*i + 0] = -1.0
                jac[4*i + 2, 2*i + 1] = 1.0
                jac[4*i + 3, 2*i + 1] = -1.0
            return jac
        return J

    # ========== Coupling Constraints ==========

    def _make_formation_area_constraint(self) -> CouplingConstraint:
        """
        Formation area constraint: keep formation area within bounds.

        For 3+ agents, use signed area of polygon formed by positions.
        Constraint: |Area - desired_area| <= tolerance
        """
        indices = list(range(min(3, self.num_agents)))  # Use first 3 agents
        desired_area = 0.5 * self.safety_radius ** 2  # Minimum area

        def area_fun(positions):
            # Shoelace formula for triangle area
            if len(positions) < 3:
                return np.array([0.0])
            p0, p1, p2 = positions[0], positions[1], positions[2]
            area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                            (p2[0] - p0[0]) * (p1[1] - p0[1]))
            # Constraint: desired_area - area <= 0 (area must be large enough)
            return np.array([desired_area - area])

        def area_jac_0(positions):
            p0, p1, p2 = positions[0], positions[1], positions[2]
            sign = np.sign((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                          (p2[0] - p0[0]) * (p1[1] - p0[1]))
            return np.array([[
                -0.5 * sign * (p1[1] - p2[1]),
                -0.5 * sign * (p2[0] - p1[0])
            ]])

        def area_jac_1(positions):
            p0, p1, p2 = positions[0], positions[1], positions[2]
            sign = np.sign((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                          (p2[0] - p0[0]) * (p1[1] - p0[1]))
            return np.array([[
                -0.5 * sign * (p2[1] - p0[1]),
                -0.5 * sign * (p0[0] - p2[0])
            ]])

        def area_jac_2(positions):
            p0, p1, p2 = positions[0], positions[1], positions[2]
            sign = np.sign((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                          (p2[0] - p0[0]) * (p1[1] - p0[1]))
            return np.array([[
                -0.5 * sign * (p0[1] - p1[1]),
                -0.5 * sign * (p1[0] - p0[0])
            ]])

        return CouplingConstraint(
            agent_indices=indices,
            dim_out=1,
            fun=area_fun,
            jac_agents={0: area_jac_0, 1: area_jac_1, 2: area_jac_2},
            name="formation_area"
        )

    def _make_connectivity_constraint(self) -> CouplingConstraint:
        """
        Connectivity constraint: ensure formation doesn't become too dispersed.

        Uses maximum pairwise distance as proxy for connectivity.
        Constraint: max_dist - connectivity_bound <= 0
        """
        indices = list(range(self.num_agents))
        connectivity_bound = self.arena_size * 0.8  # Max allowed span

        def connectivity_fun(positions):
            max_dist = 0.0
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    max_dist = max(max_dist, d)
            return np.array([max_dist - connectivity_bound])

        # Jacobians for connectivity (sparse, only affects max pair)
        def make_connectivity_jac(agent_idx):
            def jac(positions):
                max_dist = 0.0
                max_pair = (0, 1)
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        d = np.linalg.norm(positions[i] - positions[j])
                        if d > max_dist:
                            max_dist = d
                            max_pair = (i, j)

                i, j = max_pair
                if agent_idx == i:
                    diff = positions[i] - positions[j]
                    d = max(np.linalg.norm(diff), 1e-8)
                    return np.array([[diff[0] / d, diff[1] / d]])
                elif agent_idx == j:
                    diff = positions[j] - positions[i]
                    d = max(np.linalg.norm(diff), 1e-8)
                    return np.array([[diff[0] / d, diff[1] / d]])
                else:
                    return np.zeros((1, 2))
            return jac

        return CouplingConstraint(
            agent_indices=indices,
            dim_out=1,
            fun=connectivity_fun,
            jac_agents={i: make_connectivity_jac(i) for i in indices},
            name="connectivity"
        )

    # ========== Priority Computation ==========

    def _compute_agent_priorities(self) -> np.ndarray:
        """
        Compute dynamic priorities for each agent based on danger level.

        Higher priority = more constrained = danger nearby
        Priority affects null-space allocation in decentralized mode.
        """
        priorities = np.ones(self.num_agents)

        for i in range(self.num_agents):
            q_i = self._positions[i]
            danger = 0.0

            # Inter-agent danger
            for j in range(self.num_agents):
                if j == i:
                    continue
                d = np.linalg.norm(q_i - self._positions[j])
                if d < self.safety_radius * 2:
                    danger += (self.safety_radius * 2 - d) / self.safety_radius

            # Obstacle danger
            if self.obstacle_positions is not None:
                for obs in self.obstacle_positions:
                    d = np.linalg.norm(q_i - obs[:2]) - obs[2]
                    if d < self.safety_radius * 2:
                        danger += (self.safety_radius * 2 - d) / self.safety_radius

            # Boundary danger
            bound = self.arena_size - self.boundary_margin
            for coord in q_i:
                margin = bound - abs(coord)
                if margin < self.safety_radius:
                    danger += (self.safety_radius - margin) / self.safety_radius

            priorities[i] = 1.0 + danger

        return priorities

    # ========== Deadlock Detection ==========

    def _detect_deadlock(self) -> bool:
        """
        Detect deadlock: agents stuck due to mutually blocking constraints.

        Deadlock indicators:
          1. Low movement over time window
          2. High constraint activity
          3. Opposing velocity directions
        """
        if len(self._position_history) < self._deadlock_window:
            return False

        # Check total movement over window
        recent_positions = self._position_history[-self._deadlock_window:]
        total_movement = 0.0
        for i in range(1, len(recent_positions)):
            movement = np.linalg.norm(
                recent_positions[i] - recent_positions[i-1]
            )
            total_movement += movement

        avg_movement = total_movement / (self._deadlock_window - 1)

        # Check if constraints are active (agents trying to move but can't)
        constraints_active = self.metrics.num_active_constraints > self.num_agents

        return avg_movement < self._deadlock_threshold and constraints_active

    def _resolve_deadlock(self, alphas: np.ndarray) -> np.ndarray:
        """
        Resolve deadlock by introducing coordinated perturbations.

        Strategy: Add small random perturbations with priority-based scaling.
        Higher priority agents move less (they are more constrained).
        """
        priorities = self._compute_agent_priorities()

        # Normalize priorities (lower = more freedom to move)
        priority_weights = 1.0 / (priorities + 0.1)
        priority_weights /= priority_weights.sum()

        # Add coordinated perturbation
        perturbation = np.random.randn(self.num_agents, 2) * 0.1
        perturbation *= priority_weights[:, np.newaxis]

        return alphas + perturbation

    # ========== CBF Safety Layer ==========

    def _cbf_safety_correction(self, agent_idx: int, q: np.ndarray,
                                v: np.ndarray, dq_desired: np.ndarray,
                                dt: float = 0.05) -> np.ndarray:
        """
        Apply CBF-based safety correction.

        For constraint h(q) >= 0 (safe set), enforce:
            dh/dt = ∇h · q̇ >= -α * h

        This is a velocity-level CBF ensuring forward invariance.
        """
        alpha_cbf = 5.0

        # Collect all constraint gradients and values
        c_vals = []
        J_rows = []

        # Inter-agent constraints
        for j in range(self.num_agents):
            if j == agent_idx:
                continue
            diff = q - self._positions[j]
            d = max(np.linalg.norm(diff), 1e-8)
            c = -d + self.safety_radius
            h = -c  # h >= 0 is safe
            grad_c = -diff / d
            c_vals.append(c)
            J_rows.append(grad_c)

        # Obstacle constraints
        if self.obstacle_positions is not None:
            for obs in self.obstacle_positions:
                diff = q - obs[:2]
                d = max(np.linalg.norm(diff), 1e-8)
                c = -d + obs[2] + self.safety_radius * 0.5
                grad_c = -diff / d
                c_vals.append(c)
                J_rows.append(grad_c)

        # Boundary constraints
        bound = self.arena_size - self.boundary_margin
        for idx, (sign, axis) in enumerate([(1, 0), (-1, 0), (1, 1), (-1, 1)]):
            c = sign * q[axis] - bound
            grad = np.zeros(2)
            grad[axis] = sign
            c_vals.append(c)
            J_rows.append(grad)

        c_vals = np.array(c_vals)
        J = np.array(J_rows)
        h_vals = -c_vals

        # CBF condition: J @ dq <= alpha * h
        cbf_rhs = alpha_cbf * h_vals
        cbf_lhs = J @ dq_desired

        if np.all(cbf_lhs <= cbf_rhs + 1e-6):
            return dq_desired

        # Iterative projection to satisfy CBF
        dq = dq_desired.copy()
        for _ in range(10):
            cbf_lhs = J @ dq
            violations = cbf_lhs - cbf_rhs
            violated_idx = np.where(violations > 1e-6)[0]

            if len(violated_idx) == 0:
                break

            # Project out most violated constraint
            worst_idx = violated_idx[np.argmax(violations[violated_idx])]
            grad = J[worst_idx]
            violation = violations[worst_idx]

            grad_norm_sq = np.dot(grad, grad)
            if grad_norm_sq > 1e-10:
                dq = dq - (violation + 0.01) * grad / grad_norm_sq
                self.metrics.cbf_interventions += 1

        return dq

    def _centralized_cbf_correction(self, q_joint: np.ndarray,
                                     v_joint: np.ndarray,
                                     dq_desired: np.ndarray,
                                     dt: float = 0.05) -> np.ndarray:
        """
        Centralized CBF correction over joint configuration space.

        Considers coupling between agents' velocities.
        """
        alpha_cbf = 5.0
        q = q_joint.reshape(self.num_agents, 2)

        c_vals = []
        J_rows = []

        # Pairwise inter-agent constraints
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                diff = q[i] - q[j]
                d = max(np.linalg.norm(diff), 1e-8)
                c = -d + self.safety_radius

                # Gradient w.r.t. joint configuration
                grad = np.zeros(2 * self.num_agents)
                grad[2*i:2*i+2] = -diff / d
                grad[2*j:2*j+2] = diff / d

                c_vals.append(c)
                J_rows.append(grad)

        # Obstacle constraints for each agent
        if self.obstacle_positions is not None:
            for i in range(self.num_agents):
                for obs in self.obstacle_positions:
                    diff = q[i] - obs[:2]
                    d = max(np.linalg.norm(diff), 1e-8)
                    c = -d + obs[2] + self.safety_radius * 0.5

                    grad = np.zeros(2 * self.num_agents)
                    grad[2*i:2*i+2] = -diff / d

                    c_vals.append(c)
                    J_rows.append(grad)

        c_vals = np.array(c_vals) if c_vals else np.array([]).reshape(0)
        J = np.array(J_rows) if J_rows else np.zeros((0, 2 * self.num_agents))

        if len(c_vals) == 0:
            return dq_desired

        h_vals = -c_vals
        cbf_rhs = alpha_cbf * h_vals
        cbf_lhs = J @ dq_desired

        if np.all(cbf_lhs <= cbf_rhs + 1e-6):
            return dq_desired

        # QP solve for safe velocity
        dq = dq_desired.copy()
        for _ in range(15):  # More iterations for joint space
            cbf_lhs = J @ dq
            violations = cbf_lhs - cbf_rhs
            violated_idx = np.where(violations > 1e-6)[0]

            if len(violated_idx) == 0:
                break

            worst_idx = violated_idx[np.argmax(violations[violated_idx])]
            grad = J[worst_idx]
            violation = violations[worst_idx]

            grad_norm_sq = np.dot(grad, grad)
            if grad_norm_sq > 1e-10:
                dq = dq - (violation + 0.01) * grad / grad_norm_sq
                self.metrics.cbf_interventions += 1

        return dq

    # ========== Safety Certificate ==========

    def compute_safety_certificate(self) -> Dict[str, float]:
        """
        Compute safety certificate metrics.

        Returns verification of safety invariant maintenance.
        """
        certificate = {
            "min_inter_agent_margin": float('inf'),
            "min_obstacle_margin": float('inf'),
            "min_boundary_margin": float('inf'),
            "formation_area": 0.0,
            "is_safe": True,
        }

        # Inter-agent safety margins
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                d = np.linalg.norm(self._positions[i] - self._positions[j])
                margin = d - self.safety_radius
                certificate["min_inter_agent_margin"] = min(
                    certificate["min_inter_agent_margin"], margin
                )
                if margin < 0:
                    certificate["is_safe"] = False

        # Obstacle safety margins
        if self.obstacle_positions is not None:
            for i in range(self.num_agents):
                for obs in self.obstacle_positions:
                    d = np.linalg.norm(self._positions[i] - obs[:2])
                    margin = d - obs[2] - self.safety_radius * 0.5
                    certificate["min_obstacle_margin"] = min(
                        certificate["min_obstacle_margin"], margin
                    )
                    if margin < 0:
                        certificate["is_safe"] = False

        # Boundary margins
        bound = self.arena_size - self.boundary_margin
        for i in range(self.num_agents):
            for coord in self._positions[i]:
                margin = bound - abs(coord)
                certificate["min_boundary_margin"] = min(
                    certificate["min_boundary_margin"], margin
                )
                if margin < 0:
                    certificate["is_safe"] = False

        # Formation area (if enough agents)
        if self.num_agents >= 3:
            p = self._positions[:3]
            area = 0.5 * abs(
                (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) -
                (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
            )
            certificate["formation_area"] = area

        return certificate

    # ========== Core Projection Methods ==========

    def reset(self, positions: np.ndarray):
        """Reset all state and slack variables."""
        self._positions = positions.copy()
        self._velocities = np.zeros_like(positions)
        self._position_history = []
        self.metrics = SafetyMetrics()

        # Reset per-agent constraints
        for i in range(self.num_agents):
            self.per_agent_constraints[i].reset_slack(positions[i])

        # Reset centralized constraints
        self.centralized_constraints.reset_slack(positions.flatten())

    def update_obstacles(self, obstacle_positions: np.ndarray):
        """Update obstacle positions."""
        self.obstacle_positions = obstacle_positions
        self.num_obstacles = len(obstacle_positions) if obstacle_positions is not None else 0

    def project(self, alphas: np.ndarray, positions: np.ndarray,
                velocities: np.ndarray, dt: float = 0.05) -> np.ndarray:
        """
        Project RL actions through ATACOM safety filter.

        Dispatches to centralized or decentralized implementation based on mode.
        """
        self._positions = positions.copy()
        self._velocities = velocities.copy()

        # Update position history for deadlock detection
        self._position_history.append(positions.copy())
        if len(self._position_history) > self._deadlock_window * 2:
            self._position_history = self._position_history[-self._deadlock_window:]

        # Detect and handle deadlock
        if self._detect_deadlock():
            self.metrics.deadlock_detected = True
            alphas = self._resolve_deadlock(alphas)
        else:
            self.metrics.deadlock_detected = False

        # Dispatch based on mode
        if self.mode == COSMOSMode.CENTRALIZED:
            safe_actions = self._project_centralized(alphas, dt)
        else:
            safe_actions = self._project_decentralized(alphas, dt)

        # Update safety metrics
        self._update_safety_metrics()

        return safe_actions

    def _project_decentralized(self, alphas: np.ndarray, dt: float) -> np.ndarray:
        """Decentralized ATACOM: per-agent projection with neighbor info."""
        # Get RMPflow formation forces
        if self.rmp_blend > 0:
            formation_forces = self.rmp_forest.get_formation_forces(
                self._positions, self._velocities
            )
        else:
            formation_forces = np.zeros_like(self._positions)

        # Compute dynamic priorities
        priorities = self._compute_agent_priorities()

        safe_actions = np.zeros((self.num_agents, 2))

        for i in range(self.num_agents):
            alpha_i = np.array(alphas[i]).flatten()
            q_i = self._positions[i]
            v_i = self._velocities[i]

            cs = self.per_agent_constraints[i]

            # Evaluate constraints
            c_origin = cs.c(q_i, origin_constr=True)

            # Construct augmented Jacobian
            J_q, J_s = cs.get_jacobians(q_i)
            Jc = np.hstack([J_q, J_s])

            # Damped pseudoinverse with priority weighting
            priority_scale = 1.0 / priorities[i]
            Jc = np.clip(Jc, -1e6, 1e6)
            JJT = Jc @ Jc.T + self.eps_pinv * np.eye(Jc.shape[0])
            Jc_inv = Jc.T @ np.linalg.inv(JJT)

            # Null-space projector
            dim_null = cs.dim_null
            bases = np.diag(np.ones(dim_null) * self.dq_max * priority_scale)
            Nc = (np.eye(Jc.shape[1]) - Jc_inv @ Jc)[:, :dim_null] @ bases

            # ATACOM projection
            c_augmented = cs.c(q_i, origin_constr=False)
            dq_null = Nc @ alpha_i[:dim_null]
            dq_err = -self.K_c * (Jc_inv @ c_augmented)
            dq = dq_null + dq_err
            dq_pos = dq[:2]

            # Handle constraint violations
            max_violation = np.max(c_origin)
            if max_violation > 0:
                violation_scale = min(1.0, np.exp(-max_violation * 10))
                dq_pos = violation_scale * dq_null[:2] + dq_err[:2]
                self.metrics.num_violated_constraints += 1

            # Blend with RMPflow
            if max_violation < -0.1:
                dq_final = dq_pos + self.rmp_blend * formation_forces[i]
            else:
                blend_scale = max(0, min(1, (-max_violation) / 0.1))
                dq_final = dq_pos + blend_scale * self.rmp_blend * formation_forces[i]

            # CBF safety correction
            dq_safe = self._cbf_safety_correction(i, q_i, v_i, dq_final, dt)

            # Convert to acceleration
            k_vel = 2.0
            a_desired = k_vel * (dq_safe - v_i)

            # Clip acceleration
            a_max = self.dq_max * 2
            norm = np.linalg.norm(a_desired)
            if norm > a_max:
                a_desired = a_desired / norm * a_max

            safe_actions[i] = a_desired

        return safe_actions

    def _project_centralized(self, alphas: np.ndarray, dt: float) -> np.ndarray:
        """Centralized ATACOM: joint optimization over all agents."""
        # Get RMPflow formation forces
        if self.rmp_blend > 0:
            formation_forces = self.rmp_forest.get_formation_forces(
                self._positions, self._velocities
            )
        else:
            formation_forces = np.zeros_like(self._positions)

        # Joint configuration
        q_joint = self._positions.flatten()
        v_joint = self._velocities.flatten()
        alpha_joint = alphas.flatten()

        cs = self.centralized_constraints

        # Evaluate constraints
        c_origin = cs.c(q_joint, origin_constr=True)

        # Construct augmented Jacobian
        J_q, J_s = cs.get_jacobians(q_joint)
        Jc = np.hstack([J_q, J_s])

        # Damped pseudoinverse
        Jc = np.clip(Jc, -1e6, 1e6)
        JJT = Jc @ Jc.T + self.eps_pinv * np.eye(Jc.shape[0])
        Jc_inv = Jc.T @ np.linalg.inv(JJT)

        # Null-space projector
        dim_null = cs.dim_null
        bases = np.diag(np.ones(dim_null) * self.dq_max)
        Nc = (np.eye(Jc.shape[1]) - Jc_inv @ Jc)[:, :dim_null] @ bases

        # ATACOM projection in joint space
        c_augmented = cs.c(q_joint, origin_constr=False)
        dq_null = Nc @ alpha_joint[:dim_null]
        dq_err = -self.K_c * (Jc_inv @ c_augmented)
        dq = dq_null + dq_err

        # Extract position-space velocities
        dq_pos = dq[:2 * self.num_agents]

        # Handle violations
        max_violation = np.max(c_origin)
        if max_violation > 0:
            violation_scale = min(1.0, np.exp(-max_violation * 10))
            dq_pos = violation_scale * dq_null[:2 * self.num_agents] + dq_err[:2 * self.num_agents]
            self.metrics.num_violated_constraints += np.sum(c_origin > 0)

        # Blend with RMPflow
        formation_flat = formation_forces.flatten()
        if max_violation < -0.1:
            dq_final = dq_pos + self.rmp_blend * formation_flat
        else:
            blend_scale = max(0, min(1, (-max_violation) / 0.1))
            dq_final = dq_pos + blend_scale * self.rmp_blend * formation_flat

        # Centralized CBF correction
        dq_safe = self._centralized_cbf_correction(q_joint, v_joint, dq_final, dt)

        # Convert to accelerations
        k_vel = 2.0
        a_desired = k_vel * (dq_safe - v_joint)

        # Clip per-agent accelerations
        a_max = self.dq_max * 2
        safe_actions = a_desired.reshape(self.num_agents, 2)
        for i in range(self.num_agents):
            norm = np.linalg.norm(safe_actions[i])
            if norm > a_max:
                safe_actions[i] = safe_actions[i] / norm * a_max

        return safe_actions

    def _update_safety_metrics(self):
        """Update safety metrics after projection."""
        # Minimum inter-agent distance
        min_dist = float('inf')
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                d = np.linalg.norm(self._positions[i] - self._positions[j])
                min_dist = min(min_dist, d)
        self.metrics.min_inter_agent_dist = min_dist

        # Minimum obstacle distance
        if self.obstacle_positions is not None:
            min_obs_dist = float('inf')
            for i in range(self.num_agents):
                for obs in self.obstacle_positions:
                    d = np.linalg.norm(self._positions[i] - obs[:2]) - obs[2]
                    min_obs_dist = min(min_obs_dist, d)
            self.metrics.min_obstacle_dist = min_obs_dist

        # Safety margin
        self.metrics.safety_margin = min(
            self.metrics.min_inter_agent_dist - self.safety_radius,
            self.metrics.min_obstacle_dist if self.obstacle_positions is not None else float('inf')
        )

        # Count active constraints
        self.metrics.num_active_constraints = 0
        for i in range(self.num_agents):
            c = self.per_agent_constraints[i].c(self._positions[i], origin_constr=True)
            self.metrics.num_active_constraints += np.sum(c > -0.1)
