"""
COSMOS Safety Filter

COordinated Safety On Manifold for multi-agent Systems.

Combines:
- ATACOM constraint manifold projection
- RMPflow formation control
- CBF safety correction

Reference:
    Liu et al., "Safe RL on the Constraint Manifold", IEEE T-RO 2024
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import numpy as np

from cosmos.registry import SAFETY_REGISTRY
from cosmos.safety.base import BaseSafetyFilter, SafetyConfig


class COSMOSMode(Enum):
    """COSMOS operating modes."""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"


@SAFETY_REGISTRY.register("cosmos", aliases=["atacom", "manifold"])
class COSMOSFilter(BaseSafetyFilter):
    """
    COSMOS: COordinated Safety On Manifold for multi-agent Systems.

    Projects RL actions to safe actions that satisfy:
    - Inter-agent collision avoidance
    - Obstacle avoidance
    - Boundary constraints
    - Formation maintenance (soft constraint via RMPflow)

    Uses ATACOM null-space projection for hard safety guarantees.

    Config options:
        safety_radius: Minimum inter-agent distance (default: 0.5)
        boundary_margin: Distance from arena boundary (default: 0.5)
        K_c: Constraint gain (default: 50.0)
        dq_max: Maximum velocity (default: 0.8)
        rmp_formation_blend: RMPflow blend factor (default: 0.3)
        mode: "centralized" or "decentralized" (default: "decentralized")
    """

    def __init__(
        self,
        env_cfg: Any,
        safety_cfg: Optional[SafetyConfig] = None,
        mode: str = "decentralized",
        desired_distances: Optional[np.ndarray] = None,
        topology_edges: Optional[List] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(env_cfg, safety_cfg, **kwargs)

        # Get config values
        if isinstance(env_cfg, dict):
            self.arena_size = env_cfg.get('arena_size', 10.0)
            self.num_agents = env_cfg.get('num_agents', 4)
        else:
            self.arena_size = getattr(env_cfg, 'arena_size', 10.0)
            self.num_agents = getattr(env_cfg, 'num_agents', 4)

        # Safety parameters
        self.safety_radius = getattr(safety_cfg, 'safety_radius', 0.5) if safety_cfg else 0.5
        self.boundary_margin = getattr(safety_cfg, 'boundary_margin', 0.5) if safety_cfg else 0.5
        self.K_c = getattr(safety_cfg, 'K_c', 50.0) if safety_cfg else 50.0
        self.dq_max = getattr(safety_cfg, 'dq_max', 0.8) if safety_cfg else 0.8
        self.eps_pinv = getattr(safety_cfg, 'eps_pinv', 1e-4) if safety_cfg else 1e-4
        self.rmp_blend = getattr(safety_cfg, 'rmp_formation_blend', 0.3) if safety_cfg else 0.3

        self.mode = COSMOSMode(mode) if isinstance(mode, str) else mode
        self.desired_distances = desired_distances
        self.topology_edges = topology_edges
        self.obstacle_positions = obstacle_positions if obstacle_positions is not None else np.zeros((0, 3))

    def reset(self, constraint_info: Dict[str, Any]):
        """Reset filter for new episode."""
        obstacles = constraint_info.get("obstacles", None)
        if obstacles is not None:
            self.obstacle_positions = obstacles

        # Update desired distances if provided
        if "desired_distances" in constraint_info:
            self.desired_distances = constraint_info["desired_distances"]
        if "topology_edges" in constraint_info:
            self.topology_edges = constraint_info["topology_edges"]

    def project(
        self,
        actions: np.ndarray,
        constraint_info: Dict[str, Any],
        dt: float = 0.05
    ) -> np.ndarray:
        """
        Project actions to safe space using ATACOM null-space projection.

        Core mechanism:
            Jc = constraint Jacobian
            Nc = null-space projector = I - Jc^+ @ Jc
            u_safe = Nc @ u_nom - K_c * Jc^+ @ c(q)

        Args:
            actions: Nominal actions from RL policy (num_agents, act_dim)
            constraint_info: Dict with positions, velocities, obstacles
            dt: Time step

        Returns:
            safe_actions: Safe actions satisfying constraints
        """
        positions = constraint_info["positions"][:, :2]  # (num_agents, 2)
        velocities = constraint_info.get("velocities", np.zeros_like(positions))[:, :2]

        num_agents = len(positions)
        safe_actions = np.zeros_like(actions)

        for i in range(num_agents):
            # Build constraint Jacobian and values for agent i
            Jc, c_vals = self._build_constraints(i, positions)

            if len(c_vals) == 0:
                safe_actions[i] = actions[i]
                continue

            # Stack into matrix form
            Jc = np.array(Jc)  # (num_constraints, 2)
            c_vals = np.array(c_vals)  # (num_constraints,)

            # Compute pseudoinverse with damping
            JcT = Jc.T
            Jc_pinv = JcT @ np.linalg.inv(Jc @ JcT + self.eps_pinv * np.eye(len(c_vals)))

            # Null-space projector
            Nc = np.eye(2) - Jc_pinv @ Jc

            # Project action to null space + constraint correction
            u_nom = actions[i][:2] if actions[i].shape[0] >= 2 else actions[i]
            u_safe = Nc @ u_nom - self.K_c * Jc_pinv @ c_vals

            # Clip to max velocity
            u_norm = np.linalg.norm(u_safe)
            if u_norm > self.dq_max:
                u_safe = u_safe / u_norm * self.dq_max

            safe_actions[i][:2] = u_safe

        return safe_actions

    def _build_constraints(self, agent_id: int, positions: np.ndarray):
        """Build constraint Jacobian and values for one agent."""
        Jc_list = []
        c_list = []
        pos_i = positions[agent_id]

        # Inter-agent collision avoidance
        for j in range(len(positions)):
            if j == agent_id:
                continue
            pos_j = positions[j]
            diff = pos_i - pos_j
            dist = np.linalg.norm(diff)

            if dist < 1e-6:
                continue

            # Constraint: c = safety_radius - dist <= 0 (violated when positive)
            c = self.safety_radius - dist

            # Only include if constraint is active (close to or violating boundary)
            if c > -0.5:  # Active within 0.5m of boundary
                # Jacobian: dc/dq_i = -(pos_i - pos_j) / dist
                Jc = -diff / dist
                Jc_list.append(Jc)
                c_list.append(max(c, 0))  # Only correct violations

        # Obstacle avoidance
        for obs in self.obstacle_positions:
            obs_pos = obs[:2]
            obs_r = obs[2] if len(obs) > 2 else 0.3
            diff = pos_i - obs_pos
            dist = np.linalg.norm(diff)

            if dist < 1e-6:
                continue

            c = (obs_r + self.safety_radius * 0.5) - dist

            if c > -0.5:
                Jc = -diff / dist
                Jc_list.append(Jc)
                c_list.append(max(c, 0))

        # Boundary constraints
        boundary = self.arena_size - self.boundary_margin

        # x_min: c = -boundary - x <= 0
        c_xmin = -pos_i[0] - boundary
        if c_xmin > -0.5:
            Jc_list.append(np.array([-1.0, 0.0]))
            c_list.append(max(c_xmin, 0))

        # x_max: c = x - boundary <= 0
        c_xmax = pos_i[0] - boundary
        if c_xmax > -0.5:
            Jc_list.append(np.array([1.0, 0.0]))
            c_list.append(max(c_xmax, 0))

        # y_min
        c_ymin = -pos_i[1] - boundary
        if c_ymin > -0.5:
            Jc_list.append(np.array([0.0, -1.0]))
            c_list.append(max(c_ymin, 0))

        # y_max
        c_ymax = pos_i[1] - boundary
        if c_ymax > -0.5:
            Jc_list.append(np.array([0.0, 1.0]))
            c_list.append(max(c_ymax, 0))

        return Jc_list, c_list

    def update(self, constraint_info: Dict[str, Any]):
        """Update constraint information."""
        if "obstacles" in constraint_info:
            self.obstacle_positions = constraint_info["obstacles"]

    def get_safety_margin(self, constraint_info: Dict[str, Any]) -> float:
        """Compute minimum safety margin."""
        positions = constraint_info.get("positions", None)
        if positions is None:
            return float('inf')

        positions = positions[:, :2]
        num_agents = len(positions)
        min_margin = float('inf')

        # Inter-agent distances
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                margin = dist - self.safety_radius
                min_margin = min(min_margin, margin)

        # Obstacle distances
        for i in range(num_agents):
            for obs in self.obstacle_positions:
                dist = np.linalg.norm(positions[i] - obs[:2])
                margin = dist - (obs[2] if len(obs) > 2 else 0.3) - self.safety_radius * 0.5
                min_margin = min(min_margin, margin)

        return min_margin

    def is_safe(self, constraint_info: Dict[str, Any]) -> bool:
        """Check if current state is safe."""
        return self.get_safety_margin(constraint_info) > 0


@SAFETY_REGISTRY.register("cbf")
class CBFFilter(BaseSafetyFilter):
    """
    Control Barrier Function Safety Filter.

    Projects actions to satisfy CBF constraints:
        h_dot(x, u) + alpha * h(x) >= 0

    Where h(x) is a barrier function (e.g., distance - safety_radius).

    Config options:
        safety_radius: Minimum inter-agent distance (default: 0.5)
        obstacle_radius: Minimum obstacle distance (default: 0.3)
        boundary_margin: Distance from arena boundary (default: 0.5)
        cbf_alpha: CBF class-K function parameter (default: 1.0)
        arena_size: Arena size (default: 10.0)
    """

    def __init__(
        self,
        env_cfg: Any,
        safety_cfg: Optional[SafetyConfig] = None,
        desired_distances: Optional[np.ndarray] = None,
        topology_edges: Optional[List] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(env_cfg, safety_cfg, **kwargs)

        # Get config values
        if isinstance(env_cfg, dict):
            self.arena_size = env_cfg.get('arena_size', 10.0)
            self.num_agents = env_cfg.get('num_agents', 4)
        else:
            self.arena_size = getattr(env_cfg, 'arena_size', 10.0)
            self.num_agents = getattr(env_cfg, 'num_agents', 4)

        if safety_cfg:
            self.safety_radius = getattr(safety_cfg, 'safety_radius', 0.5)
            self.obstacle_radius = getattr(safety_cfg, 'obstacle_radius', 0.3)
            self.boundary_margin = getattr(safety_cfg, 'boundary_margin', 0.5)
            self.cbf_alpha = getattr(safety_cfg, 'cbf_alpha', 1.0)
        else:
            self.safety_radius = 0.5
            self.obstacle_radius = 0.3
            self.boundary_margin = 0.5
            self.cbf_alpha = 1.0

        self.obstacle_positions = obstacle_positions if obstacle_positions is not None else np.zeros((0, 3))

    def reset(self, constraint_info: Dict[str, Any]):
        """Reset filter for new episode."""
        obstacles = constraint_info.get("obstacles", None)
        if obstacles is not None:
            self.obstacle_positions = obstacles

    def project(
        self,
        actions: np.ndarray,
        constraint_info: Dict[str, Any],
        dt: float = 0.05
    ) -> np.ndarray:
        """Project actions to satisfy CBF constraints."""
        positions = constraint_info["positions"][:, :2]
        num_agents = len(positions)
        safe_actions = actions.copy()

        for i in range(num_agents):
            pos_i = positions[i]
            u_nom = actions[i][:2] if actions[i].shape[0] >= 2 else actions[i]

            A_list = []
            b_list = []

            # Inter-agent collision avoidance
            for j in range(num_agents):
                if i == j:
                    continue
                pos_j = positions[j]
                diff = pos_i - pos_j
                dist = np.linalg.norm(diff)

                if dist < 1e-6:
                    continue

                h = dist - self.safety_radius
                grad_h = diff / dist

                A_list.append(grad_h)
                b_list.append(-self.cbf_alpha * h)

            # Obstacle avoidance
            for obs in self.obstacle_positions:
                diff = pos_i - obs[:2]
                dist = np.linalg.norm(diff)

                if dist < 1e-6:
                    continue

                h = dist - self.obstacle_radius
                grad_h = diff / dist

                A_list.append(grad_h)
                b_list.append(-self.cbf_alpha * h)

            # Boundary constraints
            boundary = self.arena_size - self.boundary_margin

            A_list.append(np.array([1.0, 0.0]))
            b_list.append(-self.cbf_alpha * (pos_i[0] + boundary))

            A_list.append(np.array([-1.0, 0.0]))
            b_list.append(-self.cbf_alpha * (boundary - pos_i[0]))

            A_list.append(np.array([0.0, 1.0]))
            b_list.append(-self.cbf_alpha * (pos_i[1] + boundary))

            A_list.append(np.array([0.0, -1.0]))
            b_list.append(-self.cbf_alpha * (boundary - pos_i[1]))

            if len(A_list) > 0:
                A = np.stack(A_list)
                b = np.array(b_list)
                u_safe = self._solve_cbf_qp(u_nom, A, b)
                safe_actions[i][:2] = u_safe

        return safe_actions

    def _solve_cbf_qp(self, u_nom: np.ndarray, A: np.ndarray, b: np.ndarray, max_iters: int = 10) -> np.ndarray:
        """Solve CBF-QP using projected gradient descent."""
        u = u_nom.copy()

        for _ in range(max_iters):
            violations = b - A @ u
            if np.all(violations <= 1e-6):
                break

            idx = np.argmax(violations)
            if violations[idx] <= 1e-6:
                break

            a = A[idx]
            violation = violations[idx]
            a_norm_sq = np.dot(a, a)
            if a_norm_sq > 1e-8:
                u = u + (violation / a_norm_sq) * a

        return u

    def update(self, constraint_info: Dict[str, Any]):
        """Update constraint information."""
        if "obstacles" in constraint_info:
            self.obstacle_positions = constraint_info["obstacles"]

    def get_safety_margin(self, constraint_info: Dict[str, Any]) -> float:
        """Compute minimum safety margin."""
        positions = constraint_info.get("positions", None)
        if positions is None:
            return float('inf')

        positions = positions[:, :2]
        num_agents = len(positions)
        min_h = float('inf')

        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_h = min(min_h, dist - self.safety_radius)

        for i in range(num_agents):
            for obs in self.obstacle_positions:
                dist = np.linalg.norm(positions[i] - obs[:2])
                min_h = min(min_h, dist - self.obstacle_radius)

        boundary = self.arena_size - self.boundary_margin
        for i in range(num_agents):
            pos = positions[i]
            min_h = min(min_h, pos[0] + boundary)
            min_h = min(min_h, boundary - pos[0])
            min_h = min(min_h, pos[1] + boundary)
            min_h = min(min_h, boundary - pos[1])

        return min_h

    def is_safe(self, constraint_info: Dict[str, Any]) -> bool:
        """Check if current state is safe."""
        return self.get_safety_margin(constraint_info) > 0
