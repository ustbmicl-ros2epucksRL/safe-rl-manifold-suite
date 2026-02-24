"""
COSMOS Safety Filter

COordinated Safety On Manifold for multi-agent Systems.

Combines:
- ATACOM constraint manifold projection
- RMPflow formation control
- CBF safety correction
- Deadlock detection and resolution

Reference:
    Liu et al., "Safe RL on the Constraint Manifold", IEEE T-RO 2024
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import numpy as np

from cosmos.registry import SAFETY_REGISTRY
from cosmos.safety.base import BaseSafetyFilter, SafetyConfig

# Import original COSMOS
from formation_nav.safety import COSMOS as OriginalCOSMOS, COSMOSMode
from formation_nav.config import EnvConfig, SafetyConfig as OriginalSafetyConfig


class COSMOSFilterMode(Enum):
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
        """
        Args:
            env_cfg: Environment configuration.
            safety_cfg: Safety configuration.
            mode: "centralized" or "decentralized".
            desired_distances: Formation distances matrix.
            topology_edges: List of (i, j) edges in formation graph.
            obstacle_positions: Initial obstacle positions.
        """
        super().__init__(env_cfg, safety_cfg, **kwargs)

        # Convert to original config format
        if isinstance(env_cfg, dict):
            orig_env_cfg = EnvConfig()
            for k, v in env_cfg.items():
                if hasattr(orig_env_cfg, k):
                    setattr(orig_env_cfg, k, v)
        else:
            orig_env_cfg = env_cfg

        orig_safety_cfg = OriginalSafetyConfig()
        if self.safety_cfg:
            for attr in ['safety_radius', 'boundary_margin', 'K_c', 'dq_max',
                        'eps_pinv', 'rmp_formation_blend', 'slack_type',
                        'slack_beta', 'slack_threshold']:
                if hasattr(self.safety_cfg, attr):
                    setattr(orig_safety_cfg, attr, getattr(self.safety_cfg, attr))

        # Parse mode
        cosmos_mode = (
            COSMOSMode.CENTRALIZED if mode == "centralized"
            else COSMOSMode.DECENTRALIZED
        )

        # Default values if not provided
        if desired_distances is None:
            num_agents = getattr(orig_env_cfg, 'num_agents', 4)
            desired_distances = np.ones((num_agents, num_agents)) * 1.0
            np.fill_diagonal(desired_distances, 0)

        if topology_edges is None:
            num_agents = len(desired_distances)
            topology_edges = [(i, j) for i in range(num_agents)
                             for j in range(i+1, num_agents)]

        if obstacle_positions is None:
            obstacle_positions = np.zeros((0, 3))

        # Create original COSMOS
        self._filter = OriginalCOSMOS(
            env_cfg=orig_env_cfg,
            safety_cfg=orig_safety_cfg,
            desired_distances=desired_distances,
            topology_edges=topology_edges,
            obstacle_positions=obstacle_positions,
            mode=cosmos_mode
        )

        self.mode = mode

    def reset(self, constraint_info: Dict[str, Any]):
        """Reset filter for new episode."""
        positions = constraint_info.get("positions", None)
        if positions is not None:
            self._filter.reset(positions)

        # Update obstacles if provided
        obstacles = constraint_info.get("obstacles", None)
        if obstacles is not None:
            self._filter.update_obstacles(obstacles)

    def project(
        self,
        actions: np.ndarray,
        constraint_info: Dict[str, Any],
        dt: float = 0.05
    ) -> np.ndarray:
        """Project actions to safe space."""
        positions = constraint_info["positions"]
        velocities = constraint_info["velocities"]

        return self._filter.project(actions, positions, velocities, dt=dt)

    def update(self, constraint_info: Dict[str, Any]):
        """Update constraint information."""
        obstacles = constraint_info.get("obstacles", None)
        if obstacles is not None:
            self._filter.update_obstacles(obstacles)

    def get_safety_margin(self, constraint_info: Dict[str, Any]) -> float:
        """Compute minimum safety margin."""
        positions = constraint_info.get("positions", None)
        if positions is None:
            return float('inf')

        num_agents = len(positions)
        min_dist = float('inf')

        # Check inter-agent distances
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                margin = dist - self.safety_cfg.safety_radius
                min_dist = min(min_dist, margin)

        return min_dist

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
        arena_size: Arena half-width (default: 5.0)
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
        """
        Args:
            env_cfg: Environment configuration.
            safety_cfg: Safety configuration.
            desired_distances: Formation distances matrix (unused in CBF).
            topology_edges: Formation graph edges (unused in CBF).
            obstacle_positions: Initial obstacle positions.
        """
        super().__init__(env_cfg, safety_cfg, **kwargs)

        # Get config values
        if isinstance(env_cfg, dict):
            self.arena_size = env_cfg.get('arena_size', 5.0)
            self.num_agents = env_cfg.get('num_agents', 4)
        else:
            self.arena_size = getattr(env_cfg, 'arena_size', 5.0)
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

        # State
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
        """
        Project actions to satisfy CBF constraints using QP.

        Solves: min ||u - u_nom||^2
                s.t. h_dot + alpha * h >= 0 for all constraints

        Args:
            actions: Nominal actions from RL policy (num_agents, act_dim)
            constraint_info: Dict with positions, velocities, obstacles
            dt: Time step

        Returns:
            safe_actions: Safe actions satisfying CBF constraints
        """
        positions = constraint_info["positions"]  # (num_agents, 2 or 3)
        velocities = constraint_info.get("velocities", np.zeros_like(positions))

        num_agents = len(positions)
        act_dim = actions.shape[1] if actions.ndim > 1 else 2

        safe_actions = actions.copy()

        # Process each agent
        for i in range(num_agents):
            pos_i = positions[i][:2]  # Use 2D position
            vel_i = velocities[i][:2] if velocities[i].shape[0] >= 2 else velocities[i]
            u_nom = actions[i][:2] if actions[i].shape[0] >= 2 else actions[i]

            # Collect CBF constraints: A @ u >= b
            A_list = []
            b_list = []

            # Inter-agent collision avoidance
            for j in range(num_agents):
                if i == j:
                    continue
                pos_j = positions[j][:2]

                # h(x) = ||p_i - p_j||^2 - r^2
                diff = pos_i - pos_j
                dist_sq = np.dot(diff, diff)
                dist = np.sqrt(dist_sq)

                if dist < 1e-6:
                    continue

                h = dist - self.safety_radius

                # Gradient: dh/dp_i = (p_i - p_j) / ||p_i - p_j||
                grad_h = diff / dist

                # h_dot = grad_h^T @ v_i (assuming v_i = u_i for single integrator)
                # Constraint: grad_h^T @ u_i + alpha * h >= 0
                A_list.append(grad_h)
                b_list.append(-self.cbf_alpha * h)

            # Obstacle avoidance
            for obs in self.obstacle_positions:
                obs_pos = obs[:2]
                diff = pos_i - obs_pos
                dist = np.linalg.norm(diff)

                if dist < 1e-6:
                    continue

                h = dist - self.obstacle_radius
                grad_h = diff / dist

                A_list.append(grad_h)
                b_list.append(-self.cbf_alpha * h)

            # Boundary constraints
            boundary = self.arena_size - self.boundary_margin

            # x >= -boundary: h = x + boundary, grad_h = [1, 0]
            h_x_min = pos_i[0] + boundary
            A_list.append(np.array([1.0, 0.0]))
            b_list.append(-self.cbf_alpha * h_x_min)

            # x <= boundary: h = boundary - x, grad_h = [-1, 0]
            h_x_max = boundary - pos_i[0]
            A_list.append(np.array([-1.0, 0.0]))
            b_list.append(-self.cbf_alpha * h_x_max)

            # y >= -boundary
            h_y_min = pos_i[1] + boundary
            A_list.append(np.array([0.0, 1.0]))
            b_list.append(-self.cbf_alpha * h_y_min)

            # y <= boundary
            h_y_max = boundary - pos_i[1]
            A_list.append(np.array([0.0, -1.0]))
            b_list.append(-self.cbf_alpha * h_y_max)

            # Solve QP: min ||u - u_nom||^2 s.t. A @ u >= b
            if len(A_list) > 0:
                A = np.stack(A_list)  # (num_constraints, 2)
                b = np.array(b_list)  # (num_constraints,)

                u_safe = self._solve_cbf_qp(u_nom, A, b)
                safe_actions[i][:2] = u_safe

        return safe_actions

    def _solve_cbf_qp(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        max_iters: int = 10
    ) -> np.ndarray:
        """
        Solve CBF-QP using projected gradient descent.

        min ||u - u_nom||^2 s.t. A @ u >= b

        Args:
            u_nom: Nominal control (2,)
            A: Constraint matrix (num_constraints, 2)
            b: Constraint bounds (num_constraints,)
            max_iters: Maximum iterations

        Returns:
            u_safe: Safe control satisfying constraints
        """
        u = u_nom.copy()

        for _ in range(max_iters):
            # Check constraint violations
            violations = b - A @ u  # positive means violated

            if np.all(violations <= 1e-6):
                break

            # Find most violated constraint
            idx = np.argmax(violations)
            if violations[idx] <= 1e-6:
                break

            # Project onto constraint: A[idx] @ u = b[idx]
            a = A[idx]
            violation = violations[idx]

            # u_new = u + (violation / ||a||^2) * a
            a_norm_sq = np.dot(a, a)
            if a_norm_sq > 1e-8:
                u = u + (violation / a_norm_sq) * a

        return u

    def update(self, constraint_info: Dict[str, Any]):
        """Update constraint information."""
        obstacles = constraint_info.get("obstacles", None)
        if obstacles is not None:
            self.obstacle_positions = obstacles

    def get_safety_margin(self, constraint_info: Dict[str, Any]) -> float:
        """Compute minimum safety margin (barrier function value)."""
        positions = constraint_info.get("positions", None)
        if positions is None:
            return float('inf')

        num_agents = len(positions)
        min_h = float('inf')

        # Inter-agent distances
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = np.linalg.norm(positions[i][:2] - positions[j][:2])
                h = dist - self.safety_radius
                min_h = min(min_h, h)

        # Obstacle distances
        for i in range(num_agents):
            for obs in self.obstacle_positions:
                dist = np.linalg.norm(positions[i][:2] - obs[:2])
                h = dist - self.obstacle_radius
                min_h = min(min_h, h)

        # Boundary distances
        boundary = self.arena_size - self.boundary_margin
        for i in range(num_agents):
            pos = positions[i][:2]
            min_h = min(min_h, pos[0] + boundary)  # x >= -boundary
            min_h = min(min_h, boundary - pos[0])  # x <= boundary
            min_h = min(min_h, pos[1] + boundary)  # y >= -boundary
            min_h = min(min_h, boundary - pos[1])  # y <= boundary

        return min_h

    def is_safe(self, constraint_info: Dict[str, Any]) -> bool:
        """Check if current state is safe (all barrier functions positive)."""
        return self.get_safety_margin(constraint_info) > 0
