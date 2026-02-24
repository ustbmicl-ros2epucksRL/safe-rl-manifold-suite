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

    TODO: Implement CBF-based safety filter.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CBF filter is not implemented yet")

    def reset(self, constraint_info):
        raise NotImplementedError

    def project(self, actions, constraint_info, dt=0.05):
        raise NotImplementedError

    def update(self, constraint_info):
        raise NotImplementedError
