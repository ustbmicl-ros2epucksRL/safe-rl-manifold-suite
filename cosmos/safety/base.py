"""
Base class for Safety Filters.

Safety filters project potentially unsafe RL actions to safe actions
that satisfy safety constraints.

Supports:
- Constraint manifold projection (ATACOM, COSMOS)
- Control Barrier Functions (CBF)
- Safety shielding
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SafetyConfig:
    """Default safety filter configuration."""
    # Safety margins
    safety_radius: float = 0.5
    boundary_margin: float = 0.5
    obstacle_margin: float = 0.3

    # ATACOM/COSMOS specific
    K_c: float = 50.0
    dq_max: float = 0.8
    eps_pinv: float = 1e-4
    rmp_formation_blend: float = 0.3

    # Slack variables
    slack_type: str = "softcorner"
    slack_beta: float = 20.0
    slack_threshold: float = 0.01

    # CBF specific
    cbf_alpha: float = 1.0
    cbf_gamma: float = 0.1

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SafetyConfig":
        """Create config from dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


class BaseSafetyFilter(ABC):
    """
    Abstract base class for safety filters.

    Safety filters transform potentially unsafe RL actions into
    safe actions that satisfy safety constraints.

    Subclasses must implement:
    - reset(): Reset filter state
    - project(): Project actions to safe space
    - update(): Update constraint information
    """

    def __init__(
        self,
        env_cfg: Any,
        safety_cfg: Optional[SafetyConfig] = None,
        **kwargs
    ):
        """
        Args:
            env_cfg: Environment configuration.
            safety_cfg: Safety filter configuration.
            **kwargs: Additional arguments.
        """
        self.env_cfg = env_cfg

        if safety_cfg is None:
            self.safety_cfg = SafetyConfig()
        elif isinstance(safety_cfg, dict):
            self.safety_cfg = SafetyConfig.from_dict(safety_cfg)
        else:
            self.safety_cfg = safety_cfg

    @abstractmethod
    def reset(self, constraint_info: Dict[str, Any]):
        """
        Reset filter state for new episode.

        Args:
            constraint_info: Dict with constraint-relevant state from env.
        """
        pass

    @abstractmethod
    def project(
        self,
        actions: np.ndarray,
        constraint_info: Dict[str, Any],
        dt: float = 0.05
    ) -> np.ndarray:
        """
        Project actions to safe space.

        Args:
            actions: Raw RL actions, shape (num_agents, act_dim)
            constraint_info: Dict with current constraint state
            dt: Time step

        Returns:
            safe_actions: Safe actions, shape (num_agents, act_dim)
        """
        pass

    @abstractmethod
    def update(self, constraint_info: Dict[str, Any]):
        """
        Update constraint information (e.g., obstacle positions).

        Args:
            constraint_info: Dict with updated constraint state
        """
        pass

    # =========================================================================
    # Optional methods
    # =========================================================================

    def get_safety_margin(self, constraint_info: Dict[str, Any]) -> float:
        """
        Compute current safety margin.

        Returns minimum distance to constraint boundary.
        """
        return float('inf')

    def is_safe(self, constraint_info: Dict[str, Any]) -> bool:
        """Check if current state is safe."""
        return self.get_safety_margin(constraint_info) > 0

    def get_constraint_values(
        self,
        constraint_info: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Get current constraint values.

        Returns:
            Constraint values where positive means satisfied.
        """
        return None


class NoSafetyFilter(BaseSafetyFilter):
    """
    Pass-through filter that doesn't modify actions.

    Useful for:
    - Baseline comparisons without safety
    - Environments that handle safety internally
    """

    def __init__(self, env_cfg: Any = None, safety_cfg: Any = None, **kwargs):
        # Don't call super().__init__ to avoid requiring configs
        self.env_cfg = env_cfg
        self.safety_cfg = safety_cfg

    def reset(self, constraint_info: Dict[str, Any]):
        """No-op reset."""
        pass

    def project(
        self,
        actions: np.ndarray,
        constraint_info: Dict[str, Any],
        dt: float = 0.05
    ) -> np.ndarray:
        """Return actions unchanged."""
        return actions

    def update(self, constraint_info: Dict[str, Any]):
        """No-op update."""
        pass

    def get_safety_margin(self, constraint_info: Dict[str, Any]) -> float:
        """Return infinite margin (always safe)."""
        return float('inf')

    def is_safe(self, constraint_info: Dict[str, Any]) -> bool:
        """Always return True."""
        return True


# Register NoSafetyFilter
from cosmos.registry import SAFETY_REGISTRY
SAFETY_REGISTRY.register_module("none", NoSafetyFilter, aliases=["passthrough", "no_filter"])
