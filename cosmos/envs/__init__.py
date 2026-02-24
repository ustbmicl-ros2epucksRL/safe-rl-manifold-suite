"""
Multi-Agent Environments

Provides a unified interface for multi-agent environments with:
- Standard Gymnasium API
- Shared observation space for CTDE
- Constraint information for safety filters

Available environments:
- formation_nav: Multi-robot formation navigation (built-in)
- vmas: Vectorized Multi-Agent Simulator (GPU accelerated)
- safety_gym: Safety-Gymnasium constrained RL
- mujoco: MuJoCo physics simulation
- ma_mujoco: Multi-Agent MuJoCo
- webots_epuck: Webots E-puck robots
- epuck_sim: Simplified E-puck simulation
"""

from cosmos.envs.base import BaseMultiAgentEnv
from cosmos.envs.env_wrapper import MultiAgentEnvWrapper, make_env
from cosmos.envs.formations import FormationTopology

__all__ = ["BaseMultiAgentEnv", "MultiAgentEnvWrapper", "make_env", "FormationTopology"]

# Import environments to trigger registration
def _register_envs():
    """Import all environment modules to register them."""
    # Built-in environment
    try:
        from cosmos.envs import formation_nav
    except ImportError:
        pass

    # VMAS (Vectorized Multi-Agent Simulator)
    try:
        from cosmos.envs import vmas_wrapper
    except ImportError:
        pass

    # Safety-Gym / Safety-Gymnasium
    try:
        from cosmos.envs import safety_gym_wrapper
    except ImportError:
        pass

    # MuJoCo
    try:
        from cosmos.envs import mujoco_wrapper
    except ImportError:
        pass

    # Webots E-puck
    try:
        from cosmos.envs import webots_wrapper
    except ImportError:
        pass

_register_envs()
