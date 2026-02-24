"""
Multi-Agent Environments

Provides a unified interface for multi-agent environments with:
- Standard Gymnasium API
- Shared observation space for CTDE
- Constraint information for safety filters

Available environments:
- formation_nav: Multi-robot formation navigation
"""

from cosmos.envs.base import BaseMultiAgentEnv

__all__ = ["BaseMultiAgentEnv"]

# Import environments to trigger registration
def _register_envs():
    """Import all environment modules to register them."""
    try:
        from cosmos.envs import formation_nav
    except ImportError:
        pass

_register_envs()
