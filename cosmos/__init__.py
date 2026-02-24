"""
COSMOS: COordinated Safety On Manifold for multi-agent Systems

A modular framework for safe multi-agent reinforcement learning with:
- Pluggable environments (formation navigation, MPE, traffic, etc.)
- Pluggable MARL algorithms (MAPPO, QMIX, MADDPG, etc.)
- Pluggable safety filters (COSMOS, CBF, shielding, etc.)
- Hydra-based configuration management
- WandB experiment tracking

Usage:
    python -m cosmos.train env=formation_nav algo=mappo safety=cosmos
"""

from cosmos.registry import ENV_REGISTRY, ALGO_REGISTRY, SAFETY_REGISTRY

__version__ = "0.1.0"
__all__ = ["ENV_REGISTRY", "ALGO_REGISTRY", "SAFETY_REGISTRY"]
