"""
Multi-Agent Reinforcement Learning Algorithms

Provides a unified interface for MARL algorithms with:
- Parameter sharing support
- CTDE (Centralized Training, Decentralized Execution)
- On-policy and off-policy methods

Available algorithms:
- mappo: Multi-Agent PPO with parameter sharing
"""

from cosmos.algos.base import BaseMARLAlgo

__all__ = ["BaseMARLAlgo"]

# Import algorithms to trigger registration
def _register_algos():
    """Import all algorithm modules to register them."""
    try:
        from cosmos.algos import mappo
    except ImportError:
        pass

_register_algos()
