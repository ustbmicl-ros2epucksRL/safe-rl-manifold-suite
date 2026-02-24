"""
Multi-Agent Reinforcement Learning Algorithms

Provides a unified interface for MARL algorithms with:
- Parameter sharing support
- CTDE (Centralized Training, Decentralized Execution)
- On-policy and off-policy methods

Available algorithms:
- mappo: Multi-Agent PPO with parameter sharing
- qmix: QMIX value decomposition with monotonic mixing
- maddpg: Multi-Agent DDPG with centralized critics
"""

from cosmos.algos.base import BaseMARLAlgo, OnPolicyAlgo, OffPolicyAlgo, AlgoConfig

__all__ = ["BaseMARLAlgo", "OnPolicyAlgo", "OffPolicyAlgo", "AlgoConfig"]

# Import algorithms to trigger registration
def _register_algos():
    """Import all algorithm modules to register them."""
    try:
        from cosmos.algos import mappo
    except ImportError:
        pass

    try:
        from cosmos.algos import qmix
    except ImportError:
        pass

    try:
        from cosmos.algos import maddpg
    except ImportError:
        pass

_register_algos()
