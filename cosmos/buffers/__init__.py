"""
Experience Buffers for MARL

Provides:
- ReplayBuffer: Off-policy experience replay
- EpisodeReplayBuffer: Episode-based storage for value decomposition methods (QMIX)
- RolloutBuffer: On-policy rollout storage with GAE (MAPPO, PPO)
"""

from cosmos.buffers.replay_buffer import ReplayBuffer, EpisodeReplayBuffer
from cosmos.buffers.rollout_buffer import RolloutBuffer, RolloutData
from cosmos.registry import BUFFER_REGISTRY

# Register buffers
BUFFER_REGISTRY.register_module("replay", ReplayBuffer, aliases=["off_policy"])
BUFFER_REGISTRY.register_module("episode_replay", EpisodeReplayBuffer, aliases=["qmix_buffer"])
BUFFER_REGISTRY.register_module("rollout", RolloutBuffer, aliases=["on_policy", "ppo_buffer"])

__all__ = ["ReplayBuffer", "EpisodeReplayBuffer", "RolloutBuffer", "RolloutData"]
