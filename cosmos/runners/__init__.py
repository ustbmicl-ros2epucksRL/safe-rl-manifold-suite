"""
Runners for collecting experience from environments.

Provides EPyMARL-style runners:
- EpisodeRunner: Collect complete episodes
- ParallelRunner: Collect from multiple environments in parallel
"""

from cosmos.runners.episode_runner import EpisodeRunner
from cosmos.runners.parallel_runner import ParallelRunner

__all__ = ["EpisodeRunner", "ParallelRunner"]
