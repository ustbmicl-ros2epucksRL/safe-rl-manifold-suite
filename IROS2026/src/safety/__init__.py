"""
Safety module for IROS 2026 paper.

Components:
    - ManifoldFilter: Constraint manifold projection (Section III-A)
    - DistanceFilter: Distance-based velocity scaling (practical version)
    - ReachabilityPretrainer: HJ reachability analysis (Section III-B)
"""

from .manifold_filter import ManifoldFilter, FilterResult
from .distance_filter import DistanceFilter, DistanceFilterResult
from .reachability import ReachabilityPretrainer, FeasibilityValueNet

__all__ = [
    "ManifoldFilter",
    "FilterResult",
    "DistanceFilter",
    "DistanceFilterResult",
    "ReachabilityPretrainer",
    "FeasibilityValueNet",
]
