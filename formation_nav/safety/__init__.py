"""Safety module for multi-robot formation navigation.

Core Components:
  - COSMOS: COordinated Safety On Manifold for multi-agent Systems
  - AtacomSafetyFilter: Basic ATACOM implementation (backward compatible)
  - RMPflow policies: Geometric motion policies for formation control
"""

from formation_nav.safety.constraints import StateConstraint, ConstraintsSet
from formation_nav.safety.atacom import AtacomSafetyFilter
from formation_nav.safety.cosmos import COSMOS, COSMOSMode, SafetyMetrics
from formation_nav.safety.rmp_policies import MultiRobotRMPForest

__all__ = [
    "StateConstraint",
    "ConstraintsSet",
    "AtacomSafetyFilter",
    "COSMOS",
    "COSMOSMode",
    "SafetyMetrics",
    "MultiRobotRMPForest",
]
