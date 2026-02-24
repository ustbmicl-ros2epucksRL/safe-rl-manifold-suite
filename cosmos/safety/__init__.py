"""
Safety Filters for Safe Multi-Agent RL

Provides safety filters that project RL actions to safe actions:
- COSMOS: Constraint manifold projection with RMPflow
- CBF: Control Barrier Functions
- ATACOM: Action Transformation for Constrained Motion
- RMPflow: Riemannian Motion Policies

Available filters:
- cosmos: COSMOS constraint manifold projection
- cbf: Control Barrier Function filter
- none: No safety filtering
"""

from cosmos.safety.base import BaseSafetyFilter

__all__ = [
    "BaseSafetyFilter",
    "CBFFilter",
    "COSMOSFilter",
    "COSMOS",
    "COSMOSMode",
    "StateConstraint",
    "ConstraintsSet",
    "RMPRoot",
    "RMPNode",
]

# Import safety filters to trigger registration
def _register_filters():
    """Import all safety filter modules to register them."""
    try:
        from cosmos.safety import cosmos_filter
    except ImportError:
        pass

_register_filters()

# Lazy imports for optional components
def __getattr__(name):
    if name == "CBFFilter":
        from cosmos.safety.cosmos_filter import CBFFilter
        return CBFFilter
    elif name == "COSMOSFilter":
        from cosmos.safety.cosmos_filter import COSMOSFilter
        return COSMOSFilter
    elif name == "COSMOS":
        from cosmos.safety.atacom import COSMOS
        return COSMOS
    elif name == "COSMOSMode":
        from cosmos.safety.atacom import COSMOSMode
        return COSMOSMode
    elif name == "StateConstraint":
        from cosmos.safety.constraints import StateConstraint
        return StateConstraint
    elif name == "ConstraintsSet":
        from cosmos.safety.constraints import ConstraintsSet
        return ConstraintsSet
    elif name == "RMPRoot":
        from cosmos.safety.rmp_tree import RMPRoot
        return RMPRoot
    elif name == "RMPNode":
        from cosmos.safety.rmp_tree import RMPNode
        return RMPNode
    raise AttributeError(f"module 'cosmos.safety' has no attribute '{name}'")
