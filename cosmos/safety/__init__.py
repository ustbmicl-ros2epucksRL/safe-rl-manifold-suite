"""
Safety Filters for Safe Multi-Agent RL

Provides safety filters that project RL actions to safe actions:
- COSMOS: Constraint manifold projection with RMPflow
- CBF: Control Barrier Functions
- None: Pass-through (no safety filter)

Available filters:
- cosmos: COSMOS constraint manifold projection
- cbf: Control Barrier Function filter
- none: No safety filtering
"""

from cosmos.safety.base import BaseSafetyFilter

__all__ = ["BaseSafetyFilter"]

# Import safety filters to trigger registration
def _register_filters():
    """Import all safety filter modules to register them."""
    try:
        from cosmos.safety import cosmos_filter
    except ImportError:
        pass

_register_filters()
