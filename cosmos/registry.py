"""
Component Registry for dynamic registration and instantiation.

Provides a unified way to register and build environments, algorithms,
and safety filters by name.

Example:
    >>> from cosmos.registry import ENV_REGISTRY
    >>>
    >>> @ENV_REGISTRY.register("my_env")
    >>> class MyEnv(BaseMultiAgentEnv):
    >>>     pass
    >>>
    >>> env = ENV_REGISTRY.build("my_env", cfg=config)
"""

from typing import Dict, Type, Any, Optional, Callable, List
import logging

logger = logging.getLogger(__name__)


class Registry:
    """
    A generic registry for registering and building components.

    Supports:
    - Decorator-based registration
    - Building instances by name
    - Listing registered components
    - Aliases for components
    """

    def __init__(self, name: str):
        """
        Args:
            name: Registry name (e.g., "environment", "algorithm")
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        replace: bool = False
    ):
        """
        Decorator to register a class.

        Args:
            name: Registration name. If None, uses class name.
            aliases: Optional list of alternative names.
            replace: If True, silently replace existing registration.

        Returns:
            Decorator function.

        Example:
            >>> @REGISTRY.register("my_component")
            >>> class MyComponent:
            >>>     pass
        """
        def decorator(cls: Type) -> Type:
            key = name if name is not None else cls.__name__

            if key in self._registry and not replace:
                logger.warning(
                    f"Overwriting {self.name} '{key}' "
                    f"(was {self._registry[key]}, now {cls})"
                )

            self._registry[key] = cls
            logger.debug(f"Registered {self.name}: {key} -> {cls.__name__}")

            # Register aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = key
                    logger.debug(f"Registered alias: {alias} -> {key}")

            return cls

        return decorator

    def register_module(self, name: str, cls: Type, aliases: Optional[List[str]] = None):
        """
        Programmatically register a class (non-decorator version).

        Args:
            name: Registration name.
            cls: The class to register.
            aliases: Optional list of alternative names.
        """
        self._registry[name] = cls
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

    def build(self, name: str, **kwargs) -> Any:
        """
        Build an instance by name.

        Args:
            name: Registered component name or alias.
            **kwargs: Arguments passed to the constructor.

        Returns:
            Instance of the registered class.

        Raises:
            KeyError: If name is not registered.
        """
        # Resolve alias
        key = self._aliases.get(name, name)

        if key not in self._registry:
            available = self.list()
            raise KeyError(
                f"Unknown {self.name}: '{name}'. "
                f"Available: {available}"
            )

        cls = self._registry[key]
        logger.debug(f"Building {self.name} '{key}' with kwargs: {list(kwargs.keys())}")

        return cls(**kwargs)

    def get(self, name: str) -> Optional[Type]:
        """
        Get registered class by name without instantiating.

        Args:
            name: Registered component name or alias.

        Returns:
            The registered class or None if not found.
        """
        key = self._aliases.get(name, name)
        return self._registry.get(key)

    def list(self) -> List[str]:
        """List all registered component names."""
        return list(self._registry.keys())

    def list_with_aliases(self) -> Dict[str, List[str]]:
        """List all components with their aliases."""
        result = {name: [] for name in self._registry.keys()}
        for alias, name in self._aliases.items():
            result[name].append(alias)
        return result

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        key = self._aliases.get(name, name)
        return key in self._registry

    def __len__(self) -> int:
        """Return number of registered components."""
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self.name}, components={self.list()})"


# =============================================================================
# Global Registries
# =============================================================================

ENV_REGISTRY = Registry("environment")
"""Registry for multi-agent environments."""

ALGO_REGISTRY = Registry("algorithm")
"""Registry for MARL algorithms."""

SAFETY_REGISTRY = Registry("safety_filter")
"""Registry for safety filters."""

BUFFER_REGISTRY = Registry("buffer")
"""Registry for experience replay buffers."""


def list_all() -> Dict[str, List[str]]:
    """List all registered components across all registries."""
    return {
        "environments": ENV_REGISTRY.list(),
        "algorithms": ALGO_REGISTRY.list(),
        "safety_filters": SAFETY_REGISTRY.list(),
        "buffers": BUFFER_REGISTRY.list(),
    }
