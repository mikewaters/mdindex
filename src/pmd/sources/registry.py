"""Source registry for URI scheme routing.

This module provides a registry that maps URI schemes to document source
implementations, allowing the system to resolve any URI to its appropriate
source handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from urllib.parse import urlparse

from ..core.exceptions import PMDError
from .base import DocumentSource, SourceConfig

if TYPE_CHECKING:
    pass


# =============================================================================
# Exceptions
# =============================================================================


class RegistryError(PMDError):
    """Base exception for registry operations."""

    pass


class UnknownSchemeError(RegistryError):
    """No source registered for the given URI scheme."""

    def __init__(self, scheme: str, uri: str):
        self.scheme = scheme
        self.uri = uri
        super().__init__(
            f"No source registered for scheme '{scheme}' (URI: {uri})"
        )


class SourceCreationError(RegistryError):
    """Failed to create source instance."""

    def __init__(self, scheme: str, reason: str):
        self.scheme = scheme
        self.reason = reason
        super().__init__(f"Failed to create source for scheme '{scheme}': {reason}")


# =============================================================================
# Type Definitions
# =============================================================================

# Factory function type: takes SourceConfig, returns DocumentSource
SourceFactory = Callable[[SourceConfig], DocumentSource]


# =============================================================================
# Registry Implementation
# =============================================================================


class SourceRegistry:
    """Registry that routes URI schemes to source implementations.

    The registry maintains a mapping from URI schemes (like 'file', 'http',
    'entity') to factory functions that create source instances.

    Example:
        registry = SourceRegistry()
        registry.register('file', FileSystemSource)
        registry.register('http', HTTPSource)
        registry.register('https', HTTPSource)

        # Resolve a URI to a source
        source = registry.resolve('file:///path/to/docs', SourceConfig(...))

        # Check supported schemes
        schemes = registry.supported_schemes()  # ['file', 'http', 'https']
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._handlers: dict[str, SourceFactory] = {}

    def register(
        self,
        scheme: str,
        factory: SourceFactory,
        *,
        override: bool = False,
    ) -> None:
        """Register a source factory for a URI scheme.

        Args:
            scheme: URI scheme to handle (e.g., 'file', 'http', 'entity').
                    Case-insensitive.
            factory: Callable that takes SourceConfig and returns DocumentSource.
                     This can be a class (used as constructor) or a factory function.
            override: If True, allow overriding an existing registration.

        Raises:
            RegistryError: If scheme is already registered and override=False.
        """
        scheme_lower = scheme.lower()
        if scheme_lower in self._handlers and not override:
            raise RegistryError(
                f"Scheme '{scheme}' is already registered. "
                f"Use override=True to replace."
            )
        self._handlers[scheme_lower] = factory

    def unregister(self, scheme: str) -> bool:
        """Remove a registered scheme.

        Args:
            scheme: URI scheme to remove.

        Returns:
            True if the scheme was registered and removed, False if not found.
        """
        scheme_lower = scheme.lower()
        if scheme_lower in self._handlers:
            del self._handlers[scheme_lower]
            return True
        return False

    def resolve(self, uri: str, config: SourceConfig | None = None) -> DocumentSource:
        """Create a source instance for the given URI.

        Parses the URI to extract the scheme, looks up the registered factory,
        and creates a source instance with the provided configuration.

        Args:
            uri: URI to resolve (e.g., 'file:///path', 'https://docs.example.com')
            config: Configuration for the source. If None, creates a default
                    SourceConfig with just the URI.

        Returns:
            DocumentSource instance for handling the URI.

        Raises:
            UnknownSchemeError: If no source is registered for the URI's scheme.
            SourceCreationError: If the source factory fails.
        """
        scheme = self._extract_scheme(uri)

        factory = self._handlers.get(scheme)
        if factory is None:
            raise UnknownSchemeError(scheme, uri)

        # Create config if not provided
        if config is None:
            config = SourceConfig(uri=uri)

        try:
            return factory(config)
        except Exception as e:
            raise SourceCreationError(scheme, str(e)) from e

    def is_registered(self, scheme: str) -> bool:
        """Check if a scheme is registered.

        Args:
            scheme: URI scheme to check.

        Returns:
            True if the scheme has a registered handler.
        """
        return scheme.lower() in self._handlers

    def supported_schemes(self) -> list[str]:
        """Get list of all registered schemes.

        Returns:
            Sorted list of registered URI schemes.
        """
        return sorted(self._handlers.keys())

    def _extract_scheme(self, uri: str) -> str:
        """Extract the scheme from a URI.

        Handles both standard URIs (http://...) and bare file paths.

        Args:
            uri: URI or path to parse.

        Returns:
            Lowercase scheme string.
        """
        # Try standard URI parsing
        parsed = urlparse(uri)
        if parsed.scheme:
            return parsed.scheme.lower()

        # Fallback: treat bare paths as file:// URIs
        if uri.startswith("/") or (len(uri) > 1 and uri[1] == ":"):
            # Unix absolute path or Windows path (C:\...)
            return "file"

        # Could be a relative path or something else
        # Default to file scheme for backward compatibility
        return "file"


# =============================================================================
# Default Registry
# =============================================================================

# Global default registry instance
# Sources are registered here when their modules are imported
_default_registry: SourceRegistry | None = None


def get_default_registry() -> SourceRegistry:
    """Get the default global registry.

    The default registry is lazily initialized and includes built-in sources.

    Returns:
        The default SourceRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = SourceRegistry()
        _register_builtin_sources(_default_registry)
    return _default_registry


def _register_builtin_sources(registry: SourceRegistry) -> None:
    """Register built-in source types.

    Called when the default registry is first accessed.
    """
    # Import here to avoid circular imports
    try:
        from .filesystem import FileSystemSource
        registry.register("file", FileSystemSource)
    except ImportError:
        pass

    try:
        from .http import HTTPSource
        registry.register("http", HTTPSource)
        registry.register("https", HTTPSource)
    except ImportError:
        pass

    try:
        from .entity import EntitySource
        registry.register("entity", EntitySource)
    except ImportError:
        pass
