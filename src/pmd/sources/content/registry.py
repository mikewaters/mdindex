"""Source registry for creating DocumentSource instances from collections.

This module provides a registry pattern that maps source_type strings (stored
in collections) to factory functions that construct DocumentSource instances.
This enables generic indexing code that works with any registered source type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from loguru import logger

from .base import DocumentSource, SourceConfig

if TYPE_CHECKING:
    from pmd.core.types import Collection


# Factory signature: takes a Collection, returns a DocumentSource
SourceFactory = Callable[["Collection"], DocumentSource]


class SourceRegistry:
    """Registry mapping source_type strings to DocumentSource factories.

    This registry enables the indexing service to construct the appropriate
    DocumentSource for any collection based on its stored source_type,
    without hardcoding knowledge of specific source implementations.

    Example:
        registry = get_default_registry()
        source = registry.create_source(collection)

        for ref in source.list_documents():
            result = await source.fetch_content(ref)
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._factories: dict[str, SourceFactory] = {}

    def register(self, source_type: str, factory: SourceFactory) -> None:
        """Register a factory for a source type.

        Args:
            source_type: The source type identifier (e.g., "filesystem", "http").
            factory: A callable that takes a Collection and returns a DocumentSource.
        """
        if source_type in self._factories:
            logger.warning(f"Overwriting source factory for {source_type!r}")
        self._factories[source_type] = factory
        logger.debug(f"Registered source factory: {source_type!r}")

    def unregister(self, source_type: str) -> bool:
        """Unregister a source type.

        Args:
            source_type: The source type to remove.

        Returns:
            True if the type was registered and removed, False if not found.
        """
        if source_type in self._factories:
            del self._factories[source_type]
            logger.debug(f"Unregistered source factory: {source_type!r}")
            return True
        return False

    def is_registered(self, source_type: str) -> bool:
        """Check if a source type is registered.

        Args:
            source_type: The source type to check.

        Returns:
            True if registered, False otherwise.
        """
        return source_type in self._factories

    def create_source(self, collection: "Collection") -> DocumentSource:
        """Create a DocumentSource for the given collection.

        Args:
            collection: Collection with source_type and source_config.

        Returns:
            Configured DocumentSource instance.

        Raises:
            ValueError: If source_type is not registered.
        """
        source_type = collection.source_type or "filesystem"
        factory = self._factories.get(source_type)
        if factory is None:
            available = ", ".join(sorted(self._factories.keys())) or "(none)"
            raise ValueError(
                f"Unknown source type {source_type!r}. Available: {available}"
            )
        return factory(collection)

    @property
    def registered_types(self) -> list[str]:
        """List registered source types."""
        return sorted(self._factories.keys())


# Singleton instance
_default_registry: SourceRegistry | None = None


def get_default_registry() -> SourceRegistry:
    """Get the default source registry with built-in sources registered.

    Returns:
        The singleton SourceRegistry instance with filesystem and other
        built-in sources pre-registered.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = SourceRegistry()
        _register_builtin_sources(_default_registry)
    return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry singleton.

    This is primarily useful for testing to ensure a clean state.
    """
    global _default_registry
    _default_registry = None


def _register_builtin_sources(registry: SourceRegistry) -> None:
    """Register built-in source types.

    Args:
        registry: The registry to populate with built-in sources.
    """
    from .filesystem import FileSystemSource

    def filesystem_factory(collection: "Collection") -> DocumentSource:
        """Create a FileSystemSource from collection configuration."""
        config = SourceConfig(
            uri=collection.get_source_uri(),
            extra=collection.get_source_config_dict(),
        )
        return FileSystemSource(config)

    registry.register("filesystem", filesystem_factory)
    # Future source types will be registered here:
    # registry.register("http", http_factory)
    # registry.register("s3", s3_factory)
