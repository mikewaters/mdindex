"""Entity URI document source with pluggable resolvers.

This module provides a flexible document source that delegates to
custom resolvers for fetching content from various backends (databases,
APIs, custom systems, etc.).

URI Format: entity://<resolver>/<resource-type>[/<id>][?query]

Examples:
- entity://postgres/articles
- entity://notion/pages/abc-123
- entity://custom/projects/42/docs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

from loguru import logger

from ..core.exceptions import PMDError
from .base import (
    BaseDocumentSource,
    DocumentReference,
    FetchResult,
    SourceCapabilities,
    SourceConfig,
    SourceFetchError,
    SourceListError,
)


# =============================================================================
# Exceptions
# =============================================================================


class EntityResolverError(PMDError):
    """Base exception for entity resolver operations."""

    pass


class ResolverNotFoundError(EntityResolverError):
    """No resolver registered for the given name."""

    def __init__(self, resolver_name: str):
        self.resolver_name = resolver_name
        super().__init__(f"No resolver registered with name '{resolver_name}'")


# =============================================================================
# Entity Types
# =============================================================================


@dataclass
class EntityInfo:
    """Information about an entity from a resolver.

    Attributes:
        id: Unique entity identifier.
        title: Optional entity title.
        path: Path for storage (defaults to id if not provided).
        metadata: Additional entity metadata.
    """

    id: str
    title: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_path(self) -> str:
        """Get the storage path for this entity."""
        return self.path or self.id


@dataclass
class EntityContent:
    """Content retrieved from an entity resolver.

    Attributes:
        content: The text content.
        title: Optional title extracted from content.
        content_type: MIME type of the content.
        metadata: Additional metadata (e.g., last_modified, version).
    """

    content: str
    title: str | None = None
    content_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Resolver Protocol
# =============================================================================


@runtime_checkable
class EntityResolver(Protocol):
    """Protocol for entity resolvers.

    Resolvers are responsible for listing and fetching entities from
    specific backends. Each resolver handles a particular "scheme" or
    system type.

    Example implementation:

        class PostgresResolver:
            def __init__(self, connection_string: str):
                self.conn = connect(connection_string)

            def list_entities(
                self, resource_type: str, query: dict
            ) -> Iterator[EntityInfo]:
                for row in self.conn.execute(f"SELECT * FROM {resource_type}"):
                    yield EntityInfo(id=row['id'], title=row.get('title'))

            async def fetch_entity(
                self, resource_type: str, entity_id: str
            ) -> EntityContent:
                row = self.conn.fetchone(
                    f"SELECT * FROM {resource_type} WHERE id = %s",
                    entity_id
                )
                return EntityContent(content=row['body'], title=row['title'])
    """

    def list_entities(
        self,
        resource_type: str,
        query: dict[str, Any],
    ) -> Iterator[EntityInfo]:
        """List available entities of a given type.

        Args:
            resource_type: Type of entities to list (e.g., 'articles', 'pages').
            query: Query parameters from the URI.

        Yields:
            EntityInfo for each available entity.
        """
        ...

    async def fetch_entity(
        self,
        resource_type: str,
        entity_id: str,
    ) -> EntityContent:
        """Fetch content for a specific entity.

        Args:
            resource_type: Type of entity.
            entity_id: Unique entity identifier.

        Returns:
            EntityContent with the entity's content.
        """
        ...


# =============================================================================
# Resolver Registry
# =============================================================================


class EntityResolverRegistry:
    """Registry for entity resolvers.

    Maintains a mapping from resolver names to resolver instances.

    Example:
        registry = EntityResolverRegistry()
        registry.register("postgres", PostgresResolver(conn_string))
        registry.register("notion", NotionResolver(api_key))

        resolver = registry.get("postgres")
        for entity in resolver.list_entities("articles", {}):
            print(entity.id)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._resolvers: dict[str, EntityResolver] = {}

    def register(
        self,
        name: str,
        resolver: EntityResolver,
        *,
        override: bool = False,
    ) -> None:
        """Register a resolver.

        Args:
            name: Resolver name (used in entity:// URIs).
            resolver: Resolver instance.
            override: If True, allow replacing existing registration.

        Raises:
            EntityResolverError: If name already registered and override=False.
        """
        name_lower = name.lower()
        if name_lower in self._resolvers and not override:
            raise EntityResolverError(
                f"Resolver '{name}' already registered. Use override=True to replace."
            )
        self._resolvers[name_lower] = resolver
        logger.debug(f"Registered entity resolver: {name}")

    def unregister(self, name: str) -> bool:
        """Remove a resolver.

        Args:
            name: Resolver name.

        Returns:
            True if removed, False if not found.
        """
        name_lower = name.lower()
        if name_lower in self._resolvers:
            del self._resolvers[name_lower]
            return True
        return False

    def get(self, name: str) -> EntityResolver | None:
        """Get a resolver by name.

        Args:
            name: Resolver name.

        Returns:
            Resolver instance or None if not found.
        """
        return self._resolvers.get(name.lower())

    def list_resolvers(self) -> list[str]:
        """List all registered resolver names.

        Returns:
            Sorted list of resolver names.
        """
        return sorted(self._resolvers.keys())


# =============================================================================
# Default Registry
# =============================================================================

_default_entity_registry: EntityResolverRegistry | None = None


def get_default_entity_registry() -> EntityResolverRegistry:
    """Get the default entity resolver registry.

    Returns:
        Default EntityResolverRegistry instance.
    """
    global _default_entity_registry
    if _default_entity_registry is None:
        _default_entity_registry = EntityResolverRegistry()
    return _default_entity_registry


# =============================================================================
# Entity URI Parsing
# =============================================================================


@dataclass
class ParsedEntityURI:
    """Parsed entity URI components.

    Attributes:
        resolver: Resolver name.
        resource_type: Resource type (e.g., 'articles', 'pages').
        entity_id: Optional specific entity ID.
        query: Query parameters as dict.
    """

    resolver: str
    resource_type: str
    entity_id: str | None = None
    query: dict[str, Any] = field(default_factory=dict)


def parse_entity_uri(uri: str) -> ParsedEntityURI:
    """Parse an entity URI into components.

    URI format: entity://<resolver>/<resource-type>[/<id>][?query]

    Args:
        uri: Entity URI to parse.

    Returns:
        ParsedEntityURI with components.

    Raises:
        ValueError: If URI format is invalid.
    """
    parsed = urlparse(uri)

    if parsed.scheme != "entity":
        raise ValueError(f"Invalid entity URI scheme: {parsed.scheme}")

    if not parsed.netloc:
        raise ValueError("Entity URI must have a resolver name")

    resolver = parsed.netloc

    # Parse path: /resource-type[/id]
    path_parts = [p for p in parsed.path.split("/") if p]
    if not path_parts:
        raise ValueError("Entity URI must have a resource type")

    resource_type = path_parts[0]
    entity_id = "/".join(path_parts[1:]) if len(path_parts) > 1 else None

    # Parse query string
    query = {}
    if parsed.query:
        qs = parse_qs(parsed.query)
        # Flatten single-value lists
        query = {k: v[0] if len(v) == 1 else v for k, v in qs.items()}

    return ParsedEntityURI(
        resolver=resolver,
        resource_type=resource_type,
        entity_id=entity_id,
        query=query,
    )


# =============================================================================
# Entity Source Implementation
# =============================================================================


class EntitySource(BaseDocumentSource):
    """Document source backed by entity resolvers.

    Parses entity:// URIs and delegates to registered resolvers for
    listing and fetching content.

    Example:
        # Register a resolver
        registry = get_default_entity_registry()
        registry.register("mydb", MyDatabaseResolver(conn))

        # Create source
        config = SourceConfig(uri="entity://mydb/articles")
        source = EntitySource(config)

        # Use it
        for ref in source.list_documents():
            result = await source.fetch_content(ref)
    """

    def __init__(
        self,
        config: SourceConfig,
        registry: EntityResolverRegistry | None = None,
    ) -> None:
        """Initialize entity source.

        Args:
            config: Source configuration with entity URI.
            registry: Resolver registry (uses default if not provided).
        """
        self._config = config
        self._registry = registry or get_default_entity_registry()
        self._parsed = parse_entity_uri(config.uri)
        self._resolver = self._registry.get(self._parsed.resolver)

    @property
    def resolver_name(self) -> str:
        """Get the resolver name."""
        return self._parsed.resolver

    @property
    def resource_type(self) -> str:
        """Get the resource type."""
        return self._parsed.resource_type

    def list_documents(self) -> Iterator[DocumentReference]:
        """Enumerate entities from the resolver.

        Yields:
            DocumentReference for each entity.

        Raises:
            SourceListError: If resolver not found or listing fails.
        """
        if self._resolver is None:
            raise SourceListError(
                self._config.uri,
                f"No resolver registered for '{self._parsed.resolver}'. "
                f"Available resolvers: {self._registry.list_resolvers()}",
            )

        # If specific entity ID in URI, yield just that
        if self._parsed.entity_id:
            yield DocumentReference(
                uri=self._config.uri,
                path=self._parsed.entity_id,
                title=None,
                metadata={
                    "resolver": self._parsed.resolver,
                    "resource_type": self._parsed.resource_type,
                },
            )
            return

        # List all entities of the resource type
        try:
            for entity in self._resolver.list_entities(
                self._parsed.resource_type,
                self._parsed.query,
            ):
                yield DocumentReference(
                    uri=f"entity://{self._parsed.resolver}/{self._parsed.resource_type}/{entity.id}",
                    path=entity.get_path(),
                    title=entity.title,
                    metadata={
                        "resolver": self._parsed.resolver,
                        "resource_type": self._parsed.resource_type,
                        **entity.metadata,
                    },
                )
        except Exception as e:
            raise SourceListError(self._config.uri, str(e))

    async def fetch_content(self, ref: DocumentReference) -> FetchResult:
        """Fetch entity content.

        Args:
            ref: Reference to the entity.

        Returns:
            FetchResult with entity content.

        Raises:
            SourceFetchError: If fetching fails.
        """
        if self._resolver is None:
            raise SourceFetchError(
                ref.uri,
                f"No resolver registered for '{self._parsed.resolver}'",
                retryable=False,
            )

        # Parse the entity ID from the reference
        parsed = parse_entity_uri(ref.uri)

        if not parsed.entity_id:
            raise SourceFetchError(
                ref.uri,
                "Entity URI must include entity ID for fetching",
                retryable=False,
            )

        try:
            entity_content = await self._resolver.fetch_entity(
                parsed.resource_type,
                parsed.entity_id,
            )

            return FetchResult(
                content=entity_content.content,
                content_type=entity_content.content_type,
                encoding="utf-8",
                metadata=entity_content.metadata,
            )

        except Exception as e:
            raise SourceFetchError(ref.uri, str(e), retryable=False)

    def capabilities(self) -> SourceCapabilities:
        """Return entity source capabilities."""
        return SourceCapabilities(
            supports_incremental=False,  # Depends on resolver
            supports_etag=False,
            supports_last_modified=False,
            supports_streaming=False,
            is_readonly=True,
        )
