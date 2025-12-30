"""Tests for EntitySource and entity resolver system."""

import pytest
from typing import Any, Iterator

from pmd.sources import (
    DocumentReference,
    SourceConfig,
    SourceListError,
    SourceFetchError,
)
from pmd.sources.entity import (
    EntityContent,
    EntityInfo,
    EntityResolver,
    EntityResolverError,
    EntityResolverRegistry,
    EntitySource,
    ParsedEntityURI,
    ResolverNotFoundError,
    get_default_entity_registry,
    parse_entity_uri,
)


class MockResolver:
    """Mock entity resolver for testing."""

    def __init__(self, entities: list[EntityInfo] | None = None):
        self.entities = entities or []
        self.fetch_calls: list[tuple[str, str]] = []

    def list_entities(
        self, resource_type: str, query: dict[str, Any]
    ) -> Iterator[EntityInfo]:
        for entity in self.entities:
            yield entity

    async def fetch_entity(
        self, resource_type: str, entity_id: str
    ) -> EntityContent:
        self.fetch_calls.append((resource_type, entity_id))
        return EntityContent(
            content=f"Content for {entity_id}",
            title=f"Title for {entity_id}",
            content_type="text/plain",
            metadata={"fetched": True},
        )


class TestEntityInfo:
    """Tests for EntityInfo."""

    def test_create_minimal(self):
        """EntityInfo requires only id."""
        info = EntityInfo(id="123")
        assert info.id == "123"
        assert info.title is None
        assert info.path is None
        assert info.metadata == {}

    def test_create_full(self):
        """EntityInfo accepts all fields."""
        info = EntityInfo(
            id="abc",
            title="My Entity",
            path="custom/path",
            metadata={"key": "value"},
        )
        assert info.id == "abc"
        assert info.title == "My Entity"
        assert info.path == "custom/path"
        assert info.metadata["key"] == "value"

    def test_get_path_returns_path_if_set(self):
        """get_path returns path when set."""
        info = EntityInfo(id="123", path="custom/path")
        assert info.get_path() == "custom/path"

    def test_get_path_defaults_to_id(self):
        """get_path returns id when path not set."""
        info = EntityInfo(id="entity-456")
        assert info.get_path() == "entity-456"


class TestEntityContent:
    """Tests for EntityContent."""

    def test_create_minimal(self):
        """EntityContent requires only content."""
        content = EntityContent(content="Hello")
        assert content.content == "Hello"
        assert content.title is None
        assert content.content_type == "text/plain"
        assert content.metadata == {}

    def test_create_full(self):
        """EntityContent accepts all fields."""
        content = EntityContent(
            content="# Markdown",
            title="My Doc",
            content_type="text/markdown",
            metadata={"version": 2},
        )
        assert content.content == "# Markdown"
        assert content.title == "My Doc"
        assert content.content_type == "text/markdown"
        assert content.metadata["version"] == 2


class TestParseEntityUri:
    """Tests for parse_entity_uri function."""

    def test_parse_basic_uri(self):
        """Parses resolver and resource type."""
        result = parse_entity_uri("entity://mydb/articles")
        assert result.resolver == "mydb"
        assert result.resource_type == "articles"
        assert result.entity_id is None
        assert result.query == {}

    def test_parse_uri_with_entity_id(self):
        """Parses entity ID from path."""
        result = parse_entity_uri("entity://postgres/users/123")
        assert result.resolver == "postgres"
        assert result.resource_type == "users"
        assert result.entity_id == "123"

    def test_parse_uri_with_nested_entity_id(self):
        """Parses nested entity ID paths."""
        result = parse_entity_uri("entity://notion/pages/abc/def/ghi")
        assert result.resolver == "notion"
        assert result.resource_type == "pages"
        assert result.entity_id == "abc/def/ghi"

    def test_parse_uri_with_query(self):
        """Parses query parameters."""
        result = parse_entity_uri("entity://api/records?status=active&limit=10")
        assert result.resolver == "api"
        assert result.resource_type == "records"
        assert result.query == {"status": "active", "limit": "10"}

    def test_parse_uri_with_multi_value_query(self):
        """Parses multi-value query parameters."""
        result = parse_entity_uri("entity://api/items?tag=a&tag=b")
        assert result.query == {"tag": ["a", "b"]}

    def test_parse_invalid_scheme_raises(self):
        """Raises for non-entity scheme."""
        with pytest.raises(ValueError, match="Invalid entity URI scheme"):
            parse_entity_uri("http://example.com")

    def test_parse_missing_resolver_raises(self):
        """Raises for missing resolver name."""
        with pytest.raises(ValueError, match="must have a resolver"):
            parse_entity_uri("entity:///articles")

    def test_parse_missing_resource_type_raises(self):
        """Raises for missing resource type."""
        with pytest.raises(ValueError, match="must have a resource type"):
            parse_entity_uri("entity://mydb/")


class TestEntityResolverRegistry:
    """Tests for EntityResolverRegistry."""

    def test_register_and_get(self):
        """Can register and retrieve resolver."""
        registry = EntityResolverRegistry()
        resolver = MockResolver()

        registry.register("test", resolver)
        retrieved = registry.get("test")

        assert retrieved is resolver

    def test_register_case_insensitive(self):
        """Registration is case-insensitive."""
        registry = EntityResolverRegistry()
        resolver = MockResolver()

        registry.register("TEST", resolver)
        assert registry.get("test") is resolver
        assert registry.get("TEST") is resolver

    def test_register_duplicate_raises(self):
        """Registering duplicate raises error."""
        registry = EntityResolverRegistry()
        resolver = MockResolver()

        registry.register("test", resolver)
        with pytest.raises(EntityResolverError, match="already registered"):
            registry.register("test", MockResolver())

    def test_register_override(self):
        """Can override existing registration."""
        registry = EntityResolverRegistry()
        resolver1 = MockResolver()
        resolver2 = MockResolver()

        registry.register("test", resolver1)
        registry.register("test", resolver2, override=True)

        assert registry.get("test") is resolver2

    def test_unregister(self):
        """Can unregister resolver."""
        registry = EntityResolverRegistry()
        resolver = MockResolver()

        registry.register("test", resolver)
        assert registry.unregister("test") is True
        assert registry.get("test") is None
        assert registry.unregister("test") is False

    def test_list_resolvers(self):
        """list_resolvers returns sorted names."""
        registry = EntityResolverRegistry()
        registry.register("beta", MockResolver())
        registry.register("alpha", MockResolver())

        assert registry.list_resolvers() == ["alpha", "beta"]

    def test_get_unknown_returns_none(self):
        """get returns None for unknown resolver."""
        registry = EntityResolverRegistry()
        assert registry.get("unknown") is None


class TestDefaultEntityRegistry:
    """Tests for default entity registry."""

    def test_default_registry_exists(self):
        """Default registry is available."""
        registry = get_default_entity_registry()
        assert isinstance(registry, EntityResolverRegistry)

    def test_default_registry_singleton(self):
        """Default registry is a singleton."""
        r1 = get_default_entity_registry()
        r2 = get_default_entity_registry()
        assert r1 is r2


class TestEntitySource:
    """Tests for EntitySource."""

    def test_create_source(self):
        """Can create EntitySource from config."""
        registry = EntityResolverRegistry()
        registry.register("test", MockResolver())

        config = SourceConfig(uri="entity://test/articles")
        source = EntitySource(config, registry)

        assert source.resolver_name == "test"
        assert source.resource_type == "articles"

    def test_list_documents_with_entities(self):
        """list_documents yields references for entities."""
        entities = [
            EntityInfo(id="1", title="First"),
            EntityInfo(id="2", title="Second", path="custom/2"),
        ]
        resolver = MockResolver(entities)
        registry = EntityResolverRegistry()
        registry.register("test", resolver)

        config = SourceConfig(uri="entity://test/articles")
        source = EntitySource(config, registry)

        docs = list(source.list_documents())
        assert len(docs) == 2

        assert docs[0].uri == "entity://test/articles/1"
        assert docs[0].path == "1"
        assert docs[0].title == "First"

        assert docs[1].uri == "entity://test/articles/2"
        assert docs[1].path == "custom/2"
        assert docs[1].title == "Second"

    def test_list_documents_specific_entity(self):
        """list_documents yields single ref for specific entity URI."""
        registry = EntityResolverRegistry()
        registry.register("test", MockResolver())

        config = SourceConfig(uri="entity://test/articles/42")
        source = EntitySource(config, registry)

        docs = list(source.list_documents())
        assert len(docs) == 1
        assert docs[0].uri == "entity://test/articles/42"
        assert docs[0].path == "42"

    def test_list_documents_no_resolver_raises(self):
        """list_documents raises when resolver not found."""
        registry = EntityResolverRegistry()

        config = SourceConfig(uri="entity://unknown/articles")
        source = EntitySource(config, registry)

        with pytest.raises(SourceListError, match="No resolver registered"):
            list(source.list_documents())

    @pytest.mark.asyncio
    async def test_fetch_content(self):
        """fetch_content retrieves entity content."""
        resolver = MockResolver()
        registry = EntityResolverRegistry()
        registry.register("test", resolver)

        config = SourceConfig(uri="entity://test/articles")
        source = EntitySource(config, registry)

        ref = DocumentReference(
            uri="entity://test/articles/123",
            path="123",
        )
        result = await source.fetch_content(ref)

        assert "Content for 123" in result.content
        assert result.content_type == "text/plain"
        assert result.metadata["fetched"] is True
        assert resolver.fetch_calls == [("articles", "123")]

    @pytest.mark.asyncio
    async def test_fetch_content_no_resolver_raises(self):
        """fetch_content raises when resolver not found."""
        registry = EntityResolverRegistry()

        config = SourceConfig(uri="entity://unknown/articles")
        source = EntitySource(config, registry)

        ref = DocumentReference(
            uri="entity://unknown/articles/1",
            path="1",
        )

        with pytest.raises(SourceFetchError, match="No resolver registered"):
            await source.fetch_content(ref)

    @pytest.mark.asyncio
    async def test_fetch_content_no_entity_id_raises(self):
        """fetch_content raises when entity ID missing."""
        resolver = MockResolver()
        registry = EntityResolverRegistry()
        registry.register("test", resolver)

        config = SourceConfig(uri="entity://test/articles")
        source = EntitySource(config, registry)

        ref = DocumentReference(
            uri="entity://test/articles",
            path="articles",
        )

        with pytest.raises(SourceFetchError, match="must include entity ID"):
            await source.fetch_content(ref)

    def test_capabilities(self):
        """EntitySource reports correct capabilities."""
        registry = EntityResolverRegistry()
        registry.register("test", MockResolver())

        config = SourceConfig(uri="entity://test/items")
        source = EntitySource(config, registry)

        caps = source.capabilities()
        assert caps.supports_incremental is False
        assert caps.supports_etag is False
        assert caps.is_readonly is True


class TestResolverProtocol:
    """Tests for EntityResolver protocol compliance."""

    def test_mock_resolver_is_protocol_compliant(self):
        """MockResolver satisfies EntityResolver protocol."""
        resolver = MockResolver()
        assert isinstance(resolver, EntityResolver)
