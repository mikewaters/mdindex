"""Tests for SourceRegistry."""

import pytest
from typing import Any, Iterator

from pmd.sources import (
    DocumentReference,
    DocumentSource,
    FetchResult,
    RegistryError,
    SourceCapabilities,
    SourceConfig,
    SourceRegistry,
    UnknownSchemeError,
    get_default_registry,
)


class MockSource:
    """Mock document source for testing."""

    def __init__(self, config: SourceConfig):
        self.config = config

    def list_documents(self) -> Iterator[DocumentReference]:
        yield DocumentReference(uri="mock://doc", path="doc")

    async def fetch_content(self, ref: DocumentReference) -> FetchResult:
        return FetchResult(content="mock content")

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities()

    async def check_modified(self, ref: DocumentReference, stored: dict) -> bool:
        return True


class TestSourceRegistry:
    """Tests for SourceRegistry."""

    def test_register_and_resolve(self):
        """Can register a source and resolve URIs."""
        registry = SourceRegistry()
        registry.register("mock", MockSource)

        source = registry.resolve("mock://test/path")
        assert isinstance(source, MockSource)
        assert source.config.uri == "mock://test/path"

    def test_register_case_insensitive(self):
        """Scheme registration is case-insensitive."""
        registry = SourceRegistry()
        registry.register("MOCK", MockSource)

        source = registry.resolve("mock://test")
        assert isinstance(source, MockSource)

    def test_register_duplicate_raises(self):
        """Registering duplicate scheme raises error."""
        registry = SourceRegistry()
        registry.register("mock", MockSource)

        with pytest.raises(RegistryError, match="already registered"):
            registry.register("mock", MockSource)

    def test_register_override(self):
        """Can override existing registration."""
        registry = SourceRegistry()
        registry.register("mock", MockSource)
        registry.register("mock", MockSource, override=True)
        # Should not raise

    def test_unregister(self):
        """Can unregister a scheme."""
        registry = SourceRegistry()
        registry.register("mock", MockSource)

        assert registry.unregister("mock") is True
        assert registry.unregister("mock") is False  # Already gone

    def test_resolve_unknown_scheme_raises(self):
        """Resolving unknown scheme raises error."""
        registry = SourceRegistry()

        with pytest.raises(UnknownSchemeError) as exc_info:
            registry.resolve("unknown://path")

        assert exc_info.value.scheme == "unknown"

    def test_is_registered(self):
        """is_registered checks for scheme presence."""
        registry = SourceRegistry()
        registry.register("mock", MockSource)

        assert registry.is_registered("mock") is True
        assert registry.is_registered("MOCK") is True  # Case insensitive
        assert registry.is_registered("other") is False

    def test_supported_schemes(self):
        """supported_schemes returns sorted list."""
        registry = SourceRegistry()
        registry.register("beta", MockSource)
        registry.register("alpha", MockSource)

        schemes = registry.supported_schemes()
        assert schemes == ["alpha", "beta"]

    def test_resolve_with_config(self):
        """Can pass custom config to resolve."""
        registry = SourceRegistry()
        registry.register("mock", MockSource)

        config = SourceConfig(uri="mock://test", extra={"custom": "value"})
        source = registry.resolve("mock://test", config)

        assert source.config.extra["custom"] == "value"

    def test_resolve_bare_path_defaults_to_file(self):
        """Bare paths are treated as file:// URIs."""
        registry = SourceRegistry()
        registry.register("file", MockSource)

        source = registry.resolve("/path/to/docs")
        assert isinstance(source, MockSource)



class TestDefaultRegistry:
    """Tests for default registry."""

    def test_default_registry_exists(self):
        """Default registry is available."""
        registry = get_default_registry()
        assert isinstance(registry, SourceRegistry)

    def test_default_registry_singleton(self):
        """Default registry is a singleton."""
        r1 = get_default_registry()
        r2 = get_default_registry()
        assert r1 is r2

    def test_default_registry_has_builtin_sources(self):
        """Default registry has file, http, https, entity schemes."""
        registry = get_default_registry()
        schemes = registry.supported_schemes()

        assert "file" in schemes
        assert "http" in schemes
        assert "https" in schemes
        assert "entity" in schemes
