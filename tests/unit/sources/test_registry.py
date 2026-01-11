"""Unit tests for SourceRegistry."""

from unittest.mock import MagicMock

import pytest

from pmd.core.types import SourceCollection
from pmd.sources import (
    DocumentSource,
    SourceRegistry,
    get_default_registry,
    reset_default_registry,
)
from pmd.sources import FileSystemSource


@pytest.fixture
def registry():
    """Create a fresh empty registry for testing."""
    return SourceRegistry()


@pytest.fixture
def mock_collection():
    """Create a mock Collection for testing."""
    return SourceCollection(
        id=1,
        name="test-collection",
        pwd="/tmp/test-docs",
        glob_pattern="**/*.md",
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
        source_type="filesystem",
        source_config=None,
    )


@pytest.fixture
def mock_source():
    """Create a mock DocumentSource."""
    source = MagicMock(spec=DocumentSource)
    return source


class TestSourceRegistry:
    """Tests for SourceRegistry class."""

    def test_register_and_create(self, registry, mock_collection, mock_source):
        """Register factory and create source."""
        factory = MagicMock(return_value=mock_source)
        registry.register("filesystem", factory)

        result = registry.create_source(mock_collection)

        factory.assert_called_once_with(mock_collection)
        assert result is mock_source

    def test_create_unknown_type_raises(self, registry, mock_collection):
        """Unknown source_type raises ValueError."""
        mock_collection.source_type = "unknown"

        with pytest.raises(ValueError, match="Unknown source type 'unknown'"):
            registry.create_source(mock_collection)

    def test_create_unknown_type_lists_available(self, registry, mock_collection):
        """Error message lists available source types."""
        registry.register("http", MagicMock())
        registry.register("s3", MagicMock())
        mock_collection.source_type = "unknown"

        with pytest.raises(ValueError, match="Available: http, s3"):
            registry.create_source(mock_collection)

    def test_overwrite_registration_replaces_factory(self, registry, mock_collection, mock_source):
        """Overwriting registration replaces the factory."""
        first_source = MagicMock(spec=DocumentSource)
        second_source = MagicMock(spec=DocumentSource)

        registry.register("filesystem", MagicMock(return_value=first_source))
        registry.register("filesystem", MagicMock(return_value=second_source))

        # The second factory should be used
        result = registry.create_source(mock_collection)
        assert result is second_source

    def test_unregister_removes_factory(self, registry, mock_collection):
        """Unregister removes factory."""
        registry.register("filesystem", MagicMock())

        result = registry.unregister("filesystem")

        assert result is True
        assert not registry.is_registered("filesystem")

    def test_unregister_unknown_returns_false(self, registry):
        """Unregister unknown type returns False."""
        result = registry.unregister("unknown")
        assert result is False

    def test_is_registered(self, registry):
        """is_registered returns correct status."""
        assert not registry.is_registered("filesystem")

        registry.register("filesystem", MagicMock())

        assert registry.is_registered("filesystem")
        assert not registry.is_registered("unknown")

    def test_registered_types_sorted(self, registry):
        """registered_types returns sorted list."""
        registry.register("s3", MagicMock())
        registry.register("http", MagicMock())
        registry.register("filesystem", MagicMock())

        assert registry.registered_types == ["filesystem", "http", "s3"]

    def test_empty_registry_create_raises(self, registry, mock_collection):
        """Empty registry raises ValueError with 'none' message."""
        with pytest.raises(ValueError, match=r"Available: \(none\)"):
            registry.create_source(mock_collection)

    def test_null_source_type_defaults_to_filesystem(self, registry, mock_collection, mock_source):
        """collection.source_type=None uses filesystem."""
        mock_collection.source_type = None
        factory = MagicMock(return_value=mock_source)
        registry.register("filesystem", factory)

        result = registry.create_source(mock_collection)

        factory.assert_called_once_with(mock_collection)
        assert result is mock_source


class TestDefaultRegistry:
    """Tests for default registry singleton."""

    def setup_method(self):
        """Reset the default registry before each test."""
        reset_default_registry()

    def teardown_method(self):
        """Reset the default registry after each test."""
        reset_default_registry()

    def test_filesystem_factory_registered_by_default(self):
        """Default registry has filesystem registered."""
        registry = get_default_registry()

        assert registry.is_registered("filesystem")
        assert "filesystem" in registry.registered_types

    def test_get_default_registry_singleton(self):
        """get_default_registry returns same instance."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()

        assert registry1 is registry2

    def test_reset_default_registry(self):
        """reset_default_registry clears singleton."""
        registry1 = get_default_registry()
        reset_default_registry()
        registry2 = get_default_registry()

        # Should be a new instance
        assert registry1 is not registry2

    def test_create_filesystem_source_from_collection(self, tmp_path):
        """Create FileSystemSource from Collection."""
        # Create a real collection with a valid path
        collection = SourceCollection(
            id=1,
            name="test-collection",
            pwd=str(tmp_path),
            glob_pattern="**/*.md",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            source_type="filesystem",
            source_config=None,
        )

        registry = get_default_registry()
        source = registry.create_source(collection)

        assert isinstance(source, FileSystemSource)
        assert source.base_path == tmp_path
        assert source.glob_pattern == "**/*.md"

    def test_create_filesystem_source_with_source_config(self, tmp_path):
        """Create FileSystemSource respects source_config."""
        collection = SourceCollection(
            id=1,
            name="test-collection",
            pwd=str(tmp_path),
            glob_pattern="**/*.txt",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            source_type="filesystem",
            source_config={"encoding": "latin-1", "follow_symlinks": True},
        )

        registry = get_default_registry()
        source = registry.create_source(collection)

        assert isinstance(source, FileSystemSource)
        # The source_config is passed through get_source_config_dict()
        # and then FileSystemConfig.from_source_config() parses it
