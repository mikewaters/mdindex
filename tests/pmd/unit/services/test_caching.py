"""Tests for DocumentCacher service."""

import pytest
from pathlib import Path

from pmd.core.config import CacheConfig
from pmd.services.caching import DocumentCacher


class TestDocumentCacher:
    """Tests for DocumentCacher service."""

    def test_cache_document_creates_file(self, tmp_path: Path):
        """Caching a document creates the file on disk."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        uri = cacher.cache_document("my-collection", "doc.md", "# Hello\n\nWorld")

        cached_path = tmp_path / "my-collection" / "doc.md"
        assert cached_path.exists()
        assert cached_path.read_text() == "# Hello\n\nWorld"
        assert uri == cached_path.as_uri()

    def test_cache_document_creates_nested_directories(self, tmp_path: Path):
        """Caching creates nested directories as needed."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        uri = cacher.cache_document("my-collection", "nested/path/doc.md", "Content")

        cached_path = tmp_path / "my-collection" / "nested" / "path" / "doc.md"
        assert cached_path.exists()
        assert cached_path.read_text() == "Content"

    def test_cache_document_overwrites_existing(self, tmp_path: Path):
        """Caching overwrites existing cached files."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        # Cache initial version
        cacher.cache_document("my-collection", "doc.md", "Version 1")

        # Cache updated version
        cacher.cache_document("my-collection", "doc.md", "Version 2")

        cached_path = tmp_path / "my-collection" / "doc.md"
        assert cached_path.read_text() == "Version 2"

    def test_remove_document_deletes_file(self, tmp_path: Path):
        """Removing a document deletes the cached file."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        # Cache a document
        cacher.cache_document("my-collection", "doc.md", "Content")
        cached_path = tmp_path / "my-collection" / "doc.md"
        assert cached_path.exists()

        # Remove it
        result = cacher.remove_document("my-collection", "doc.md")

        assert result is True
        assert not cached_path.exists()

    def test_remove_document_returns_false_if_not_exists(self, tmp_path: Path):
        """Removing a non-existent document returns False."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        result = cacher.remove_document("my-collection", "nonexistent.md")

        assert result is False

    def test_remove_document_cleans_empty_directories(self, tmp_path: Path):
        """Removing a document cleans up empty parent directories up to collection root."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        # Cache a document in nested directory
        cacher.cache_document("my-collection", "nested/path/doc.md", "Content")
        nested_path = tmp_path / "my-collection" / "nested" / "path"
        assert nested_path.exists()

        # Remove it
        cacher.remove_document("my-collection", "nested/path/doc.md")

        # Nested directories should be cleaned up
        assert not nested_path.exists()
        assert not (tmp_path / "my-collection" / "nested").exists()
        # Collection root is kept (empty directory cleanup stops there)
        assert (tmp_path / "my-collection").exists()

    def test_remove_collection_deletes_all_files(self, tmp_path: Path):
        """Removing a collection deletes all cached files."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        # Cache multiple documents
        cacher.cache_document("my-collection", "doc1.md", "Content 1")
        cacher.cache_document("my-collection", "doc2.md", "Content 2")
        cacher.cache_document("my-collection", "nested/doc3.md", "Content 3")

        # Remove collection
        count = cacher.remove_collection("my-collection")

        assert count == 3
        assert not (tmp_path / "my-collection").exists()

    def test_remove_collection_returns_zero_if_not_exists(self, tmp_path: Path):
        """Removing a non-existent collection returns 0."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        count = cacher.remove_collection("nonexistent")

        assert count == 0

    def test_get_cached_path_returns_path_if_exists(self, tmp_path: Path):
        """get_cached_path returns path when file exists."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        cacher.cache_document("my-collection", "doc.md", "Content")

        result = cacher.get_cached_path("my-collection", "doc.md")

        assert result == tmp_path / "my-collection" / "doc.md"

    def test_get_cached_path_returns_none_if_not_exists(self, tmp_path: Path):
        """get_cached_path returns None when file doesn't exist."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        result = cacher.get_cached_path("my-collection", "nonexistent.md")

        assert result is None

    def test_enabled_property(self, tmp_path: Path):
        """enabled property reflects config."""
        enabled_config = CacheConfig(enabled=True, base_path=tmp_path)
        disabled_config = CacheConfig(enabled=False, base_path=tmp_path)

        assert DocumentCacher(enabled_config).enabled is True
        assert DocumentCacher(disabled_config).enabled is False

    def test_base_path_property(self, tmp_path: Path):
        """base_path property returns resolved path."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        assert cacher.base_path == tmp_path.resolve()

    def test_cache_document_handles_unicode(self, tmp_path: Path):
        """Caching handles unicode content correctly."""
        config = CacheConfig(enabled=True, base_path=tmp_path)
        cacher = DocumentCacher(config)

        # Use actual unicode characters (not surrogate pairs)
        content = "# Title\n\nUnicode: \u4e2d\u6587 \u65e5\u672c\u8a9e \U0001F600"
        cacher.cache_document("my-collection", "unicode.md", content)

        cached_path = tmp_path / "my-collection" / "unicode.md"
        assert cached_path.read_text(encoding="utf-8") == content


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Default config has sensible defaults."""
        config = CacheConfig()

        assert config.enabled is False
        assert "pmd" in str(config.base_path)
        assert "files" in str(config.base_path)

    def test_custom_base_path(self, tmp_path: Path):
        """Custom base_path is respected."""
        config = CacheConfig(enabled=True, base_path=tmp_path / "custom")

        assert config.base_path == tmp_path / "custom"

    def test_enabled_flag(self):
        """Enabled flag can be set."""
        config = CacheConfig(enabled=True)

        assert config.enabled is True
