"""Tests for base source types and protocol."""

import pytest
from dataclasses import FrozenInstanceError

from pmd.sources import (
    DocumentReference,
    FetchResult,
    SourceCapabilities,
    SourceConfig,
    SourceError,
    SourceFetchError,
    SourceListError,
)


class TestSourceConfig:
    """Tests for SourceConfig."""

    def test_create_with_uri_only(self):
        """SourceConfig can be created with just URI."""
        config = SourceConfig(uri="file:///path/to/docs")
        assert config.uri == "file:///path/to/docs"
        assert config.extra == {}

    def test_create_with_extra(self):
        """SourceConfig can include extra configuration."""
        config = SourceConfig(
            uri="https://example.com",
            extra={"timeout": 30, "auth": "bearer"},
        )
        assert config.uri == "https://example.com"
        assert config.extra["timeout"] == 30

    def test_get_returns_extra_value(self):
        """get() retrieves values from extra dict."""
        config = SourceConfig(uri="test://", extra={"key": "value"})
        assert config.get("key") == "value"

    def test_get_returns_default_for_missing(self):
        """get() returns default for missing keys."""
        config = SourceConfig(uri="test://")
        assert config.get("missing") is None
        assert config.get("missing", "default") == "default"

    def test_is_frozen(self):
        """SourceConfig is immutable."""
        config = SourceConfig(uri="test://")
        with pytest.raises(FrozenInstanceError):
            config.uri = "other://"


class TestDocumentReference:
    """Tests for DocumentReference."""

    def test_create_minimal(self):
        """DocumentReference can be created with required fields only."""
        ref = DocumentReference(uri="file:///doc.md", path="doc.md")
        assert ref.uri == "file:///doc.md"
        assert ref.path == "doc.md"
        assert ref.title is None
        assert ref.metadata == {}

    def test_create_full(self):
        """DocumentReference can include all optional fields."""
        ref = DocumentReference(
            uri="https://example.com/doc",
            path="doc.html",
            title="My Document",
            metadata={"etag": "abc123"},
        )
        assert ref.title == "My Document"
        assert ref.metadata["etag"] == "abc123"

    def test_get_metadata(self):
        """get_metadata retrieves metadata values."""
        ref = DocumentReference(
            uri="test://",
            path="test",
            metadata={"key": "value"},
        )
        assert ref.get_metadata("key") == "value"
        assert ref.get_metadata("missing") is None
        assert ref.get_metadata("missing", "default") == "default"

    def test_is_frozen(self):
        """DocumentReference is immutable."""
        ref = DocumentReference(uri="test://", path="test")
        with pytest.raises(FrozenInstanceError):
            ref.path = "other"


class TestFetchResult:
    """Tests for FetchResult."""

    def test_create_minimal(self):
        """FetchResult can be created with content only."""
        result = FetchResult(content="Hello, world!")
        assert result.content == "Hello, world!"
        assert result.content_type == "text/plain"
        assert result.encoding == "utf-8"
        assert result.metadata == {}

    def test_create_full(self):
        """FetchResult can include all fields."""
        result = FetchResult(
            content="<html>",
            content_type="text/html",
            encoding="iso-8859-1",
            metadata={"http_status": 200},
        )
        assert result.content_type == "text/html"
        assert result.encoding == "iso-8859-1"
        assert result.metadata["http_status"] == 200


class TestSourceCapabilities:
    """Tests for SourceCapabilities."""

    def test_defaults(self):
        """SourceCapabilities has sensible defaults."""
        caps = SourceCapabilities()
        assert caps.supports_incremental is False
        assert caps.supports_etag is False
        assert caps.supports_last_modified is False
        assert caps.supports_streaming is False
        assert caps.is_readonly is True

    def test_custom_values(self):
        """SourceCapabilities accepts custom values."""
        caps = SourceCapabilities(
            supports_incremental=True,
            supports_etag=True,
        )
        assert caps.supports_incremental is True
        assert caps.supports_etag is True


class TestSourceExceptions:
    """Tests for source exceptions."""

    def test_source_error_is_pmd_error(self):
        """SourceError inherits from PMDError."""
        from pmd.core.exceptions import PMDError
        assert issubclass(SourceError, PMDError)

    def test_source_list_error(self):
        """SourceListError contains source info."""
        error = SourceListError("file:///path", "Directory not found")
        assert error.source_uri == "file:///path"
        assert error.reason == "Directory not found"
        assert "file:///path" in str(error)

    def test_source_fetch_error(self):
        """SourceFetchError contains fetch info."""
        error = SourceFetchError("https://example.com/doc", "404 Not Found", retryable=False)
        assert error.uri == "https://example.com/doc"
        assert error.reason == "404 Not Found"
        assert error.retryable is False

    def test_source_fetch_error_retryable(self):
        """SourceFetchError can be marked retryable."""
        error = SourceFetchError("https://example.com", "Timeout", retryable=True)
        assert error.retryable is True
