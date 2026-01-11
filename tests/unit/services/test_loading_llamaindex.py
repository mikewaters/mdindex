"""Tests for LlamaIndexLoaderAdapter."""

import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pmd.core.config import Config
from pmd.core.exceptions import CollectionNotFoundError
from pmd.services import ServiceContainer
from pmd.services.loading import LoadingService
from pmd.services.loading_llamaindex import LlamaIndexLoaderAdapter
from pmd.store.source_metadata import SourceMetadataRepository


@dataclass
class MockLlamaIndexDocument:
    """Mock LlamaIndex Document for testing."""

    text: str
    metadata: dict[str, Any]
    id_: str | None = None


class MockLoader:
    """Mock LlamaIndex loader for testing."""

    def __init__(self, documents: list[MockLlamaIndexDocument]):
        self._documents = documents

    def load_data(self, **kwargs) -> list[MockLlamaIndexDocument]:
        return self._documents


class TestURIConstruction:
    """Tests for URI construction fallback chain."""

    @pytest.mark.asyncio
    async def test_uri_from_metadata_path(self):
        """Uses metadata[uri_key] when present."""
        doc = MockLlamaIndexDocument(
            text="Content",
            metadata={"path": "docs/test.md", "title": "Test"},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        result = await adapter.load_eager()

        assert len(result) == 1
        assert result[0].ref.uri == "llamaindex://docs/test.md"
        assert result[0].ref.path == "docs/test.md"

    @pytest.mark.asyncio
    async def test_uri_from_id(self):
        """Falls back to id_ when no path."""
        doc = MockLlamaIndexDocument(
            text="Content",
            metadata={},
            id_="unique-doc-id",
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        result = await adapter.load_eager()

        assert len(result) == 1
        assert result[0].ref.uri == "llamaindex://unique-doc-id"
        assert result[0].ref.path == "unique-doc-id"

    @pytest.mark.asyncio
    async def test_uri_from_hash(self):
        """Falls back to content hash when no id."""
        doc = MockLlamaIndexDocument(
            text="Unique content here",
            metadata={"author": "test"},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        result = await adapter.load_eager()

        assert len(result) == 1
        assert result[0].ref.uri.startswith("llamaindex://hash-")
        assert result[0].ref.path.startswith("hash-")

    @pytest.mark.asyncio
    async def test_hash_includes_namespace(self):
        """Hash fallback includes namespace for uniqueness."""
        doc = MockLlamaIndexDocument(
            text="Same content",
            metadata={},
        )

        adapter1 = LlamaIndexLoaderAdapter(
            loader=MockLoader([doc]),
            content_type="text/markdown",
            namespace="source1",
        )
        adapter2 = LlamaIndexLoaderAdapter(
            loader=MockLoader([doc]),
            content_type="text/markdown",
            namespace="source2",
        )

        result1 = await adapter1.load_eager()
        result2 = await adapter2.load_eager()

        # Different namespaces should produce different hashes
        assert result1[0].ref.uri != result2[0].ref.uri

    @pytest.mark.asyncio
    async def test_custom_uri_key(self):
        """Uses custom uri_key when specified."""
        doc = MockLlamaIndexDocument(
            text="Content",
            metadata={"source_url": "https://example.com/doc"},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/html",
            uri_key="source_url",
        )

        result = await adapter.load_eager()

        assert result[0].ref.path == "https://example.com/doc"


class TestMetadataAugmentation:
    """Tests for metadata extraction and augmentation."""

    @pytest.mark.asyncio
    async def test_metadata_augmentation(self):
        """Extracted metadata preserved, LlamaIndex in _llamaindex."""
        doc = MockLlamaIndexDocument(
            text="# Document Title\n\nContent with #python tag",
            metadata={"path": "test.md", "source": "web", "page": 1},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        result = await adapter.load_eager()

        assert len(result) == 1
        extracted = result[0].extracted_metadata
        assert extracted is not None
        # LlamaIndex metadata should be under _llamaindex key
        assert "_llamaindex" in extracted.attributes
        assert extracted.attributes["_llamaindex"]["source"] == "web"
        assert extracted.attributes["_llamaindex"]["page"] == 1

    @pytest.mark.asyncio
    async def test_empty_llamaindex_metadata(self):
        """Handles empty LlamaIndex metadata gracefully."""
        doc = MockLlamaIndexDocument(
            text="# Title\n\nContent",
            metadata={},
            id_="doc-1",
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        result = await adapter.load_eager()

        assert len(result) == 1
        # Should still have extracted metadata, just no _llamaindex key
        # (or _llamaindex is empty)
        extracted = result[0].extracted_metadata
        assert extracted is not None


class TestMultiDocumentHandling:
    """Tests for multi-document validation."""

    @pytest.mark.asyncio
    async def test_single_doc_default(self):
        """Raises ValueError if multiple docs and allow_multiple=False."""
        docs = [
            MockLlamaIndexDocument(text="Doc 1", metadata={"path": "doc1.md"}),
            MockLlamaIndexDocument(text="Doc 2", metadata={"path": "doc2.md"}),
        ]
        loader = MockLoader(docs)
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
            allow_multiple=False,
        )

        with pytest.raises(ValueError, match="allow_multiple=False"):
            await adapter.load_eager()

    @pytest.mark.asyncio
    async def test_allow_multiple_true(self):
        """Accepts multiple docs when allow_multiple=True."""
        docs = [
            MockLlamaIndexDocument(text="Doc 1", metadata={"path": "doc1.md"}),
            MockLlamaIndexDocument(text="Doc 2", metadata={"path": "doc2.md"}),
        ]
        loader = MockLoader(docs)
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
            allow_multiple=True,
        )

        result = await adapter.load_eager()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_duplicate_uri_rejected(self):
        """Raises ValueError on duplicate URIs."""
        docs = [
            MockLlamaIndexDocument(text="Doc 1", metadata={"path": "same.md"}),
            MockLlamaIndexDocument(text="Doc 2", metadata={"path": "same.md"}),
        ]
        loader = MockLoader(docs)
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
            allow_multiple=True,
        )

        with pytest.raises(ValueError, match="Duplicate URI"):
            await adapter.load_eager()


class TestContentAndTitle:
    """Tests for content extraction and title handling."""

    @pytest.mark.asyncio
    async def test_content_type_propagated(self):
        """content_type appears in LoadedDocument."""
        doc = MockLlamaIndexDocument(
            text="Content",
            metadata={"path": "test.md"},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/html",
        )

        result = await adapter.load_eager()

        assert result[0].content_type == "text/html"

    @pytest.mark.asyncio
    async def test_title_from_metadata(self):
        """Uses title from metadata when present."""
        doc = MockLlamaIndexDocument(
            text="Content without heading",
            metadata={"path": "test.md", "title": "Custom Title"},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        result = await adapter.load_eager()

        assert result[0].title == "Custom Title"

    @pytest.mark.asyncio
    async def test_title_from_heading(self):
        """Extracts title from markdown heading."""
        doc = MockLlamaIndexDocument(
            text="# Heading Title\n\nContent",
            metadata={"path": "test.md"},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        result = await adapter.load_eager()

        assert result[0].title == "Heading Title"

    @pytest.mark.asyncio
    async def test_title_fallback_to_path(self):
        """Falls back to path when no title or heading."""
        doc = MockLlamaIndexDocument(
            text="Content without heading",
            metadata={"path": "my-document.md"},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        result = await adapter.load_eager()

        assert result[0].title == "my-document.md"


class TestContentExtraction:
    """Tests for extracting content from LlamaIndex documents."""

    @pytest.mark.asyncio
    async def test_content_from_text_attribute(self):
        """Extracts content from .text attribute."""
        doc = MockLlamaIndexDocument(
            text="Document content here",
            metadata={"path": "test.md"},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        result = await adapter.load_eager()

        assert result[0].content == "Document content here"

    @pytest.mark.asyncio
    async def test_content_from_get_content_method(self):
        """Falls back to get_content() method."""

        class DocWithMethod:
            metadata = {"path": "test.md"}
            id_ = None

            def get_content(self):
                return "Content from method"

        loader = MockLoader([])
        loader._documents = [DocWithMethod()]

        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        result = await adapter.load_eager()

        assert result[0].content == "Content from method"


class TestLoadingServiceIntegration:
    """Tests for LoadingService.load_from_llamaindex integration."""

    @pytest.mark.asyncio
    async def test_collection_id_injected(self, config: Config):
        """LoadingService injects collection_id correctly."""
        doc = MockLlamaIndexDocument(
            text="Content",
            metadata={"path": "test.md"},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        async with ServiceContainer(config) as services:
            # Create a collection
            services.collection_repo.create("llama-test", "/tmp", "**/*.md")

            from pmd.sources import get_default_registry

            source_metadata_repo = SourceMetadataRepository(services.db)
            loading_service = LoadingService(
                db=services.db,
                collection_repo=services.collection_repo,
                document_repo=services.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            result = await loading_service.load_from_llamaindex("llama-test", adapter)

            assert len(result.documents) == 1
            # collection_id should be injected (not 0)
            assert result.documents[0].collection_id > 0

    @pytest.mark.asyncio
    async def test_collection_not_found(self, config: Config):
        """Raises CollectionNotFoundError for unknown collection."""
        doc = MockLlamaIndexDocument(
            text="Content",
            metadata={"path": "test.md"},
        )
        loader = MockLoader([doc])
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
        )

        async with ServiceContainer(config) as services:
            from pmd.sources import get_default_registry

            source_metadata_repo = SourceMetadataRepository(services.db)
            loading_service = LoadingService(
                db=services.db,
                collection_repo=services.collection_repo,
                document_repo=services.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            with pytest.raises(CollectionNotFoundError):
                await loading_service.load_from_llamaindex("nonexistent", adapter)

    @pytest.mark.asyncio
    async def test_enumerated_paths_populated(self, config: Config):
        """EagerLoadResult has enumerated_paths populated."""
        docs = [
            MockLlamaIndexDocument(text="Doc 1", metadata={"path": "doc1.md"}),
            MockLlamaIndexDocument(text="Doc 2", metadata={"path": "doc2.md"}),
        ]
        loader = MockLoader(docs)
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
            allow_multiple=True,
        )

        async with ServiceContainer(config) as services:
            services.collection_repo.create("llama-test", "/tmp", "**/*.md")

            from pmd.sources import get_default_registry

            source_metadata_repo = SourceMetadataRepository(services.db)
            loading_service = LoadingService(
                db=services.db,
                collection_repo=services.collection_repo,
                document_repo=services.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            result = await loading_service.load_from_llamaindex("llama-test", adapter)

            assert len(result.enumerated_paths) == 2
            assert "doc1.md" in result.enumerated_paths
            assert "doc2.md" in result.enumerated_paths


class TestLoadKwargs:
    """Tests for passing kwargs to loader."""

    @pytest.mark.asyncio
    async def test_load_kwargs_passed(self):
        """load_kwargs are passed to loader.load_data()."""

        class KwargsCapturingLoader:
            captured_kwargs: dict = {}

            def load_data(self, **kwargs):
                self.captured_kwargs = kwargs
                return [MockLlamaIndexDocument(text="Content", metadata={"path": "test.md"})]

        loader = KwargsCapturingLoader()
        adapter = LlamaIndexLoaderAdapter(
            loader=loader,
            content_type="text/markdown",
            load_kwargs={"urls": ["https://example.com"], "timeout": 30},
        )

        await adapter.load_eager()

        assert loader.captured_kwargs == {"urls": ["https://example.com"], "timeout": 30}
