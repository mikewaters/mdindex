"""Tests for LoadingService."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pmd.core.config import Config
from pmd.core.exceptions import SourceCollectionNotFoundError
from pmd.app import create_application
from pmd.services.loading import LoadingService, LoadedDocument, EagerLoadResult, LoadResult
from pmd.sources import FileSystemSource, SourceConfig, SourceFetchError
from pmd.sources.content.base import DocumentReference, FetchResult
from pmd.store.repositories.source_metadata import SourceMetadataRepository


def _filesystem_source_for(collection) -> FileSystemSource:
    return FileSystemSource(
        SourceConfig(
            uri=collection.get_source_uri(),
            extra=collection.get_source_config_dict(),
        )
    )


class TestLoadingServiceEager:
    """Tests for LoadingService.load_collection_eager method."""

    @pytest.mark.asyncio
    async def test_load_eager_returns_all_documents(self, config: Config, tmp_path: Path):
        """Eager mode returns complete list of documents."""
        # Create test files
        (tmp_path / "doc1.md").write_text("# Document 1\n\nContent one.")
        (tmp_path / "doc2.md").write_text("# Document 2\n\nContent two.")

        async with await create_application(config) as app:
            app.source_collection_repo.create("test", str(tmp_path), "**/*.md")
            collection = app.source_collection_repo.get_by_name("test")
            source = _filesystem_source_for(collection)
            source_metadata_repo = SourceMetadataRepository(app.db)

            from pmd.sources import get_default_registry
            loader = LoadingService(
                db=app.db,
                source_collection_repo=app.source_collection_repo,
                document_repo=app.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            result = await loader.load_collection_eager("test", source=source)

            assert isinstance(result, EagerLoadResult)
            assert len(result.documents) == 2
            assert all(isinstance(doc, LoadedDocument) for doc in result.documents)
            assert result.errors == []

    @pytest.mark.asyncio
    async def test_enumerated_paths_complete(self, config: Config, tmp_path: Path):
        """All paths are in enumerated_paths even if some are skipped."""
        (tmp_path / "doc1.md").write_text("# Document 1\n\nContent.")
        (tmp_path / "doc2.md").write_text("# Document 2\n\nContent.")

        async with await create_application(config) as app:
            app.source_collection_repo.create("test", str(tmp_path), "**/*.md")
            collection = app.source_collection_repo.get_by_name("test")
            source = _filesystem_source_for(collection)
            source_metadata_repo = SourceMetadataRepository(app.db)

            from pmd.sources import get_default_registry
            loader = LoadingService(
                db=app.db,
                source_collection_repo=app.source_collection_repo,
                document_repo=app.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            result = await loader.load_collection_eager("test", source=source)

            # Both paths should be enumerated
            assert len(result.enumerated_paths) == 2
            paths = result.enumerated_paths
            assert any("doc1.md" in p for p in paths)
            assert any("doc2.md" in p for p in paths)

    @pytest.mark.asyncio
    async def test_collection_not_found_raises(self, config: Config):
        """load_collection_eager raises for unknown collection."""
        async with await create_application(config) as app:
            source_metadata_repo = SourceMetadataRepository(app.db)

            from pmd.sources import get_default_registry
            loader = LoadingService(
                db=app.db,
                source_collection_repo=app.source_collection_repo,
                document_repo=app.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            with pytest.raises(SourceCollectionNotFoundError):
                await loader.load_collection_eager("nonexistent")

    @pytest.mark.asyncio
    async def test_resolves_source_from_collection(self, config: Config, tmp_path: Path):
        """Source is created from registry when None."""
        (tmp_path / "doc.md").write_text("# Test\n\nContent.")

        async with await create_application(config) as app:
            app.source_collection_repo.create("test", str(tmp_path), "**/*.md")
            source_metadata_repo = SourceMetadataRepository(app.db)

            from pmd.sources import get_default_registry
            loader = LoadingService(
                db=app.db,
                source_collection_repo=app.source_collection_repo,
                document_repo=app.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            # Don't pass source - should be resolved from collection
            result = await loader.load_collection_eager("test")

            assert len(result.documents) == 1


class TestLoadingServiceStream:
    """Tests for LoadingService.load_collection_stream method."""

    @pytest.mark.asyncio
    async def test_load_stream_yields_documents(self, config: Config, tmp_path: Path):
        """Stream mode yields documents as iterator."""
        (tmp_path / "doc1.md").write_text("# Document 1\n\nContent one.")
        (tmp_path / "doc2.md").write_text("# Document 2\n\nContent two.")

        async with await create_application(config) as app:
            app.source_collection_repo.create("test", str(tmp_path), "**/*.md")
            collection = app.source_collection_repo.get_by_name("test")
            source = _filesystem_source_for(collection)
            source_metadata_repo = SourceMetadataRepository(app.db)

            from pmd.sources import get_default_registry
            loader = LoadingService(
                db=app.db,
                source_collection_repo=app.source_collection_repo,
                document_repo=app.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            result = await loader.load_collection_stream("test", source=source)

            assert isinstance(result, LoadResult)
            # enumerated_paths should be populated before iteration
            assert len(result.enumerated_paths) == 2

            # Collect documents from iterator
            docs = []
            async for doc in result.documents:
                docs.append(doc)
                assert isinstance(doc, LoadedDocument)

            assert len(docs) == 2


class TestLoadingServiceChangeDetection:
    """Tests for LoadingService change detection."""

    @pytest.mark.asyncio
    async def test_skip_unchanged_content_hash(self, config: Config, tmp_path: Path):
        """Skips when content hash matches existing document."""
        (tmp_path / "doc.md").write_text("# Test\n\nContent.")

        async with await create_application(config) as app:
            app.source_collection_repo.create("test", str(tmp_path), "**/*.md")
            collection = app.source_collection_repo.get_by_name("test")
            source = _filesystem_source_for(collection)
            source_metadata_repo = SourceMetadataRepository(app.db)

            from pmd.sources import get_default_registry
            loader = LoadingService(
                db=app.db,
                source_collection_repo=app.source_collection_repo,
                document_repo=app.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            # First load - should return document
            result1 = await loader.load_collection_eager("test", source=source)
            assert len(result1.documents) == 1

            # Index the document so hash is stored
            doc = result1.documents[0]
            app.document_repo.add_or_update(
                collection.id,
                doc.path,
                doc.title,
                doc.content,
            )

            # Second load - should skip (content unchanged)
            result2 = await loader.load_collection_eager("test", source=source)
            assert len(result2.documents) == 0
            # But path should still be enumerated
            assert len(result2.enumerated_paths) == 1

    @pytest.mark.asyncio
    async def test_force_reloads_all(self, config: Config, tmp_path: Path):
        """force=True ignores change detection and reloads all."""
        (tmp_path / "doc.md").write_text("# Test\n\nContent.")

        async with await create_application(config) as app:
            app.source_collection_repo.create("test", str(tmp_path), "**/*.md")
            collection = app.source_collection_repo.get_by_name("test")
            source = _filesystem_source_for(collection)
            source_metadata_repo = SourceMetadataRepository(app.db)

            from pmd.sources import get_default_registry
            loader = LoadingService(
                db=app.db,
                source_collection_repo=app.source_collection_repo,
                document_repo=app.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            # First load and index
            result1 = await loader.load_collection_eager("test", source=source)
            doc = result1.documents[0]
            app.document_repo.add_or_update(
                collection.id,
                doc.path,
                doc.title,
                doc.content,
            )

            # Second load with force=True - should return document
            result2 = await loader.load_collection_eager("test", source=source, force=True)
            assert len(result2.documents) == 1


class TestLoadingServiceTitleExtraction:
    """Tests for LoadingService title extraction."""

    @pytest.mark.asyncio
    async def test_extracts_title_from_content(self, config: Config, tmp_path: Path):
        """Title is extracted from markdown heading."""
        (tmp_path / "doc.md").write_text("# My Document Title\n\nContent.")

        async with await create_application(config) as app:
            app.source_collection_repo.create("test", str(tmp_path), "**/*.md")
            collection = app.source_collection_repo.get_by_name("test")
            source = _filesystem_source_for(collection)
            source_metadata_repo = SourceMetadataRepository(app.db)

            from pmd.sources import get_default_registry
            loader = LoadingService(
                db=app.db,
                source_collection_repo=app.source_collection_repo,
                document_repo=app.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            result = await loader.load_collection_eager("test", source=source)

            assert len(result.documents) == 1
            assert result.documents[0].title == "My Document Title"

    @pytest.mark.asyncio
    async def test_title_fallback_to_filename(self, config: Config, tmp_path: Path):
        """Title falls back to filename when no heading."""
        (tmp_path / "my-doc.md").write_text("No heading here, just content.")

        async with await create_application(config) as app:
            app.source_collection_repo.create("test", str(tmp_path), "**/*.md")
            collection = app.source_collection_repo.get_by_name("test")
            source = _filesystem_source_for(collection)
            source_metadata_repo = SourceMetadataRepository(app.db)

            from pmd.sources import get_default_registry
            loader = LoadingService(
                db=app.db,
                source_collection_repo=app.source_collection_repo,
                document_repo=app.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            result = await loader.load_collection_eager("test", source=source)

            assert len(result.documents) == 1
            # Should use the filename stem as title
            assert result.documents[0].title == "my-doc"


class TestLoadingServiceErrorHandling:
    """Tests for LoadingService error handling."""

    @pytest.mark.asyncio
    async def test_errors_captured_not_raised(self, config: Config, tmp_path: Path):
        """Fetch errors are captured in errors list, not raised."""
        # Create a source that will fail to fetch one document
        (tmp_path / "good.md").write_text("# Good\n\nContent.")

        async with await create_application(config) as app:
            app.source_collection_repo.create("test", str(tmp_path), "**/*.md")
            collection = app.source_collection_repo.get_by_name("test")
            source_metadata_repo = SourceMetadataRepository(app.db)

            # Create a mock source that fails on one document
            mock_source = MagicMock()
            mock_source.list_documents.return_value = [
                DocumentReference(uri="file:///good.md", path="good.md"),
                DocumentReference(uri="file:///bad.md", path="bad.md"),
            ]

            good_result = FetchResult(content="# Good\n\nContent.", content_type="text/markdown")

            async def mock_fetch(ref):
                if "bad" in ref.path:
                    raise SourceFetchError("file:///bad.md", "File not found")
                return good_result

            mock_source.fetch_content = mock_fetch
            mock_source.check_modified = AsyncMock(return_value=True)

            from pmd.sources import get_default_registry
            loader = LoadingService(
                db=app.db,
                source_collection_repo=app.source_collection_repo,
                document_repo=app.document_repo,
                source_metadata_repo=source_metadata_repo,
                source_registry=get_default_registry(),
            )

            # Should not raise, but capture error
            result = await loader.load_collection_eager("test", source=mock_source)

            assert len(result.documents) == 1  # Only good doc
            assert len(result.errors) == 1  # One error captured
            assert "bad.md" in result.errors[0][0]


class TestLoadedDocumentAccessors:
    """Tests for LoadedDocument convenience accessors."""

    def test_content_accessor(self):
        """content property returns fetch_result.content."""
        fetch_result = FetchResult(content="Test content", content_type="text/markdown")
        ref = DocumentReference(uri="file:///test.md", path="test.md")

        doc = LoadedDocument(
            ref=ref,
            fetch_result=fetch_result,
            title="Test",
            fetch_duration_ms=100,
            source_collection_id=1,
        )

        assert doc.content == "Test content"

    def test_content_type_accessor(self):
        """content_type property returns fetch_result.content_type."""
        fetch_result = FetchResult(content="Test", content_type="text/markdown")
        ref = DocumentReference(uri="file:///test.md", path="test.md")

        doc = LoadedDocument(
            ref=ref,
            fetch_result=fetch_result,
            title="Test",
            fetch_duration_ms=100,
            source_collection_id=1,
        )

        assert doc.content_type == "text/markdown"

    def test_path_accessor(self):
        """path property returns ref.path."""
        fetch_result = FetchResult(content="Test", content_type="text/markdown")
        ref = DocumentReference(uri="file:///docs/test.md", path="docs/test.md")

        doc = LoadedDocument(
            ref=ref,
            fetch_result=fetch_result,
            title="Test",
            fetch_duration_ms=100,
            source_collection_id=1,
        )

        assert doc.path == "docs/test.md"
