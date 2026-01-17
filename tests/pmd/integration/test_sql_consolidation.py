"""Integration tests for SQL consolidation.

These tests verify that services properly use repository methods instead of
direct SQL, and that the refactored code produces correct results.
"""

import pytest
from pathlib import Path

from pmd.app import create_application
from pmd.sources import FileSystemSource, SourceConfig
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.database import Database
from pmd.store.repositories.documents import DocumentRepository
from pmd.store.repositories.embeddings import EmbeddingRepository
from pmd.store.repositories.fts import FTS5SearchRepository
from pmd.store.repositories.content import ContentRepository


def _filesystem_source_for(collection) -> FileSystemSource:
    """Create a filesystem source for a collection."""
    return FileSystemSource(
        SourceConfig(
            uri=collection.get_source_uri(),
            extra=collection.get_source_config_dict(),
        )
    )


class TestDocumentRepositoryIntegration:
    """Integration tests for DocumentRepository methods."""

    @pytest.mark.asyncio
    async def test_get_id_returns_correct_document_id(
        self, config, test_corpus_path: Path
    ):
        """get_id should return the correct document ID after indexing."""
        async with await create_application(config) as app:
            # Create collection and index
            collection = app.source_collection_repo.create(
                "test-get-id",
                str(test_corpus_path),
                "**/*.md",
            )
            source = _filesystem_source_for(collection)
            await app.indexing.index_collection("test-get-id", source)

            # Verify get_id returns valid ID for a known document
            doc_id = app.document_repo.get_id(collection.id, "Graph RAG.md")
            assert doc_id is not None
            assert isinstance(doc_id, int)
            assert doc_id > 0

    @pytest.mark.asyncio
    async def test_get_ids_by_paths_returns_mapping(
        self, config, test_corpus_path: Path
    ):
        """get_ids_by_paths should return correct mapping after indexing."""
        async with await create_application(config) as app:
            collection = app.source_collection_repo.create(
                "test-ids-by-paths",
                str(test_corpus_path),
                "**/*.md",
            )
            source = _filesystem_source_for(collection)
            await app.indexing.index_collection("test-ids-by-paths", source)

            # Query multiple paths (using known files in test corpus)
            paths = ["Graph RAG.md", "Highlight.md"]
            mapping = app.document_repo.get_ids_by_paths(paths)

            # Should return mapping for both paths
            assert len(mapping) == 2
            assert "Graph RAG.md" in mapping
            assert "Highlight.md" in mapping
            assert all(isinstance(v, int) for v in mapping.values())

    @pytest.mark.asyncio
    async def test_count_active_returns_correct_count(
        self, config, test_corpus_path: Path
    ):
        """count_active should return correct document count after indexing."""
        async with await create_application(config) as app:
            collection = app.source_collection_repo.create(
                "test-count-active",
                str(test_corpus_path),
                "**/*.md",
            )
            source = _filesystem_source_for(collection)
            result = await app.indexing.index_collection("test-count-active", source)

            # count_active should match indexed count
            count = app.document_repo.count_active(collection.id)
            assert count == result.indexed

    @pytest.mark.asyncio
    async def test_list_active_with_content_returns_tuples(
        self, config, test_corpus_path: Path
    ):
        """list_active_with_content should return (path, hash, content) tuples."""
        async with await create_application(config) as app:
            collection = app.source_collection_repo.create(
                "test-list-active",
                str(test_corpus_path),
                "**/*.md",
            )
            source = _filesystem_source_for(collection)
            result = await app.indexing.index_collection("test-list-active", source)

            # List active documents with content
            docs = app.document_repo.list_active_with_content(collection.id)

            # Should return correct number of tuples
            assert len(docs) == result.indexed

            # Each tuple should have (path, hash, content)
            for path, hash_val, content in docs:
                assert isinstance(path, str)
                assert isinstance(hash_val, str)
                assert isinstance(content, str)
                assert len(hash_val) == 64  # SHA256 hex


class TestStatusServiceIntegration:
    """Integration tests for StatusService with repository methods."""

    @pytest.mark.asyncio
    async def test_status_after_indexing(self, config, test_corpus_path: Path):
        """StatusService should return correct counts after indexing."""
        async with await create_application(config) as app:
            # Create and index collection
            collection = app.source_collection_repo.create(
                "test-status",
                str(test_corpus_path),
                "**/*.md",
            )
            source = _filesystem_source_for(collection)
            result = await app.indexing.index_collection("test-status", source)

            # Get status (not async)
            status = app.status.get_index_status()

            # Verify counts match
            assert status.total_documents == result.indexed
            assert len(status.source_collections) == 1
            assert status.source_collections[0].name == "test-status"


class TestOrphanCleanupIntegration:
    """Integration tests for orphan cleanup via repository methods."""

    @pytest.mark.asyncio
    async def test_orphan_count_after_delete(
        self, config, test_corpus_path: Path
    ):
        """ContentRepository should correctly count orphans after document deletion."""
        async with await create_application(config) as app:
            content_repo = ContentRepository(app.db)

            # Create and index collection
            collection = app.source_collection_repo.create(
                "test-orphan-count",
                str(test_corpus_path),
                "**/*.md",
            )
            source = _filesystem_source_for(collection)
            await app.indexing.index_collection("test-orphan-count", source)

            # Initially no orphans
            assert content_repo.count_orphaned() == 0

            # Soft-delete a document (using valid path from test corpus)
            app.document_repo.delete(collection.id, "Graph RAG.md")

            # Should now have orphaned content
            # (unless content is shared with another doc, which is unlikely)
            orphan_count = content_repo.count_orphaned()
            assert orphan_count >= 1

    @pytest.mark.asyncio
    async def test_embedding_orphan_detection(
        self, config, test_corpus_path: Path
    ):
        """EmbeddingRepository should correctly detect documents missing embeddings."""
        async with await create_application(config) as app:
            embed_repo = EmbeddingRepository(app.db)

            # Create and index collection
            collection = app.source_collection_repo.create(
                "test-embed-orphan",
                str(test_corpus_path),
                "**/*.md",
            )
            source = _filesystem_source_for(collection)
            result = await app.indexing.index_collection("test-embed-orphan", source)

            # Without embeddings, all docs should be missing embeddings
            missing_count = embed_repo.count_documents_missing_embeddings()
            assert missing_count == result.indexed

            # List should return paths
            missing_paths = embed_repo.list_paths_missing_embeddings(limit=5)
            assert len(missing_paths) == 5  # Limited to 5
            assert all(isinstance(p, str) for p in missing_paths)

    @pytest.mark.asyncio
    async def test_fts_orphan_detection(
        self, config, test_corpus_path: Path
    ):
        """FTS5SearchRepository should correctly detect documents missing FTS entries."""
        async with await create_application(config) as app:
            fts_repo = FTS5SearchRepository(app.db)

            # Create and index collection
            collection = app.source_collection_repo.create(
                "test-fts-orphan",
                str(test_corpus_path),
                "**/*.md",
            )
            source = _filesystem_source_for(collection)
            result = await app.indexing.index_collection("test-fts-orphan", source)

            # After indexing, FTS should be populated for indexable content
            # Note: Some documents may not be indexed (very short content, etc.)
            missing_count = fts_repo.count_documents_missing_fts()

            # Verify FTS is mostly populated (most docs should be indexed)
            # Some docs won't be indexed due to short content or other reasons
            assert missing_count < result.indexed

            # Verify the missing paths list works
            missing_paths = fts_repo.list_paths_missing_fts(limit=5)
            assert len(missing_paths) <= min(5, missing_count)


class TestPipelineIntegration:
    """Integration tests for pipeline modules using repository methods."""

    @pytest.mark.asyncio
    async def test_indexing_pipeline_uses_repository_methods(
        self, config, test_corpus_path: Path
    ):
        """Indexing pipeline should work correctly with repository methods."""
        async with await create_application(config) as app:
            # Create and index collection
            collection = app.source_collection_repo.create(
                "test-pipeline",
                str(test_corpus_path),
                "**/*.md",
            )
            source = _filesystem_source_for(collection)
            result = await app.indexing.index_collection("test-pipeline", source)

            # Verify indexing worked
            assert result.indexed > 0
            assert result.errors == []

            # Verify we can retrieve documents using repository methods
            doc_count = app.document_repo.count_active(collection.id)
            assert doc_count == result.indexed

            # Verify FTS detection works (some docs may not be indexed)
            fts_repo = FTS5SearchRepository(app.db)
            missing_fts = fts_repo.count_documents_missing_fts()
            # Most docs should have FTS entries
            assert missing_fts < result.indexed

    @pytest.mark.asyncio
    async def test_reindex_with_force_uses_repository_methods(
        self, config, test_corpus_path: Path
    ):
        """Reindexing with force should work correctly using repository methods."""
        async with await create_application(config) as app:
            # Create and index collection
            collection = app.source_collection_repo.create(
                "test-reindex",
                str(test_corpus_path),
                "**/*.md",
            )
            source = _filesystem_source_for(collection)
            result1 = await app.indexing.index_collection("test-reindex", source)

            # Reindex with force
            result2 = await app.indexing.index_collection(
                "test-reindex", source, force=True
            )

            # Should reindex all documents
            assert result2.indexed == result1.indexed

            # Document count should remain the same
            doc_count = app.document_repo.count_active(collection.id)
            assert doc_count == result1.indexed
