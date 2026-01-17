"""Tests for repository extensions supporting SQL consolidation.

These tests verify the new repository methods added to support the
removal of direct SQL usage outside pmd.store.
"""

import pytest
from pathlib import Path

from pmd.store.database import Database
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.repositories.documents import DocumentRepository
from pmd.store.repositories.content import ContentRepository
from pmd.store.repositories.embeddings import EmbeddingRepository
from pmd.store.repositories.fts import FTS5SearchRepository
from pmd.store.schema import EMBEDDING_DIMENSION


class TestContentRepository:
    """Tests for ContentRepository."""

    def test_count_orphaned_returns_zero_when_empty(self, db: Database):
        """count_orphaned should return 0 for empty database."""
        repo = ContentRepository(db)
        assert repo.count_orphaned() == 0

    def test_count_orphaned_counts_unreferenced_content(self, db: Database):
        """count_orphaned should count content not referenced by active documents."""
        content_repo = ContentRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        # Create a collection and document
        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc_repo.add_or_update(collection.id, "test.md", "Test", "content")

        # Initially no orphans
        assert content_repo.count_orphaned() == 0

        # Soft-delete the document
        doc_repo.delete(collection.id, "test.md")

        # Now content is orphaned
        assert content_repo.count_orphaned() == 1

    def test_delete_orphaned_removes_unreferenced_content(self, db: Database):
        """delete_orphaned should remove content not referenced by any documents.

        Note: The 'documents' table has a foreign key to 'content.hash', so
        content can only be deleted when no documents (active or inactive)
        reference it. This test verifies the orphan cleanup after documents
        are fully removed.
        """
        from pmd.store.repositories.embeddings import EmbeddingRepository

        content_repo = ContentRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)
        embed_repo = EmbeddingRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc, _ = doc_repo.add_or_update(collection.id, "test.md", "Test", "content")

        # Verify content exists
        assert content_repo.count() == 1

        # Delete orphaned embeddings first (they reference content via hash)
        embed_repo.delete_orphaned()

        # Hard-delete the document (not just soft-delete) to remove FK reference
        db.execute("DELETE FROM documents WHERE source_collection_id = ? AND path = ?",
                   (collection.id, "test.md"))

        # Now content is truly orphaned (no documents reference it)
        assert content_repo.count_orphaned() == 1

        # Delete orphaned content
        deleted = content_repo.delete_orphaned()
        assert deleted == 1

        # Verify content is gone
        assert content_repo.count() == 0

    def test_get_by_hash_returns_content(self, db: Database):
        """get_by_hash should return content for existing hash."""
        content_repo = ContentRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc, _ = doc_repo.add_or_update(collection.id, "test.md", "Test", "my content")

        content = content_repo.get_by_hash(doc.hash)
        assert content == "my content"

    def test_get_by_hash_returns_none_for_missing(self, db: Database):
        """get_by_hash should return None for non-existent hash."""
        repo = ContentRepository(db)
        assert repo.get_by_hash("nonexistent") is None

    def test_exists_returns_true_for_existing(self, db: Database):
        """exists should return True for existing content."""
        content_repo = ContentRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc, _ = doc_repo.add_or_update(collection.id, "test.md", "Test", "content")

        assert content_repo.exists(doc.hash)

    def test_exists_returns_false_for_missing(self, db: Database):
        """exists should return False for non-existent content."""
        repo = ContentRepository(db)
        assert not repo.exists("nonexistent")


class TestDocumentRepositoryExtensions:
    """Tests for DocumentRepository extensions."""

    def test_count_active_returns_zero_when_empty(self, db: Database):
        """count_active should return 0 for empty database."""
        repo = DocumentRepository(db)
        assert repo.count_active() == 0

    def test_count_active_counts_active_documents(self, db: Database):
        """count_active should count only active documents."""
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc_repo.add_or_update(collection.id, "doc1.md", "Doc 1", "content 1")
        doc_repo.add_or_update(collection.id, "doc2.md", "Doc 2", "content 2")

        assert doc_repo.count_active() == 2

        doc_repo.delete(collection.id, "doc1.md")
        assert doc_repo.count_active() == 1

    def test_count_active_scoped_by_collection(self, db: Database):
        """count_active should filter by collection when provided."""
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        c1 = collection_repo.create("c1", "/tmp/c1", "**/*.md")
        c2 = collection_repo.create("c2", "/tmp/c2", "**/*.md")

        doc_repo.add_or_update(c1.id, "doc1.md", "Doc 1", "content 1")
        doc_repo.add_or_update(c2.id, "doc2.md", "Doc 2", "content 2")
        doc_repo.add_or_update(c2.id, "doc3.md", "Doc 3", "content 3")

        assert doc_repo.count_active() == 3
        assert doc_repo.count_active(c1.id) == 1
        assert doc_repo.count_active(c2.id) == 2

    def test_get_id_returns_id(self, db: Database):
        """get_id should return document ID."""
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc_repo.add_or_update(collection.id, "test.md", "Test", "content")

        doc_id = doc_repo.get_id(collection.id, "test.md")
        assert doc_id is not None
        assert isinstance(doc_id, int)
        assert doc_id > 0

    def test_get_id_returns_none_for_missing(self, db: Database):
        """get_id should return None for non-existent document."""
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        assert doc_repo.get_id(collection.id, "nonexistent.md") is None

    def test_get_ids_by_paths_returns_mapping(self, db: Database):
        """get_ids_by_paths should return path->ID mapping."""
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc_repo.add_or_update(collection.id, "doc1.md", "Doc 1", "c1")
        doc_repo.add_or_update(collection.id, "doc2.md", "Doc 2", "c2")

        # Get IDs using the get_id method for comparison
        doc1_id = doc_repo.get_id(collection.id, "doc1.md")
        doc2_id = doc_repo.get_id(collection.id, "doc2.md")

        mapping = doc_repo.get_ids_by_paths(["doc1.md", "doc2.md"])
        assert mapping == {"doc1.md": doc1_id, "doc2.md": doc2_id}

    def test_get_ids_by_paths_empty_input(self, db: Database):
        """get_ids_by_paths should return empty dict for empty input."""
        repo = DocumentRepository(db)
        assert repo.get_ids_by_paths([]) == {}

    def test_list_active_with_content_returns_tuples(self, db: Database):
        """list_active_with_content should return (path, hash, content) tuples."""
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc, _ = doc_repo.add_or_update(collection.id, "test.md", "Test", "content")

        results = doc_repo.list_active_with_content(collection.id)
        assert len(results) == 1
        path, hash_val, content = results[0]
        assert path == "test.md"
        assert hash_val == doc.hash
        assert content == "content"


class TestEmbeddingRepositoryExtensions:
    """Tests for EmbeddingRepository extensions."""

    def test_count_distinct_hashes_returns_zero_when_empty(self, db: Database):
        """count_distinct_hashes should return 0 for empty database."""
        repo = EmbeddingRepository(db)
        assert repo.count_distinct_hashes() == 0

    def test_count_distinct_hashes_counts_unique_hashes(self, db: Database):
        """count_distinct_hashes should count unique content hashes."""
        repo = EmbeddingRepository(db)

        # Store embeddings for two different hashes
        repo.store_embedding("hash1", 0, 0, [0.1] * EMBEDDING_DIMENSION, "model")
        repo.store_embedding("hash1", 1, 0, [0.2] * EMBEDDING_DIMENSION, "model")
        repo.store_embedding("hash2", 0, 0, [0.3] * EMBEDDING_DIMENSION, "model")

        assert repo.count_distinct_hashes() == 2

    def test_count_documents_missing_embeddings(self, db: Database):
        """count_documents_missing_embeddings should count docs without embeddings."""
        embed_repo = EmbeddingRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc1, _ = doc_repo.add_or_update(collection.id, "doc1.md", "Doc 1", "c1")
        doc2, _ = doc_repo.add_or_update(collection.id, "doc2.md", "Doc 2", "c2")

        # Both docs missing embeddings initially
        assert embed_repo.count_documents_missing_embeddings() == 2

        # Add embedding for first doc
        embed_repo.store_embedding(
            doc1.hash, 0, 0, [0.1] * EMBEDDING_DIMENSION, "model"
        )

        assert embed_repo.count_documents_missing_embeddings() == 1

    def test_list_paths_missing_embeddings(self, db: Database):
        """list_paths_missing_embeddings should return paths without embeddings."""
        embed_repo = EmbeddingRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc1, _ = doc_repo.add_or_update(collection.id, "doc1.md", "Doc 1", "c1")
        doc2, _ = doc_repo.add_or_update(collection.id, "doc2.md", "Doc 2", "c2")

        paths = embed_repo.list_paths_missing_embeddings()
        assert set(paths) == {"doc1.md", "doc2.md"}

        embed_repo.store_embedding(
            doc1.hash, 0, 0, [0.1] * EMBEDDING_DIMENSION, "model"
        )

        paths = embed_repo.list_paths_missing_embeddings()
        assert paths == ["doc2.md"]

    def test_count_orphaned_embeddings(self, db: Database):
        """count_orphaned should count embeddings without active documents."""
        embed_repo = EmbeddingRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc, _ = doc_repo.add_or_update(collection.id, "test.md", "Test", "content")
        embed_repo.store_embedding(
            doc.hash, 0, 0, [0.1] * EMBEDDING_DIMENSION, "model"
        )

        # No orphans while document active
        assert embed_repo.count_orphaned() == 0

        # Soft-delete document
        doc_repo.delete(collection.id, "test.md")

        # Now embedding is orphaned
        assert embed_repo.count_orphaned() == 1

    def test_delete_orphaned_embeddings(self, db: Database):
        """delete_orphaned should remove embeddings without active documents."""
        embed_repo = EmbeddingRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc, _ = doc_repo.add_or_update(collection.id, "test.md", "Test", "content")
        embed_repo.store_embedding(
            doc.hash, 0, 0, [0.1] * EMBEDDING_DIMENSION, "model"
        )
        doc_repo.delete(collection.id, "test.md")

        deleted = embed_repo.delete_orphaned()
        assert deleted >= 1

        assert embed_repo.count_distinct_hashes() == 0


class TestFTS5RepositoryExtensions:
    """Tests for FTS5SearchRepository extensions."""

    def test_count_documents_missing_fts(self, db: Database):
        """count_documents_missing_fts should count docs without FTS entries."""
        fts_repo = FTS5SearchRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc_repo.add_or_update(collection.id, "doc1.md", "Doc 1", "content 1")
        doc_repo.add_or_update(collection.id, "doc2.md", "Doc 2", "content 2")

        # Both docs missing FTS initially
        assert fts_repo.count_documents_missing_fts() == 2

        # Add FTS for first doc
        doc_id = doc_repo.get_id(collection.id, "doc1.md")
        fts_repo.index_document(doc_id, "doc1.md", "content 1")

        assert fts_repo.count_documents_missing_fts() == 1

    def test_list_paths_missing_fts(self, db: Database):
        """list_paths_missing_fts should return paths without FTS entries."""
        fts_repo = FTS5SearchRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc_repo.add_or_update(collection.id, "doc1.md", "Doc 1", "content 1")
        doc_repo.add_or_update(collection.id, "doc2.md", "Doc 2", "content 2")

        paths = fts_repo.list_paths_missing_fts()
        assert set(paths) == {"doc1.md", "doc2.md"}

        doc_id = doc_repo.get_id(collection.id, "doc1.md")
        fts_repo.index_document(doc_id, "doc1.md", "content 1")

        paths = fts_repo.list_paths_missing_fts()
        assert paths == ["doc2.md"]

    def test_count_orphaned_fts(self, db: Database):
        """count_orphaned should count FTS entries without active documents."""
        fts_repo = FTS5SearchRepository(db)
        doc_repo = DocumentRepository(db)
        collection_repo = SourceCollectionRepository(db)

        collection = collection_repo.create("test", "/tmp", "**/*.md")
        doc_repo.add_or_update(collection.id, "test.md", "Test", "content")
        doc_id = doc_repo.get_id(collection.id, "test.md")
        fts_repo.index_document(doc_id, "test.md", "content")

        # No orphans while document active
        assert fts_repo.count_orphaned() == 0

        # Soft-delete document
        doc_repo.delete(collection.id, "test.md")

        # Now FTS entry is orphaned
        assert fts_repo.count_orphaned() == 1
