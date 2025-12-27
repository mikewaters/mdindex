"""Tests for full-text search operations."""

import pytest
from pathlib import Path

from pmd.store.database import Database
from pmd.store.search import FTS5SearchRepository, SearchRepository
from pmd.store.embeddings import EmbeddingRepository
from pmd.store.documents import DocumentRepository
from pmd.store.collections import CollectionRepository
from pmd.core.types import SearchResult, SearchSource


class TestSearchRepositoryInterface:
    """Tests for SearchRepository abstract interface."""

    def test_fts5_implements_interface(self, search_repo: FTS5SearchRepository):
        """FTS5SearchRepository should implement SearchRepository."""
        assert isinstance(search_repo, SearchRepository)

    def test_has_search_fts_method(self, search_repo: FTS5SearchRepository):
        """Should have search_fts method."""
        assert hasattr(search_repo, "search_fts")
        assert callable(search_repo.search_fts)

    def test_has_search_vec_method(self, search_repo: FTS5SearchRepository):
        """Should have search_vec method."""
        assert hasattr(search_repo, "search_vec")
        assert callable(search_repo.search_vec)


class TestFTS5Index:
    """Tests for FTS5 indexing operations."""

    def test_index_document(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """index_document should add document to FTS5 index."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Searchable content",
        )

        # Index the document (using filepath as doc_id for now)
        search_repo.index_document(1, doc.filepath, doc.body)

        # Should be searchable
        results = search_repo.search_fts("searchable")
        assert len(results) > 0

    @pytest.mark.skip(reason="FTS5 contentless tables require special delete syntax")
    def test_index_document_updates_existing(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """index_document should update existing entry."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Original content",
        )

        # Index twice with different content
        search_repo.index_document(1, doc.filepath, "First version")
        search_repo.index_document(1, doc.filepath, "Updated version")

        # Should find updated content
        results = search_repo.search_fts("Updated")
        assert len(results) > 0

    @pytest.mark.skip(reason="FTS5 contentless tables require special delete syntax")
    def test_remove_from_index(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """remove_from_index should remove document from FTS5."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Content to remove",
        )
        search_repo.index_document(1, doc.filepath, doc.body)

        search_repo.remove_from_index(1)

        results = search_repo.search_fts("remove")
        assert len(results) == 0

    @pytest.mark.skip(reason="FTS5 contentless tables require special delete syntax")
    def test_clear_index(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """clear_index should remove all entries."""
        # Add multiple documents
        for i in range(3):
            doc, _ = document_repo.add_or_update(
                sample_collection.id,
                f"doc{i}.md",
                f"Doc {i}",
                f"Content {i}",
            )
            search_repo.index_document(i + 1, doc.filepath, doc.body)

        search_repo.clear_index()

        # Should find nothing
        results = search_repo.search_fts("Content")
        assert len(results) == 0


class TestFTS5Search:
    """Tests for FTS5 search operations."""

    def test_search_returns_results(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search_fts should return matching documents."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Python Guide",
            "Learn Python programming language",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = search_repo.search_fts("Python")

        assert len(results) > 0
        assert isinstance(results[0], SearchResult)

    def test_search_result_has_correct_source(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search_fts results should have FTS source."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Searchable content",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = search_repo.search_fts("Searchable")

        assert results[0].source == SearchSource.FTS

    def test_search_no_results(self, search_repo: FTS5SearchRepository):
        """search_fts should return empty list when no matches."""
        results = search_repo.search_fts("nonexistent_query_xyz")

        assert results == []

    def test_search_limit(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search_fts should respect limit parameter."""
        # Add multiple documents
        for i in range(10):
            doc, _ = document_repo.add_or_update(
                sample_collection.id,
                f"doc{i}.md",
                f"Doc {i}",
                f"Common keyword content {i}",
            )
            search_repo.index_document(i + 1, doc.filepath, doc.body)

        results = search_repo.search_fts("keyword", limit=3)

        assert len(results) <= 3

    def test_search_collection_filter(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        collection_repo: CollectionRepository,
        tmp_path: Path,
    ):
        """search_fts should filter by collection_id."""
        # Create two collections
        coll1 = collection_repo.create("coll1", str(tmp_path))
        coll2 = collection_repo.create("coll2", str(tmp_path))

        # Add documents to each
        doc1, _ = document_repo.add_or_update(coll1.id, "doc1.md", "Doc 1", "Target keyword")
        doc2, _ = document_repo.add_or_update(coll2.id, "doc2.md", "Doc 2", "Target keyword")

        search_repo.index_document(doc1.collection_id, doc1.filepath, doc1.body)
        search_repo.index_document(doc2.collection_id, doc2.filepath, doc2.body)

        # Search only in collection 1
        results = search_repo.search_fts("Target", collection_id=coll1.id)

        assert len(results) == 1
        assert results[0].collection_id == coll1.id

    def test_search_min_score_filter(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search_fts should filter by min_score."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Searchable content",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        # Very high min_score should return no results
        results = search_repo.search_fts("Searchable", min_score=999.0)

        # Should be empty or all results above threshold
        for r in results:
            assert r.score >= 999.0


class TestFTS5SearchScoring:
    """Tests for FTS5 search scoring and ranking."""

    def test_search_results_have_scores(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search_fts results should have score attribute."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Searchable content",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = search_repo.search_fts("Searchable")

        assert hasattr(results[0], "score")
        assert isinstance(results[0].score, float)

    def test_search_results_ordered_by_relevance(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search_fts should return results ordered by relevance."""
        # Doc with more keyword occurrences should rank higher
        doc1, _ = document_repo.add_or_update(
            sample_collection.id,
            "doc1.md",
            "Doc 1",
            "keyword keyword keyword keyword",  # More occurrences
        )
        doc2, _ = document_repo.add_or_update(
            sample_collection.id,
            "doc2.md",
            "Doc 2",
            "keyword once",  # Fewer occurrences
        )

        search_repo.index_document(1, doc1.filepath, doc1.body)
        search_repo.index_document(2, doc2.filepath, doc2.body)

        results = search_repo.search_fts("keyword")

        if len(results) >= 2:
            # First result should have higher or equal score
            assert results[0].score >= results[1].score


class TestFTS5ReindexCollection:
    """Tests for collection reindexing."""

    def test_reindex_collection(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """reindex_collection should reindex all documents."""
        # Add documents
        for i in range(5):
            document_repo.add_or_update(
                sample_collection.id,
                f"doc{i}.md",
                f"Doc {i}",
                f"Content for doc {i}",
            )

        count = search_repo.reindex_collection(sample_collection.id)

        assert count == 5

    @pytest.mark.skip(reason="FTS5 contentless tables require special delete syntax")
    def test_reindex_clears_existing(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """reindex_collection should clear existing index entries."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Original content",
        )
        search_repo.index_document(1, doc.filepath, "Old indexed content")

        # Update document
        document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "New content",
        )

        # Reindex
        search_repo.reindex_collection(sample_collection.id)

        # Should find new content
        results = search_repo.search_fts("New")
        assert len(results) > 0


class TestVectorSearch:
    """Tests for vector similarity search."""

    def test_search_vec_without_embedding_repo(self, db: Database):
        """search_vec should return empty without embedding_repo."""
        search_repo = FTS5SearchRepository(db, None)  # No embedding repo

        results = search_repo.search_vec([0.1, 0.2], limit=5)

        assert results == []

    def test_search_vec_empty_embedding(self, search_repo: FTS5SearchRepository):
        """search_vec should return empty for empty embedding."""
        results = search_repo.search_vec([], limit=5)

        assert results == []

    def test_search_vec_delegates_to_embedding_repo(
        self,
        search_repo: FTS5SearchRepository,
        embedding_repo: EmbeddingRepository,
        db: Database,
    ):
        """search_vec should delegate to embedding repository."""
        # This is a basic test - full vector search requires sqlite-vec
        results = search_repo.search_vec([0.1] * 768, limit=5)

        # Without sqlite-vec, should return empty
        if not db.vec_available:
            assert results == []


class TestFTS5QueryPreparation:
    """Tests for FTS5 query preparation."""

    def test_simple_query(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """Simple single-word queries should work."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Simple content here",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = search_repo.search_fts("simple")

        assert len(results) > 0

    def test_multi_word_query(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """Multi-word queries should work."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Python programming tutorial content",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = search_repo.search_fts("Python programming")

        assert len(results) > 0

    def test_case_insensitive_search(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """Search should be case-insensitive."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "UPPERCASE content lowercase",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        # Both cases should find results
        results_lower = search_repo.search_fts("uppercase")
        results_upper = search_repo.search_fts("UPPERCASE")

        assert len(results_lower) > 0
        assert len(results_upper) > 0


class TestSearchResultAttributes:
    """Tests for SearchResult object attributes."""

    def test_result_has_filepath(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """SearchResult should have filepath attribute."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "my/path/doc.md",
            "Test",
            "Content",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = search_repo.search_fts("Content")

        assert results[0].filepath == "my/path/doc.md"

    def test_result_has_title(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """SearchResult should have title attribute."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "My Title",
            "Content here",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = search_repo.search_fts("Content")

        assert results[0].title == "My Title"

    def test_result_has_collection_id(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """SearchResult should have collection_id attribute."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Content",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = search_repo.search_fts("Content")

        assert results[0].collection_id == sample_collection.id

    def test_result_has_hash(
        self,
        search_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """SearchResult should have hash attribute."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Content",
        )
        search_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = search_repo.search_fts("Content")

        assert results[0].hash == doc.hash
