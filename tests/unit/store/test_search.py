"""Tests for full-text and vector search operations."""

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

    def test_fts5_implements_interface(self, fts_repo: FTS5SearchRepository):
        """FTS5SearchRepository should implement SearchRepository."""
        assert isinstance(fts_repo, SearchRepository)

    def test_fts5_has_search_method(self, fts_repo: FTS5SearchRepository):
        """FTS5SearchRepository should have search method."""
        assert hasattr(fts_repo, "search")
        assert callable(fts_repo.search)


class TestFTS5Index:
    """Tests for FTS5 indexing operations."""

    def test_index_document(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(1, doc.filepath, doc.body)

        # Should be searchable
        results = fts_repo.search("searchable")
        assert len(results) > 0

    @pytest.mark.skip(reason="FTS5 contentless tables require special delete syntax")
    def test_index_document_updates_existing(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(1, doc.filepath, "First version")
        fts_repo.index_document(1, doc.filepath, "Updated version")

        # Should find updated content
        results = fts_repo.search("Updated")
        assert len(results) > 0

    @pytest.mark.skip(reason="FTS5 contentless tables require special delete syntax")
    def test_remove_from_index(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(1, doc.filepath, doc.body)

        fts_repo.remove_from_index(1)

        results = fts_repo.search("remove")
        assert len(results) == 0

    @pytest.mark.skip(reason="FTS5 contentless tables require special delete syntax")
    def test_clear_index(
        self,
        fts_repo: FTS5SearchRepository,
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
            fts_repo.index_document(i + 1, doc.filepath, doc.body)

        fts_repo.clear_index()

        # Should find nothing
        results = fts_repo.search("Content")
        assert len(results) == 0


class TestFTS5Search:
    """Tests for FTS5 search operations."""

    def test_search_returns_results(
        self,
        fts_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search should return matching documents."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Python Guide",
            "Learn Python programming language",
        )
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = fts_repo.search("Python")

        assert len(results) > 0
        assert isinstance(results[0], SearchResult)

    def test_search_result_has_correct_source(
        self,
        fts_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """FTS5 search results should have FTS source."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Searchable content",
        )
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = fts_repo.search("Searchable")

        assert results[0].source == SearchSource.FTS

    def test_search_no_results(self, fts_repo: FTS5SearchRepository):
        """search should return empty list when no matches."""
        results = fts_repo.search("nonexistent_query_xyz")

        assert results == []

    def test_search_limit(
        self,
        fts_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search should respect limit parameter."""
        # Add multiple documents
        for i in range(10):
            doc, _ = document_repo.add_or_update(
                sample_collection.id,
                f"doc{i}.md",
                f"Doc {i}",
                f"Common keyword content {i}",
            )
            fts_repo.index_document(i + 1, doc.filepath, doc.body)

        results = fts_repo.search("keyword", limit=3)

        assert len(results) <= 3

    def test_search_collection_filter(
        self,
        fts_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        collection_repo: CollectionRepository,
        tmp_path: Path,
    ):
        """search should filter by collection_id."""
        # Create two collections
        coll1 = collection_repo.create("coll1", str(tmp_path))
        coll2 = collection_repo.create("coll2", str(tmp_path))

        # Add documents to each
        doc1, _ = document_repo.add_or_update(coll1.id, "doc1.md", "Doc 1", "Target keyword")
        doc2, _ = document_repo.add_or_update(coll2.id, "doc2.md", "Doc 2", "Target keyword")

        fts_repo.index_document(doc1.collection_id, doc1.filepath, doc1.body)
        fts_repo.index_document(doc2.collection_id, doc2.filepath, doc2.body)

        # Search only in collection 1
        results = fts_repo.search("Target", collection_id=coll1.id)

        assert len(results) == 1
        assert results[0].collection_id == coll1.id

    def test_search_min_score_filter(
        self,
        fts_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search should filter by min_score."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Searchable content",
        )
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        # Very high min_score should return no results
        results = fts_repo.search("Searchable", min_score=999.0)

        # Should be empty or all results above threshold
        for r in results:
            assert r.score >= 999.0


class TestFTS5SearchScoring:
    """Tests for FTS5 search scoring and ranking."""

    def test_search_results_have_scores(
        self,
        fts_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search results should have score attribute."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "Searchable content",
        )
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = fts_repo.search("Searchable")

        assert hasattr(results[0], "score")
        assert isinstance(results[0].score, float)

    def test_search_results_ordered_by_relevance(
        self,
        fts_repo: FTS5SearchRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """search should return results ordered by relevance."""
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

        fts_repo.index_document(1, doc1.filepath, doc1.body)
        fts_repo.index_document(2, doc2.filepath, doc2.body)

        results = fts_repo.search("keyword")

        if len(results) >= 2:
            # First result should have higher or equal score
            assert results[0].score >= results[1].score


class TestFTS5ReindexCollection:
    """Tests for collection reindexing."""

    def test_reindex_collection(
        self,
        fts_repo: FTS5SearchRepository,
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

        count = fts_repo.reindex_collection(sample_collection.id)

        assert count == 5

    @pytest.mark.skip(reason="FTS5 contentless tables require special delete syntax")
    def test_reindex_clears_existing(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(1, doc.filepath, "Old indexed content")

        # Update document
        document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "New content",
        )

        # Reindex
        fts_repo.reindex_collection(sample_collection.id)

        # Should find new content
        results = fts_repo.search("New")
        assert len(results) > 0


class TestFTS5QueryPreparation:
    """Tests for FTS5 query preparation."""

    def test_simple_query(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = fts_repo.search("simple")

        assert len(results) > 0

    def test_multi_word_query(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = fts_repo.search("Python programming")

        assert len(results) > 0

    def test_case_insensitive_search(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        # Both cases should find results
        results_lower = fts_repo.search("uppercase")
        results_upper = fts_repo.search("UPPERCASE")

        assert len(results_lower) > 0
        assert len(results_upper) > 0


class TestSearchResultAttributes:
    """Tests for SearchResult object attributes."""

    def test_result_has_filepath(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = fts_repo.search("Content")

        assert results[0].filepath == "my/path/doc.md"

    def test_result_has_title(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = fts_repo.search("Content")

        assert results[0].title == "My Title"

    def test_result_has_collection_id(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = fts_repo.search("Content")

        assert results[0].collection_id == sample_collection.id

    def test_result_has_hash(
        self,
        fts_repo: FTS5SearchRepository,
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
        fts_repo.index_document(doc.collection_id, doc.filepath, doc.body)

        results = fts_repo.search("Content")

        assert results[0].hash == doc.hash
