"""Tests for tag-based document retrieval."""

import pytest
from unittest.mock import MagicMock, patch

from pmd.search.metadata.retrieval import (
    TagRetriever,
    TagSearchConfig,
    create_tag_retriever,
)
from pmd.core.types import SearchSource


class TestTagRetrieverBasics:
    """Basic functionality tests for TagRetriever."""

    def test_empty_query_tags_returns_empty(self):
        """Empty query tags should return empty results."""
        db = MagicMock()
        metadata_repo = MagicMock()

        retriever = TagRetriever(db, metadata_repo)

        assert retriever.search({}, limit=10) == []
        assert retriever.search(set(), limit=10) == []

    def test_no_matching_documents_returns_empty(self):
        """Should return empty when no documents have matching tags."""
        db = MagicMock()
        metadata_repo = MagicMock()
        metadata_repo.find_documents_with_any_tag.return_value = []

        retriever = TagRetriever(db, metadata_repo)
        results = retriever.search({"python"}, limit=10)

        assert results == []
        metadata_repo.find_documents_with_any_tag.assert_called_once_with({"python"})

    def test_converts_set_to_dict(self):
        """Set of tags should be converted to dict with weight 1.0."""
        db = MagicMock()
        metadata_repo = MagicMock()
        metadata_repo.find_documents_with_any_tag.return_value = []

        retriever = TagRetriever(db, metadata_repo)
        retriever.search({"python", "rust"}, limit=10)

        # Should call with the set
        call_args = metadata_repo.find_documents_with_any_tag.call_args[0][0]
        assert call_args == {"python", "rust"}

    def test_default_config(self):
        """Should use default config when none provided."""
        db = MagicMock()
        metadata_repo = MagicMock()

        retriever = TagRetriever(db, metadata_repo)

        assert retriever.config.normalize_scores is True
        assert retriever.config.min_score == 0.0
        assert retriever.config.max_results == 100

    def test_custom_config(self):
        """Should use provided config."""
        db = MagicMock()
        metadata_repo = MagicMock()
        config = TagSearchConfig(normalize_scores=False, min_score=0.5)

        retriever = TagRetriever(db, metadata_repo, config)

        assert retriever.config.normalize_scores is False
        assert retriever.config.min_score == 0.5


class TestTagRetrieverScoring:
    """Tests for score calculation."""

    def _setup_retriever(self):
        """Create retriever with mocked dependencies."""
        db = MagicMock()
        metadata_repo = MagicMock()
        return TagRetriever(db, metadata_repo), db, metadata_repo

    def test_weighted_scoring(self):
        """Scores should be weighted sum of matching tags."""
        retriever, db, metadata_repo = self._setup_retriever()

        # Setup mocks
        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python", "ml"}

        # Setup DB to return document
        db.execute.return_value.fetchall.return_value = [{
            "id": 1,
            "path": "doc.md",
            "title": "Test Doc",
            "hash": "abc123",
            "collection_id": 1,
            "modified_at": "2024-01-01",
            "body": "Test content",
        }]

        # Search with weighted tags
        query_tags = {"python": 1.0, "ml": 0.5, "rust": 0.3}
        results = retriever.search(query_tags, limit=10)

        assert len(results) == 1
        # Score = 1.0 (python) + 0.5 (ml) = 1.5
        # Normalized: 1.5 / 1.5 = 1.0
        assert results[0].score == pytest.approx(1.0)

    def test_score_normalization(self):
        """Scores should be normalized to 0-1 range."""
        retriever, db, metadata_repo = self._setup_retriever()

        # Setup mocks - two documents
        metadata_repo.find_documents_with_any_tag.return_value = [1, 2]
        metadata_repo.get_tags.side_effect = [
            {"python", "ml"},  # doc 1: score = 1.5
            {"python"},        # doc 2: score = 1.0
        ]

        db.execute.return_value.fetchall.return_value = [
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "abc", "collection_id": 1, "modified_at": "2024-01-01", "body": "Content 1"},
            {"id": 2, "path": "doc2.md", "title": "Doc 2", "hash": "def", "collection_id": 1, "modified_at": "2024-01-01", "body": "Content 2"},
        ]

        query_tags = {"python": 1.0, "ml": 0.5}
        results = retriever.search(query_tags, limit=10)

        assert len(results) == 2
        # Highest score is 1.5, normalized to 1.0
        assert results[0].score == pytest.approx(1.0)
        # Second score is 1.0 / 1.5 = 0.67
        assert results[1].score == pytest.approx(1.0 / 1.5)

    def test_no_normalization_when_disabled(self):
        """Scores should not be normalized when config disables it."""
        db = MagicMock()
        metadata_repo = MagicMock()
        config = TagSearchConfig(normalize_scores=False)
        retriever = TagRetriever(db, metadata_repo, config)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python", "ml"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Test", "hash": "abc",
            "collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        query_tags = {"python": 1.0, "ml": 0.5}
        results = retriever.search(query_tags, limit=10)

        # Raw score should be preserved
        assert results[0].score == pytest.approx(1.5)

    def test_min_score_filter(self):
        """Results below min_score should be filtered out."""
        db = MagicMock()
        metadata_repo = MagicMock()
        config = TagSearchConfig(normalize_scores=True, min_score=0.5)
        retriever = TagRetriever(db, metadata_repo, config)

        metadata_repo.find_documents_with_any_tag.return_value = [1, 2]
        metadata_repo.get_tags.side_effect = [
            {"python", "ml"},  # doc 1: high score
            {"rust"},          # doc 2: low score (doesn't match python/ml)
        ]

        db.execute.return_value.fetchall.return_value = [
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "abc", "collection_id": 1, "modified_at": "2024-01-01", "body": "Content 1"},
            {"id": 2, "path": "doc2.md", "title": "Doc 2", "hash": "def", "collection_id": 1, "modified_at": "2024-01-01", "body": "Content 2"},
        ]

        # Only python and ml are searched, rust won't contribute
        query_tags = {"python": 1.0, "ml": 0.5}
        results = retriever.search(query_tags, limit=10)

        # Only doc1 should be returned (doc2 has 0 score, filtered)
        assert len(results) == 1
        assert results[0].filepath == "doc1.md"


class TestTagRetrieverCollectionFilter:
    """Tests for collection filtering."""

    def test_collection_filter_applied(self):
        """Collection ID should filter results."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1, 2]
        metadata_repo.get_tags.side_effect = [{"python"}, {"python"}]

        db.execute.return_value.fetchall.return_value = [
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "abc", "collection_id": 1, "modified_at": "2024-01-01", "body": "Content 1"},
        ]  # Only doc from collection 1 returned

        results = retriever.search({"python"}, limit=10, collection_id=1)

        # Check that collection filter was added to SQL
        call_args = db.execute.call_args
        sql = call_args[0][0]
        assert "collection_id = ?" in sql


class TestTagRetrieverResults:
    """Tests for result format."""

    def test_returns_search_result_objects(self):
        """Results should be SearchResult objects with correct fields."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1,
            "path": "docs/python.md",
            "title": "Python Guide",
            "hash": "abc123def456",
            "collection_id": 42,
            "modified_at": "2024-01-15T10:30:00",
            "body": "This is the body content of the document.",
        }]

        results = retriever.search({"python": 1.0}, limit=10)

        assert len(results) == 1
        result = results[0]

        assert result.filepath == "docs/python.md"
        assert result.display_path == "docs/python.md"
        assert result.title == "Python Guide"
        assert result.hash == "abc123def456"
        assert result.collection_id == 42
        assert result.modified_at == "2024-01-15T10:30:00"
        assert result.body == "This is the body content of the document."
        assert result.body_length == len("This is the body content of the document.")
        assert result.score == 1.0

    def test_results_sorted_by_score(self):
        """Results should be sorted by score descending."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1, 2, 3]
        metadata_repo.get_tags.side_effect = [
            {"python"},        # doc 1: score = 1.0
            {"python", "ml"},  # doc 2: score = 1.5
            {"ml"},            # doc 3: score = 0.5
        ]

        db.execute.return_value.fetchall.return_value = [
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "a", "collection_id": 1, "modified_at": "2024-01-01", "body": "A"},
            {"id": 2, "path": "doc2.md", "title": "Doc 2", "hash": "b", "collection_id": 1, "modified_at": "2024-01-01", "body": "B"},
            {"id": 3, "path": "doc3.md", "title": "Doc 3", "hash": "c", "collection_id": 1, "modified_at": "2024-01-01", "body": "C"},
        ]

        query_tags = {"python": 1.0, "ml": 0.5}
        results = retriever.search(query_tags, limit=10)

        assert len(results) == 3
        # Sorted by normalized score: 1.5/1.5=1.0, 1.0/1.5=0.67, 0.5/1.5=0.33
        assert results[0].filepath == "doc2.md"
        assert results[1].filepath == "doc1.md"
        assert results[2].filepath == "doc3.md"

    def test_limit_applied(self):
        """Limit should cap number of results."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        # 5 matching documents
        metadata_repo.find_documents_with_any_tag.return_value = [1, 2, 3, 4, 5]
        metadata_repo.get_tags.side_effect = [{"python"}] * 5

        db.execute.return_value.fetchall.return_value = [
            {"id": i, "path": f"doc{i}.md", "title": f"Doc {i}", "hash": f"h{i}",
             "collection_id": 1, "modified_at": "2024-01-01", "body": "Content"}
            for i in range(1, 6)
        ]

        results = retriever.search({"python"}, limit=3)

        assert len(results) == 3


class TestTagRetrieverFactory:
    """Tests for factory function."""

    def test_create_tag_retriever(self):
        """Factory should create configured retriever."""
        db = MagicMock()
        metadata_repo = MagicMock()

        retriever = create_tag_retriever(db, metadata_repo)

        assert isinstance(retriever, TagRetriever)
        assert retriever.db is db
        assert retriever.metadata_repo is metadata_repo
        assert retriever.config is not None

    def test_create_with_custom_config(self):
        """Factory should accept custom config."""
        db = MagicMock()
        metadata_repo = MagicMock()
        config = TagSearchConfig(min_score=0.3)

        retriever = create_tag_retriever(db, metadata_repo, config)

        assert retriever.config.min_score == 0.3


class TestTagRetrieverEdgeCases:
    """Edge case tests."""

    def test_document_with_no_body(self):
        """Should handle documents with None body."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1,
            "path": "doc.md",
            "title": "Doc",
            "hash": "abc",
            "collection_id": 1,
            "modified_at": "2024-01-01",
            "body": None,
        }]

        results = retriever.search({"python"}, limit=10)

        assert len(results) == 1
        assert results[0].body == ""
        assert results[0].body_length == 0

    def test_document_with_no_matching_tags(self):
        """Documents with tags not in query should have 0 score and be filtered."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"rust"}  # Not in query

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        # Query for python, but doc has rust
        results = retriever.search({"python"}, limit=10)

        # Document should be filtered (score = 0)
        assert len(results) == 0

    def test_empty_title(self):
        """Should handle documents with None title."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": None, "hash": "abc",
            "collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        results = retriever.search({"python"}, limit=10)

        assert len(results) == 1
        assert results[0].title == ""

    def test_single_tag_search(self):
        """Should work with single tag."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        results = retriever.search({"python": 1.0}, limit=10)

        assert len(results) == 1
        assert results[0].score == 1.0
