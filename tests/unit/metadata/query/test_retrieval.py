"""Tests for tag-based document retrieval.

This module tests the TagRetriever class and related functionality for
retrieving documents based on tag matches with weighted scoring.
"""

import pytest
from unittest.mock import MagicMock, call

from pmd.metadata import (
    TagRetriever,
    TagSearchConfig,
    create_tag_retriever,
)
from pmd.core.types import SearchResult, SearchSource


class TestTagSearchConfig:
    """Tests for TagSearchConfig dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        config = TagSearchConfig()

        assert config.normalize_scores is True
        assert config.min_score == 0.0
        assert config.max_results == 100

    def test_custom_values(self):
        """Should accept custom values."""
        config = TagSearchConfig(
            normalize_scores=False,
            min_score=0.5,
            max_results=50,
        )

        assert config.normalize_scores is False
        assert config.min_score == 0.5
        assert config.max_results == 50

    def test_partial_custom_values(self):
        """Should allow partial customization with defaults."""
        config = TagSearchConfig(min_score=0.3)

        assert config.normalize_scores is True
        assert config.min_score == 0.3
        assert config.max_results == 100


class TestTagRetrieverBasics:
    """Basic functionality tests for TagRetriever."""

    def test_initialization_with_default_config(self):
        """Should initialize with default config when none provided."""
        db = MagicMock()
        metadata_repo = MagicMock()

        retriever = TagRetriever(db, metadata_repo)

        assert retriever.db is db
        assert retriever.metadata_repo is metadata_repo
        assert retriever.config.normalize_scores is True
        assert retriever.config.min_score == 0.0
        assert retriever.config.max_results == 100

    def test_initialization_with_custom_config(self):
        """Should use provided config."""
        db = MagicMock()
        metadata_repo = MagicMock()
        config = TagSearchConfig(normalize_scores=False, min_score=0.5)

        retriever = TagRetriever(db, metadata_repo, config)

        assert retriever.config is config
        assert retriever.config.normalize_scores is False
        assert retriever.config.min_score == 0.5

    def test_empty_query_tags_returns_empty(self):
        """Empty query tags should return empty results."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        # Test empty dict
        assert retriever.search({}, limit=10) == []

        # Test empty set
        assert retriever.search(set(), limit=10) == []

        # Ensure no calls to dependencies
        metadata_repo.find_documents_with_any_tag.assert_not_called()
        db.execute.assert_not_called()

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


class TestTagRetrieverScoring:
    """Tests for score calculation."""

    def _setup_retriever(self, config=None):
        """Create retriever with mocked dependencies."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo, config)
        return retriever, db, metadata_repo

    def test_weighted_scoring_single_document(self):
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
            "source_collection_id": 1,
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

    def test_weighted_scoring_multiple_documents(self):
        """Should calculate different scores for different documents."""
        retriever, db, metadata_repo = self._setup_retriever()

        # Setup mocks - three documents with different tags
        metadata_repo.find_documents_with_any_tag.return_value = [1, 2, 3]
        metadata_repo.get_tags.side_effect = [
            {"python", "ml"},     # doc 1: score = 1.5
            {"python"},           # doc 2: score = 1.0
            {"ml", "data"},       # doc 3: score = 0.5 + 0.2 = 0.7
        ]

        db.execute.return_value.fetchall.return_value = [
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "abc", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 1"},
            {"id": 2, "path": "doc2.md", "title": "Doc 2", "hash": "def", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 2"},
            {"id": 3, "path": "doc3.md", "title": "Doc 3", "hash": "ghi", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 3"},
        ]

        query_tags = {"python": 1.0, "ml": 0.5, "data": 0.2}
        results = retriever.search(query_tags, limit=10)

        assert len(results) == 3
        # Results should be sorted by score (highest first)
        # doc1: 1.5/1.5 = 1.0, doc2: 1.0/1.5 = 0.67, doc3: 0.7/1.5 = 0.47
        assert results[0].filepath == "doc1.md"
        assert results[1].filepath == "doc2.md"
        assert results[2].filepath == "doc3.md"

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
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "abc", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 1"},
            {"id": 2, "path": "doc2.md", "title": "Doc 2", "hash": "def", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 2"},
        ]

        query_tags = {"python": 1.0, "ml": 0.5}
        results = retriever.search(query_tags, limit=10)

        assert len(results) == 2
        # Highest score is 1.5, normalized to 1.0
        assert results[0].score == pytest.approx(1.0)
        # Second score is 1.0 / 1.5 = 0.6667
        assert results[1].score == pytest.approx(1.0 / 1.5)

    def test_no_normalization_when_disabled(self):
        """Scores should not be normalized when config disables it."""
        config = TagSearchConfig(normalize_scores=False)
        retriever, db, metadata_repo = self._setup_retriever(config)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python", "ml"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Test", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        query_tags = {"python": 1.0, "ml": 0.5}
        results = retriever.search(query_tags, limit=10)

        # Raw score should be preserved
        assert results[0].score == pytest.approx(1.5)

    def test_min_score_filter(self):
        """Results below min_score should be filtered out."""
        config = TagSearchConfig(normalize_scores=True, min_score=0.5)
        retriever, db, metadata_repo = self._setup_retriever(config)

        metadata_repo.find_documents_with_any_tag.return_value = [1, 2, 3]
        metadata_repo.get_tags.side_effect = [
            {"python", "ml"},  # doc 1: high score (1.5 -> 1.0 normalized)
            {"python"},        # doc 2: medium score (1.0 -> 0.67 normalized)
            {"web"},           # doc 3: low score (0.3 -> 0.2 normalized)
        ]

        db.execute.return_value.fetchall.return_value = [
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "abc", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 1"},
            {"id": 2, "path": "doc2.md", "title": "Doc 2", "hash": "def", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 2"},
            {"id": 3, "path": "doc3.md", "title": "Doc 3", "hash": "ghi", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 3"},
        ]

        query_tags = {"python": 1.0, "ml": 0.5, "web": 0.3}
        results = retriever.search(query_tags, limit=10)

        # Only docs with score >= 0.5 should be returned
        assert len(results) == 2
        assert results[0].filepath == "doc1.md"
        assert results[1].filepath == "doc2.md"

    def test_zero_score_documents_filtered(self):
        """Documents with no matching tags should be filtered out."""
        retriever, db, metadata_repo = self._setup_retriever()

        metadata_repo.find_documents_with_any_tag.return_value = [1, 2]
        metadata_repo.get_tags.side_effect = [
            {"python"},  # doc 1: matches
            {"rust"},    # doc 2: doesn't match query
        ]

        db.execute.return_value.fetchall.return_value = [
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "abc", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 1"},
            {"id": 2, "path": "doc2.md", "title": "Doc 2", "hash": "def", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 2"},
        ]

        # Only query for python (doc 2 has rust)
        query_tags = {"python": 1.0}
        results = retriever.search(query_tags, limit=10)

        # Only doc1 should be returned
        assert len(results) == 1
        assert results[0].filepath == "doc1.md"


class TestTagRetrieverCollectionFilter:
    """Tests for collection filtering."""

    def test_collection_filter_applied_to_query(self):
        """Collection ID should be passed to SQL query."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        db.execute.return_value.fetchall.return_value = [
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "abc", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 1"},
        ]

        results = retriever.search({"python"}, limit=10, source_collection_id=1)

        # Check that collection filter was added to SQL
        call_args = db.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]

        assert "source_collection_id = ?" in sql
        assert 1 in params

    def test_no_collection_filter_when_none(self):
        """Collection filter should not be applied when None."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        db.execute.return_value.fetchall.return_value = [
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "abc", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 1"},
        ]

        results = retriever.search({"python"}, limit=10, source_collection_id=None)

        # Check that collection filter was NOT added to SQL
        call_args = db.execute.call_args
        sql = call_args[0][0]

        # Should not have the filter clause
        assert "source_collection_id = ?" not in sql


class TestTagRetrieverResults:
    """Tests for result format and content."""

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
            "source_collection_id": 42,
            "modified_at": "2024-01-15T10:30:00",
            "body": "This is the body content of the document.",
        }]

        results = retriever.search({"python": 1.0}, limit=10)

        assert len(results) == 1
        result = results[0]

        assert isinstance(result, SearchResult)
        assert result.filepath == "docs/python.md"
        assert result.display_path == "docs/python.md"
        assert result.title == "Python Guide"
        assert result.hash == "abc123def456"
        assert result.source_collection_id == 42
        assert result.modified_at == "2024-01-15T10:30:00"
        assert result.body == "This is the body content of the document."
        assert result.body_length == len("This is the body content of the document.")
        assert result.score == 1.0
        assert result.source == SearchSource.TAG
        assert result.chunk_pos is None

    def test_result_context_and_snippet(self):
        """Context and snippet should be first 200 chars of body."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        long_body = "a" * 500
        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01",
            "body": long_body,
        }]

        results = retriever.search({"python"}, limit=10)

        assert len(results) == 1
        assert results[0].context == long_body[:200]
        assert results[0].snippet == long_body[:200]

    def test_results_sorted_by_score_descending(self):
        """Results should be sorted by score from highest to lowest."""
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
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "a", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "A"},
            {"id": 2, "path": "doc2.md", "title": "Doc 2", "hash": "b", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "B"},
            {"id": 3, "path": "doc3.md", "title": "Doc 3", "hash": "c", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "C"},
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
             "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content"}
            for i in range(1, 6)
        ]

        results = retriever.search({"python"}, limit=3)

        assert len(results) == 3

    def test_limit_larger_than_results(self):
        """Limit larger than results should return all results."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1, 2]
        metadata_repo.get_tags.side_effect = [{"python"}] * 2

        db.execute.return_value.fetchall.return_value = [
            {"id": i, "path": f"doc{i}.md", "title": f"Doc {i}", "hash": f"h{i}",
             "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content"}
            for i in range(1, 3)
        ]

        results = retriever.search({"python"}, limit=100)

        assert len(results) == 2


class TestTagRetrieverEdgeCases:
    """Edge case tests."""

    def test_document_with_none_body(self):
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
            "source_collection_id": 1,
            "modified_at": "2024-01-01",
            "body": None,
        }]

        results = retriever.search({"python"}, limit=10)

        assert len(results) == 1
        assert results[0].body == ""
        assert results[0].body_length == 0
        assert results[0].context is None
        assert results[0].snippet is None

    def test_document_with_empty_body(self):
        """Should handle documents with empty string body."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01", "body": "",
        }]

        results = retriever.search({"python"}, limit=10)

        assert len(results) == 1
        assert results[0].body == ""
        assert results[0].body_length == 0

    def test_document_with_none_title(self):
        """Should handle documents with None title."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": None, "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        results = retriever.search({"python"}, limit=10)

        assert len(results) == 1
        assert results[0].title == ""

    def test_single_tag_search_with_dict(self):
        """Should work with single tag as dict."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        results = retriever.search({"python": 1.0}, limit=10)

        assert len(results) == 1
        assert results[0].score == 1.0

    def test_single_tag_search_with_set(self):
        """Should work with single tag as set."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        results = retriever.search({"python"}, limit=10)

        assert len(results) == 1
        assert results[0].score == 1.0

    def test_document_missing_from_tags_map(self):
        """Should handle documents missing from get_tags response."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        # get_tags returns empty set
        metadata_repo.get_tags.return_value = set()

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        results = retriever.search({"python"}, limit=10)

        # Document has no tags, so score is 0 and it's filtered
        assert len(results) == 0

    def test_large_number_of_matching_tags(self):
        """Should handle documents with many matching tags."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        # Document has many tags
        many_tags = {f"tag{i}" for i in range(100)}
        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = many_tags

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        # Query with many tags, all weight 1.0
        query_tags = {f"tag{i}": 1.0 for i in range(100)}
        results = retriever.search(query_tags, limit=10)

        assert len(results) == 1
        # Raw score would be 100.0, normalized to 1.0
        assert results[0].score == 1.0

    def test_fractional_weights(self):
        """Should handle fractional weights correctly."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python", "ml", "web"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        # Use fractional weights
        query_tags = {"python": 0.7, "ml": 0.3, "web": 0.15}
        results = retriever.search(query_tags, limit=10)

        assert len(results) == 1
        # Raw score = 0.7 + 0.3 + 0.15 = 1.15, normalized to 1.0
        assert results[0].score == 1.0

    def test_zero_weight_tags(self):
        """Should handle zero weight tags (should contribute nothing)."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python", "ml"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        query_tags = {"python": 1.0, "ml": 0.0}
        results = retriever.search(query_tags, limit=10)

        assert len(results) == 1
        # Only python counts
        assert results[0].score == 1.0

    def test_negative_weight_tags(self):
        """Should handle negative weights (unusual but valid)."""
        config = TagSearchConfig(normalize_scores=False)
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo, config)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python", "ml"}

        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content",
        }]

        query_tags = {"python": 1.0, "ml": -0.3}
        results = retriever.search(query_tags, limit=10)

        # Score = 1.0 - 0.3 = 0.7
        # But since score > 0, document is included
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.7)

    def test_very_short_body(self):
        """Should handle body shorter than 200 chars."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1]
        metadata_repo.get_tags.return_value = {"python"}

        short_body = "Short content."
        db.execute.return_value.fetchall.return_value = [{
            "id": 1, "path": "doc.md", "title": "Doc", "hash": "abc",
            "source_collection_id": 1, "modified_at": "2024-01-01",
            "body": short_body,
        }]

        results = retriever.search({"python"}, limit=10)

        assert len(results) == 1
        assert results[0].context == short_body
        assert results[0].snippet == short_body


class TestTagRetrieverFactory:
    """Tests for factory function."""

    def test_create_tag_retriever_default_config(self):
        """Factory should create retriever with default config."""
        db = MagicMock()
        metadata_repo = MagicMock()

        retriever = create_tag_retriever(db, metadata_repo)

        assert isinstance(retriever, TagRetriever)
        assert retriever.db is db
        assert retriever.metadata_repo is metadata_repo
        assert retriever.config is not None
        assert retriever.config.normalize_scores is True

    def test_create_tag_retriever_custom_config(self):
        """Factory should accept custom config."""
        db = MagicMock()
        metadata_repo = MagicMock()
        config = TagSearchConfig(min_score=0.3, max_results=50)

        retriever = create_tag_retriever(db, metadata_repo, config)

        assert isinstance(retriever, TagRetriever)
        assert retriever.config is config
        assert retriever.config.min_score == 0.3
        assert retriever.config.max_results == 50

    def test_create_tag_retriever_none_config(self):
        """Factory should accept None config and use default."""
        db = MagicMock()
        metadata_repo = MagicMock()

        retriever = create_tag_retriever(db, metadata_repo, None)

        assert isinstance(retriever, TagRetriever)
        assert retriever.config is not None
        assert retriever.config.normalize_scores is True


class TestTagRetrieverDatabaseInteraction:
    """Tests for database query construction."""

    def test_sql_query_structure(self):
        """SQL query should have correct structure."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1, 2, 3]
        metadata_repo.get_tags.side_effect = [{"python"}] * 3

        db.execute.return_value.fetchall.return_value = [
            {"id": i, "path": f"doc{i}.md", "title": f"Doc {i}", "hash": f"h{i}",
             "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content"}
            for i in range(1, 4)
        ]

        retriever.search({"python"}, limit=10)

        # Check SQL structure
        call_args = db.execute.call_args
        sql = call_args[0][0]

        assert "SELECT" in sql
        assert "FROM documents d" in sql
        assert "JOIN content c ON d.hash = c.hash" in sql
        assert "WHERE d.id IN" in sql
        assert "d.active = 1" in sql

    def test_sql_parameters_match_doc_ids(self):
        """SQL parameters should include all doc IDs."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1, 2, 3]
        metadata_repo.get_tags.side_effect = [{"python"}] * 3

        db.execute.return_value.fetchall.return_value = [
            {"id": i, "path": f"doc{i}.md", "title": f"Doc {i}", "hash": f"h{i}",
             "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content"}
            for i in range(1, 4)
        ]

        retriever.search({"python"}, limit=10)

        call_args = db.execute.call_args
        params = call_args[0][1]

        # Should have 3 placeholders for doc IDs
        assert len(params) == 3
        assert 1 in params
        assert 2 in params
        assert 3 in params

    def test_get_tags_called_for_each_doc(self):
        """get_tags should be called once per matching document."""
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo)

        metadata_repo.find_documents_with_any_tag.return_value = [1, 2, 3]
        metadata_repo.get_tags.side_effect = [{"python"}] * 3

        db.execute.return_value.fetchall.return_value = [
            {"id": i, "path": f"doc{i}.md", "title": f"Doc {i}", "hash": f"h{i}",
             "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content"}
            for i in range(1, 4)
        ]

        retriever.search({"python"}, limit=10)

        # get_tags should be called 3 times
        assert metadata_repo.get_tags.call_count == 3
        metadata_repo.get_tags.assert_any_call(1)
        metadata_repo.get_tags.assert_any_call(2)
        metadata_repo.get_tags.assert_any_call(3)


class TestTagRetrieverMinScoreInteraction:
    """Tests for interaction between min_score and other features."""

    def test_min_score_with_normalization(self):
        """Min score should be applied after normalization."""
        config = TagSearchConfig(normalize_scores=True, min_score=0.6)
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo, config)

        metadata_repo.find_documents_with_any_tag.return_value = [1, 2]
        metadata_repo.get_tags.side_effect = [
            {"python", "ml"},  # doc 1: score 1.5 -> normalized 1.0
            {"python"},        # doc 2: score 1.0 -> normalized 0.67
        ]

        db.execute.return_value.fetchall.return_value = [
            {"id": 1, "path": "doc1.md", "title": "Doc 1", "hash": "abc", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 1"},
            {"id": 2, "path": "doc2.md", "title": "Doc 2", "hash": "def", "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content 2"},
        ]

        query_tags = {"python": 1.0, "ml": 0.5}
        results = retriever.search(query_tags, limit=10)

        # Only doc1 passes (1.0 >= 0.6), doc2 filtered (0.67 >= 0.6 but < 0.6)
        # Actually 0.67 > 0.6, so both should pass
        assert len(results) == 2

    def test_min_score_filters_before_limit(self):
        """Min score should filter before limit is applied."""
        config = TagSearchConfig(min_score=0.8)
        db = MagicMock()
        metadata_repo = MagicMock()
        retriever = TagRetriever(db, metadata_repo, config)

        # Create 5 docs with varying scores
        metadata_repo.find_documents_with_any_tag.return_value = [1, 2, 3, 4, 5]
        metadata_repo.get_tags.side_effect = [
            {"python", "ml"},     # score 1.5 -> 1.0 normalized
            {"python"},           # score 1.0 -> 0.67 normalized
            {"ml"},               # score 0.5 -> 0.33 normalized
            {"python", "web"},    # score 1.3 -> 0.87 normalized
            {"web"},              # score 0.3 -> 0.2 normalized
        ]

        db.execute.return_value.fetchall.return_value = [
            {"id": i, "path": f"doc{i}.md", "title": f"Doc {i}", "hash": f"h{i}",
             "source_collection_id": 1, "modified_at": "2024-01-01", "body": "Content"}
            for i in range(1, 6)
        ]

        query_tags = {"python": 1.0, "ml": 0.5, "web": 0.3}
        results = retriever.search(query_tags, limit=3)

        # Only docs with score >= 0.8 after normalization: doc1 (1.0), doc4 (0.87)
        assert len(results) == 2
        assert results[0].filepath == "doc1.md"
        assert results[1].filepath == "doc4.md"
