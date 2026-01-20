"""Tests for idx.search.hybrid module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idx.search.hybrid import HybridSearch, rrf_fusion, RRF_K
from idx.search.models import SearchCriteria, SearchResult, SearchResults


class TestRrfFusion:
    """Tests for rrf_fusion function."""

    def test_basic_fusion_two_lists(self) -> None:
        """Basic RRF fusion of two ranked lists."""
        list1 = [
            ("a.md", "vault", 0.9),
            ("b.md", "vault", 0.8),
            ("c.md", "vault", 0.7),
        ]
        list2 = [
            ("b.md", "vault", 0.95),
            ("c.md", "vault", 0.85),
            ("d.md", "vault", 0.75),
        ]

        fused = rrf_fusion([list1, list2])

        # Should have 4 unique items
        assert len(fused) == 4

        # b.md and c.md appear in both lists, should have highest RRF scores
        paths = [path for path, ds, score in fused]
        # Items in both lists should be ranked higher
        assert paths[0] in ("b.md", "c.md")
        assert paths[1] in ("b.md", "c.md")

    def test_items_in_both_lists_get_higher_scores(self) -> None:
        """Items appearing in both lists accumulate RRF scores."""
        list1 = [("shared.md", "vault", 0.9)]
        list2 = [("shared.md", "vault", 0.8)]

        fused = rrf_fusion([list1, list2])

        # Single item appears in both, gets score from both
        assert len(fused) == 1
        path, ds, score = fused[0]
        assert path == "shared.md"

        # Score should be sum of 1/(k+1) from each list
        expected_score = 2 * (1.0 / (RRF_K + 1))
        assert abs(score - expected_score) < 0.0001

    def test_items_in_one_list_still_included(self) -> None:
        """Items appearing in only one list are included."""
        list1 = [("only1.md", "vault", 0.9)]
        list2 = [("only2.md", "vault", 0.8)]

        fused = rrf_fusion([list1, list2])

        assert len(fused) == 2
        paths = {path for path, ds, score in fused}
        assert paths == {"only1.md", "only2.md"}

    def test_empty_lists_handling(self) -> None:
        """Empty lists are handled gracefully."""
        # All empty
        assert rrf_fusion([]) == []
        assert rrf_fusion([[]]) == []
        assert rrf_fusion([[], []]) == []

        # One empty, one with items
        list1 = [("a.md", "vault", 0.9)]
        fused = rrf_fusion([list1, []])
        assert len(fused) == 1
        assert fused[0][0] == "a.md"

    def test_custom_k_value(self) -> None:
        """Custom k value affects score calculation."""
        items = [("a.md", "vault", 0.9)]

        # With default k=60
        fused_default = rrf_fusion([items])
        _, _, score_default = fused_default[0]

        # With k=10
        fused_k10 = rrf_fusion([items], k=10)
        _, _, score_k10 = fused_k10[0]

        # Lower k should give higher score for rank 1
        assert score_k10 > score_default
        assert abs(score_k10 - 1.0 / 11) < 0.0001  # 1/(10+1)
        assert abs(score_default - 1.0 / 61) < 0.0001  # 1/(60+1)

    def test_different_datasets_kept_separate(self) -> None:
        """Same path in different datasets are separate items."""
        list1 = [("readme.md", "vault1", 0.9)]
        list2 = [("readme.md", "vault2", 0.8)]

        fused = rrf_fusion([list1, list2])

        assert len(fused) == 2
        keys = {(path, ds) for path, ds, score in fused}
        assert keys == {("readme.md", "vault1"), ("readme.md", "vault2")}

    def test_ranking_preserved_within_list(self) -> None:
        """Items ranked higher in lists get better RRF contribution."""
        # Same items but different rankings
        list1 = [
            ("a.md", "vault", 0.9),  # rank 1
            ("b.md", "vault", 0.8),  # rank 2
        ]
        list2 = [
            ("a.md", "vault", 0.7),  # rank 1
            ("b.md", "vault", 0.6),  # rank 2
        ]

        fused = rrf_fusion([list1, list2])

        # a.md is rank 1 in both lists, b.md is rank 2 in both
        # So a.md should have higher RRF score
        scores = {path: score for path, ds, score in fused}
        assert scores["a.md"] > scores["b.md"]

    def test_three_lists_fusion(self) -> None:
        """RRF works with more than two lists."""
        list1 = [("a.md", "vault", 0.9), ("b.md", "vault", 0.8)]
        list2 = [("b.md", "vault", 0.9), ("c.md", "vault", 0.8)]
        list3 = [("c.md", "vault", 0.9), ("a.md", "vault", 0.8)]

        fused = rrf_fusion([list1, list2, list3])

        # All three items appear in two lists each
        assert len(fused) == 3
        # Scores should be equal since each appears in 2 lists with same rank distribution
        scores = [score for path, ds, score in fused]
        assert abs(scores[0] - scores[1]) < 0.0001
        assert abs(scores[1] - scores[2]) < 0.0001


class TestHybridSearch:
    """Tests for HybridSearch class."""

    @pytest.fixture
    def mock_fts_results(self) -> SearchResults:
        """Mock FTS search results."""
        return SearchResults(
            results=[
                SearchResult(path="doc1.md", dataset_name="vault", score=0.9, scores={"fts": 0.9}),
                SearchResult(path="doc2.md", dataset_name="vault", score=0.8, scores={"fts": 0.8}),
                SearchResult(path="doc3.md", dataset_name="vault", score=0.7, scores={"fts": 0.7}),
            ],
            query="test",
            mode="fts",
            total_candidates=3,
            timing_ms=10.0,
        )

    @pytest.fixture
    def mock_vector_results(self) -> list[tuple[str, str, float]]:
        """Mock vector search results."""
        return [
            ("doc2.md", "vault", 0.95),
            ("doc4.md", "vault", 0.85),
            ("doc1.md", "vault", 0.75),
        ]

    @pytest.fixture
    def hybrid_search(self, tmp_path: Path) -> HybridSearch:
        """Create HybridSearch with mocked dependencies."""
        mock_session = MagicMock()
        mock_vector = MagicMock()

        with patch("idx.search.hybrid.FTSSearch"):
            search = HybridSearch(
                session=mock_session,
                vector_search=mock_vector,
            )
            return search

    def test_search_combines_fts_and_vector(
        self,
        hybrid_search: HybridSearch,
        mock_fts_results: SearchResults,
        mock_vector_results: list[tuple[str, str, float]],
    ) -> None:
        """Search combines results from both FTS and vector."""
        hybrid_search._fts.search.return_value = mock_fts_results
        hybrid_search._vector.search_with_scores.return_value = mock_vector_results

        criteria = SearchCriteria(query="test", mode="hybrid", limit=10)
        results = hybrid_search.search(criteria)

        # Should call both search methods
        hybrid_search._fts.search.assert_called_once()
        hybrid_search._vector.search_with_scores.assert_called_once()

        # Should have results from both sources
        assert len(results.results) > 0
        paths = {r.path for r in results.results}
        # doc1, doc2, doc3 from FTS, doc4 only from vector
        assert "doc1.md" in paths
        assert "doc2.md" in paths
        assert "doc4.md" in paths

    def test_search_normalizes_scores(
        self,
        hybrid_search: HybridSearch,
        mock_fts_results: SearchResults,
        mock_vector_results: list[tuple[str, str, float]],
    ) -> None:
        """RRF scores are normalized to 0-1 range."""
        hybrid_search._fts.search.return_value = mock_fts_results
        hybrid_search._vector.search_with_scores.return_value = mock_vector_results

        criteria = SearchCriteria(query="test", mode="hybrid", limit=10)
        results = hybrid_search.search(criteria)

        # Top result should have score of 1.0 after normalization
        assert results.results[0].score == 1.0

        # All scores should be between 0 and 1
        for result in results.results:
            assert 0.0 <= result.score <= 1.0

    def test_search_deduplicates_by_path(
        self,
        hybrid_search: HybridSearch,
    ) -> None:
        """Results are deduplicated by (dataset_name, path)."""
        # Both sources return same doc
        fts_results = SearchResults(
            results=[
                SearchResult(path="same.md", dataset_name="vault", score=0.9, scores={"fts": 0.9}),
            ],
            query="test",
            mode="fts",
            total_candidates=1,
        )
        vec_results = [("same.md", "vault", 0.8)]

        hybrid_search._fts.search.return_value = fts_results
        hybrid_search._vector.search_with_scores.return_value = vec_results

        criteria = SearchCriteria(query="test", mode="hybrid", limit=10)
        results = hybrid_search.search(criteria)

        # Should only have one result for same.md
        assert len(results.results) == 1
        assert results.results[0].path == "same.md"

        # Should have higher score since it appeared in both lists
        assert results.results[0].score == 1.0

    def test_search_respects_limit(
        self,
        hybrid_search: HybridSearch,
        mock_fts_results: SearchResults,
        mock_vector_results: list[tuple[str, str, float]],
    ) -> None:
        """Search respects the limit parameter."""
        hybrid_search._fts.search.return_value = mock_fts_results
        hybrid_search._vector.search_with_scores.return_value = mock_vector_results

        criteria = SearchCriteria(query="test", mode="hybrid", limit=2)
        results = hybrid_search.search(criteria)

        assert len(results.results) <= 2

    def test_search_includes_component_scores(
        self,
        hybrid_search: HybridSearch,
        mock_fts_results: SearchResults,
        mock_vector_results: list[tuple[str, str, float]],
    ) -> None:
        """Results include component scores from each source."""
        hybrid_search._fts.search.return_value = mock_fts_results
        hybrid_search._vector.search_with_scores.return_value = mock_vector_results

        criteria = SearchCriteria(query="test", mode="hybrid", limit=10)
        results = hybrid_search.search(criteria)

        # Find doc2 which appears in both
        doc2 = next(r for r in results.results if r.path == "doc2.md")
        assert "rrf" in doc2.scores
        assert "fts" in doc2.scores
        assert "vector" in doc2.scores

        # Find doc3 which appears only in FTS
        doc3 = next((r for r in results.results if r.path == "doc3.md"), None)
        if doc3:
            assert "fts" in doc3.scores
            assert "vector" not in doc3.scores

    def test_search_returns_metadata(
        self,
        hybrid_search: HybridSearch,
        mock_fts_results: SearchResults,
        mock_vector_results: list[tuple[str, str, float]],
    ) -> None:
        """Search returns proper metadata."""
        hybrid_search._fts.search.return_value = mock_fts_results
        hybrid_search._vector.search_with_scores.return_value = mock_vector_results

        criteria = SearchCriteria(query="test query", mode="hybrid", limit=10)
        results = hybrid_search.search(criteria)

        assert results.query == "test query"
        assert results.mode == "hybrid"
        assert results.total_candidates > 0
        assert results.timing_ms is not None

    def test_search_empty_results(
        self,
        hybrid_search: HybridSearch,
    ) -> None:
        """Search handles empty results from both sources."""
        empty_fts = SearchResults(
            results=[],
            query="test",
            mode="fts",
            total_candidates=0,
        )
        hybrid_search._fts.search.return_value = empty_fts
        hybrid_search._vector.search_with_scores.return_value = []

        criteria = SearchCriteria(query="test", mode="hybrid", limit=10)
        results = hybrid_search.search(criteria)

        assert len(results.results) == 0
        assert results.mode == "hybrid"

    def test_search_passes_dataset_filter(
        self,
        hybrid_search: HybridSearch,
        mock_fts_results: SearchResults,
        mock_vector_results: list[tuple[str, str, float]],
    ) -> None:
        """Dataset filter is passed to both search methods."""
        hybrid_search._fts.search.return_value = mock_fts_results
        hybrid_search._vector.search_with_scores.return_value = mock_vector_results

        criteria = SearchCriteria(
            query="test",
            mode="hybrid",
            dataset_name="my-vault",
            limit=10,
        )
        hybrid_search.search(criteria)

        # Check FTS was called with dataset filter
        fts_call = hybrid_search._fts.search.call_args
        assert fts_call[0][0].dataset_name == "my-vault"

        # Check vector was called with dataset filter
        vec_call = hybrid_search._vector.search_with_scores.call_args
        assert vec_call[1]["dataset_name"] == "my-vault"
