"""Tests for Reciprocal Rank Fusion algorithm."""

import pytest

from pmd.core.types import RankedResult, SearchResult, SearchSource
from pmd.search.fusion import (
    calculate_reciprocal_rank,
    reciprocal_rank_fusion,
    rrf_formula,
)


def make_search_result(
    filepath: str,
    score: float,
    source: SearchSource = SearchSource.FTS,
    title: str = "Test",
) -> SearchResult:
    """Helper to create SearchResult objects for testing."""
    return SearchResult(
        filepath=filepath,
        display_path=filepath,
        title=title,
        context=None,
        hash="hash_" + filepath,
        source_collection_id=1,
        modified_at="2024-01-01T00:00:00",
        body_length=100,
        body="Test content",
        score=score,
        source=source,
    )


class TestRRFFormula:
    """Tests for rrf_formula function."""

    def test_rank_zero(self):
        """Rank 0 should give highest score."""
        score = rrf_formula(rank=0, k=60, weight=1.0)
        expected = 1.0 / (60 + 0 + 1)
        assert score == pytest.approx(expected)

    def test_increasing_rank_decreases_score(self):
        """Higher ranks should have lower scores."""
        score_0 = rrf_formula(rank=0)
        score_1 = rrf_formula(rank=1)
        score_10 = rrf_formula(rank=10)

        assert score_0 > score_1 > score_10

    def test_weight_multiplies_score(self):
        """Weight should multiply the base score."""
        base_score = rrf_formula(rank=0, weight=1.0)
        weighted_score = rrf_formula(rank=0, weight=2.0)

        assert weighted_score == pytest.approx(base_score * 2.0)

    def test_k_affects_smoothing(self):
        """Larger k should result in smaller scores."""
        score_k60 = rrf_formula(rank=0, k=60)
        score_k100 = rrf_formula(rank=0, k=100)

        assert score_k60 > score_k100

    def test_default_parameters(self):
        """Default parameters should work correctly."""
        score = rrf_formula(rank=5)
        expected = 1.0 / (60 + 5 + 1)
        assert score == pytest.approx(expected)


class TestCalculateReciprocalRank:
    """Tests for calculate_reciprocal_rank function."""

    def test_inverse_of_rrf_formula(self):
        """Should be the inverse of rrf_formula."""
        for rank in [0, 1, 5, 10, 50]:
            score = rrf_formula(rank, k=60)
            recovered_rank = calculate_reciprocal_rank(score, k=60)
            assert recovered_rank == pytest.approx(rank)

    def test_zero_score_returns_infinity(self):
        """Zero score should return infinity."""
        result = calculate_reciprocal_rank(0.0)
        assert result == float("inf")

    def test_negative_score_returns_infinity(self):
        """Negative score should return infinity."""
        result = calculate_reciprocal_rank(-1.0)
        assert result == float("inf")


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion function."""

    def test_empty_result_lists(self):
        """Empty result lists should return empty output."""
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_single_result_list(self):
        """Single result list should return ranked results."""
        results = [
            make_search_result("doc1.md", 0.9),
            make_search_result("doc2.md", 0.8),
        ]

        fused = reciprocal_rank_fusion([results])

        assert len(fused) == 2
        assert all(isinstance(r, RankedResult) for r in fused)

    def test_multiple_result_lists(self):
        """Multiple result lists should be fused."""
        fts_results = [
            make_search_result("doc1.md", 0.9, SearchSource.FTS),
            make_search_result("doc2.md", 0.8, SearchSource.FTS),
        ]
        vec_results = [
            make_search_result("doc2.md", 0.95, SearchSource.VECTOR),
            make_search_result("doc3.md", 0.85, SearchSource.VECTOR),
        ]

        fused = reciprocal_rank_fusion([fts_results, vec_results])

        # Should have unique documents
        assert len(fused) == 3
        filepaths = {r.file for r in fused}
        assert filepaths == {"doc1.md", "doc2.md", "doc3.md"}

    def test_duplicate_documents_accumulate_scores(self):
        """Documents appearing in multiple lists should have accumulated scores."""
        # doc1 appears first in both lists
        list1 = [make_search_result("doc1.md", 0.9)]
        list2 = [make_search_result("doc1.md", 0.85)]

        fused = reciprocal_rank_fusion([list1, list2])

        assert len(fused) == 1
        # Score should be higher than single appearance
        single_list_fused = reciprocal_rank_fusion([list1])
        assert fused[0].score > single_list_fused[0].score

    def test_results_sorted_by_score(self):
        """Results should be sorted by fused score (highest first)."""
        results1 = [
            make_search_result("low.md", 0.5),
            make_search_result("mid.md", 0.7),
        ]
        results2 = [
            make_search_result("high.md", 0.9),
            make_search_result("mid.md", 0.8),
        ]

        fused = reciprocal_rank_fusion([results1, results2])

        # Scores should be in descending order
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_top_rank_bonus_applied(self):
        """Top-ranked items should receive bonus scores."""
        # Two lists with same document at different ranks
        list1 = [make_search_result("first.md", 0.9)]
        list2 = [
            make_search_result("other.md", 0.9),
            make_search_result("first.md", 0.8),
        ]

        fused = reciprocal_rank_fusion([list1, list2])

        # first.md gets rank-0 bonus from list1, rank-1 from list2
        first_result = next(r for r in fused if r.file == "first.md")
        assert first_result.score > 0  # Should have accumulated score

    def test_custom_k_value(self):
        """Custom k value should be used in scoring."""
        results = [make_search_result("doc1.md", 0.9)]

        fused_k60 = reciprocal_rank_fusion([results], k=60)
        fused_k100 = reciprocal_rank_fusion([results], k=100)

        # Larger k means smaller individual scores
        assert fused_k60[0].score > fused_k100[0].score

    def test_original_query_weight(self):
        """Original query results should be weighted higher."""
        results = [make_search_result("doc1.md", 0.9)]

        fused_default = reciprocal_rank_fusion([results], original_query_weight=2.0)
        fused_low = reciprocal_rank_fusion([results], original_query_weight=1.0)

        # Higher weight should give higher score
        assert fused_default[0].score > fused_low[0].score

    def test_custom_weights(self):
        """Custom weights should override default weighting."""
        list1 = [make_search_result("doc1.md", 0.9)]
        list2 = [make_search_result("doc2.md", 0.9)]

        fused = reciprocal_rank_fusion([list1, list2], weights=[3.0, 1.0])

        # doc1 should have higher score due to higher weight
        doc1 = next(r for r in fused if r.file == "doc1.md")
        doc2 = next(r for r in fused if r.file == "doc2.md")
        assert doc1.score > doc2.score

    def test_preserves_document_metadata(self):
        """Fused results should preserve document metadata."""
        result = make_search_result("test.md", 0.9, title="My Title")

        fused = reciprocal_rank_fusion([[result]])

        assert fused[0].file == "test.md"
        assert fused[0].title == "My Title"
        assert fused[0].display_path == "test.md"

    def test_preserves_body_content(self):
        """Fused results should preserve body content."""
        result = make_search_result("test.md", 0.9)

        fused = reciprocal_rank_fusion([[result]])

        assert fused[0].body == "Test content"

    def test_fts_score_preserved(self):
        """FTS source results should have fts_score set."""
        fts_result = make_search_result("doc.md", 0.9, SearchSource.FTS)

        fused = reciprocal_rank_fusion([[fts_result]])

        assert fused[0].fts_score == 0.9

    def test_vec_score_preserved(self):
        """Vector source results should have vec_score set."""
        vec_result = make_search_result("doc.md", 0.9, SearchSource.VECTOR)

        fused = reciprocal_rank_fusion([[vec_result]])

        assert fused[0].vec_score == 0.9

    def test_keeps_highest_individual_score(self):
        """When doc appears in multiple lists, keep highest individual score."""
        fts_result = make_search_result("doc.md", 0.7, SearchSource.FTS)
        vec_result = make_search_result("doc.md", 0.9, SearchSource.VECTOR)

        fused = reciprocal_rank_fusion([[fts_result], [vec_result]])

        # Should keep the vec score since it's higher
        assert fused[0].vec_score == 0.9


class TestReciprocalRankFusionEdgeCases:
    """Edge case tests for reciprocal_rank_fusion."""

    def test_all_empty_lists(self):
        """All empty lists should return empty result."""
        fused = reciprocal_rank_fusion([[], [], []])
        assert fused == []

    def test_one_empty_one_populated(self):
        """Mix of empty and populated lists should work."""
        results = [make_search_result("doc.md", 0.9)]
        fused = reciprocal_rank_fusion([[], results, []])

        assert len(fused) == 1

    def test_large_number_of_lists(self):
        """Should handle many result lists."""
        lists = []
        for i in range(10):
            lists.append([make_search_result(f"doc{i}.md", 0.9)])

        fused = reciprocal_rank_fusion(lists)

        assert len(fused) == 10

    def test_large_number_of_results(self):
        """Should handle many results per list."""
        results = [make_search_result(f"doc{i}.md", 0.9 - i * 0.01) for i in range(100)]

        fused = reciprocal_rank_fusion([results])

        assert len(fused) == 100

    def test_weights_shorter_than_lists(self):
        """Weights shorter than lists should use 1.0 for extra lists."""
        list1 = [make_search_result("doc1.md", 0.9)]
        list2 = [make_search_result("doc2.md", 0.9)]
        list3 = [make_search_result("doc3.md", 0.9)]

        fused = reciprocal_rank_fusion([list1, list2, list3], weights=[2.0])

        # doc1 gets weight 2.0, doc2 and doc3 get default 1.0
        doc1 = next(r for r in fused if r.file == "doc1.md")
        doc2 = next(r for r in fused if r.file == "doc2.md")
        assert doc1.score > doc2.score

    def test_same_document_multiple_positions_same_list(self):
        """Same document shouldn't appear twice in same list (but if it does, handle it)."""
        # This tests defensive behavior
        result1 = make_search_result("doc.md", 0.9)
        result2 = make_search_result("doc.md", 0.8)

        fused = reciprocal_rank_fusion([[result1, result2]])

        # Should only appear once in output
        assert len([r for r in fused if r.file == "doc.md"]) == 1


class TestTagProvenance:
    """Tests for TAG source provenance tracking."""

    def test_tag_score_preserved(self):
        """Tag source results should have tag_score set."""
        tag_result = make_search_result("doc.md", 0.9, SearchSource.TAG)

        fused = reciprocal_rank_fusion([[tag_result]])

        assert fused[0].tag_score == 0.9
        assert fused[0].fts_score is None
        assert fused[0].vec_score is None

    def test_tag_rank_preserved(self):
        """Tag source results should have tag_rank set."""
        tag_results = [
            make_search_result("doc1.md", 0.9, SearchSource.TAG),
            make_search_result("doc2.md", 0.8, SearchSource.TAG),
        ]

        fused = reciprocal_rank_fusion([tag_results])

        doc1 = next(r for r in fused if r.file == "doc1.md")
        doc2 = next(r for r in fused if r.file == "doc2.md")
        assert doc1.tag_rank == 0
        assert doc2.tag_rank == 1

    def test_three_sources_fused(self):
        """FTS, vector, and tag results should all be tracked."""
        fts_result = make_search_result("doc.md", 0.7, SearchSource.FTS)
        vec_result = make_search_result("doc.md", 0.8, SearchSource.VECTOR)
        tag_result = make_search_result("doc.md", 0.9, SearchSource.TAG)

        fused = reciprocal_rank_fusion([[fts_result], [vec_result], [tag_result]])

        assert len(fused) == 1
        result = fused[0]
        assert result.fts_score == 0.7
        assert result.vec_score == 0.8
        assert result.tag_score == 0.9
        assert result.sources_count == 3

    def test_sources_count_with_tag(self):
        """Sources count should include tag source."""
        fts_result = make_search_result("doc.md", 0.9, SearchSource.FTS)
        tag_result = make_search_result("doc.md", 0.85, SearchSource.TAG)

        fused = reciprocal_rank_fusion([[fts_result], [tag_result]])

        assert fused[0].sources_count == 2

    def test_mixed_tag_and_fts(self):
        """Tag and FTS results should be tracked separately."""
        fts_result = make_search_result("doc1.md", 0.9, SearchSource.FTS)
        tag_result = make_search_result("doc2.md", 0.85, SearchSource.TAG)

        fused = reciprocal_rank_fusion([[fts_result], [tag_result]])

        doc1 = next(r for r in fused if r.file == "doc1.md")
        doc2 = next(r for r in fused if r.file == "doc2.md")

        assert doc1.fts_score == 0.9
        assert doc1.tag_score is None
        assert doc2.tag_score == 0.85
        assert doc2.fts_score is None

    def test_tag_only_ranking(self):
        """Tag-only results should rank correctly."""
        tag_results = [
            make_search_result("high.md", 0.95, SearchSource.TAG),
            make_search_result("mid.md", 0.8, SearchSource.TAG),
            make_search_result("low.md", 0.6, SearchSource.TAG),
        ]

        fused = reciprocal_rank_fusion([tag_results])

        # Check order matches ranks
        assert fused[0].file == "high.md"
        assert fused[1].file == "mid.md"
        assert fused[2].file == "low.md"

    def test_keeps_best_tag_rank(self):
        """When doc appears in multiple tag lists, keep best rank."""
        list1 = [
            make_search_result("other.md", 0.9, SearchSource.TAG),
            make_search_result("doc.md", 0.8, SearchSource.TAG),  # rank 1
        ]
        list2 = [
            make_search_result("doc.md", 0.85, SearchSource.TAG),  # rank 0
        ]

        fused = reciprocal_rank_fusion([list1, list2])

        doc = next(r for r in fused if r.file == "doc.md")
        assert doc.tag_rank == 0  # Best rank from list2

    def test_keeps_best_tag_score(self):
        """When doc appears in multiple tag lists, keep best score."""
        list1 = [make_search_result("doc.md", 0.7, SearchSource.TAG)]
        list2 = [make_search_result("doc.md", 0.9, SearchSource.TAG)]

        fused = reciprocal_rank_fusion([list1, list2])

        assert fused[0].tag_score == 0.9  # Best score from list2
