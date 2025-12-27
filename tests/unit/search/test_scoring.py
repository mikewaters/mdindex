"""Tests for score normalization and blending."""

import pytest

from pmd.core.types import RankedResult, RerankDocumentResult
from pmd.search.scoring import (
    blend_scores,
    confidence_score,
    normalize_scores,
    weighted_score,
)


def make_ranked_result(
    filepath: str,
    score: float,
    fts_score: float | None = None,
    vec_score: float | None = None,
    rerank_score: float | None = None,
) -> RankedResult:
    """Helper to create RankedResult objects for testing."""
    return RankedResult(
        file=filepath,
        display_path=filepath,
        title="Test " + filepath,
        body="Test content",
        score=score,
        fts_score=fts_score,
        vec_score=vec_score,
        rerank_score=rerank_score,
    )


def make_rerank_result(filepath: str, score: float) -> RerankDocumentResult:
    """Helper to create RerankDocumentResult objects for testing."""
    return RerankDocumentResult(
        file=filepath,
        relevant=score > 0.5,
        confidence=score,
        score=score,
        raw_token="yes" if score > 0.5 else "no",
    )


class TestNormalizeScores:
    """Tests for normalize_scores function."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        result = normalize_scores([])
        assert result == []

    def test_single_result(self):
        """Single result should be normalized to 1.0."""
        results = [make_ranked_result("doc.md", 0.5)]

        normalized = normalize_scores(results)

        assert len(normalized) == 1
        assert normalized[0].score == 1.0

    def test_multiple_results_normalized(self):
        """Multiple results should be normalized to 0-1 range."""
        results = [
            make_ranked_result("doc1.md", 10.0),
            make_ranked_result("doc2.md", 5.0),
            make_ranked_result("doc3.md", 2.0),
        ]

        normalized = normalize_scores(results)

        assert normalized[0].score == 1.0  # Highest
        assert normalized[1].score == 0.5  # Half of max
        assert normalized[2].score == 0.2  # 2/10

    def test_preserves_relative_order(self):
        """Normalization should preserve relative ordering."""
        results = [
            make_ranked_result("high.md", 0.9),
            make_ranked_result("mid.md", 0.5),
            make_ranked_result("low.md", 0.1),
        ]

        normalized = normalize_scores(results)

        scores = [r.score for r in normalized]
        assert scores[0] > scores[1] > scores[2]

    def test_preserves_metadata(self):
        """Normalization should preserve all metadata."""
        original = make_ranked_result(
            "test.md",
            score=0.8,
            fts_score=0.9,
            vec_score=0.7,
            rerank_score=0.85,
        )

        normalized = normalize_scores([original])

        assert normalized[0].file == "test.md"
        assert normalized[0].title == "Test test.md"
        assert normalized[0].body == "Test content"
        assert normalized[0].fts_score == 0.9
        assert normalized[0].vec_score == 0.7
        assert normalized[0].rerank_score == 0.85

    def test_zero_max_score(self):
        """Zero max score should return original results."""
        results = [
            make_ranked_result("doc1.md", 0.0),
            make_ranked_result("doc2.md", 0.0),
        ]

        normalized = normalize_scores(results)

        # Should return unchanged when max is 0
        assert normalized[0].score == 0.0
        assert normalized[1].score == 0.0

    def test_all_same_score(self):
        """All same scores should normalize to 1.0."""
        results = [
            make_ranked_result("doc1.md", 0.5),
            make_ranked_result("doc2.md", 0.5),
            make_ranked_result("doc3.md", 0.5),
        ]

        normalized = normalize_scores(results)

        for r in normalized:
            assert r.score == 1.0


class TestBlendScores:
    """Tests for blend_scores function."""

    def test_empty_inputs(self):
        """Empty inputs should return empty list."""
        result = blend_scores([], [])
        assert result == []

    def test_no_rerank_results(self):
        """Missing rerank results should use default 0.5."""
        rrf_results = [make_ranked_result("doc.md", 0.8)]

        blended = blend_scores(rrf_results, [])

        assert len(blended) == 1
        # Should use 0.5 as default rerank score
        assert blended[0].rerank_score == 0.5

    def test_top_3_weighting(self):
        """Top 3 results should use 75% RRF + 25% rerank."""
        rrf_results = [
            make_ranked_result("doc1.md", 1.0),
            make_ranked_result("doc2.md", 0.9),
            make_ranked_result("doc3.md", 0.8),
        ]
        rerank_results = [
            make_rerank_result("doc1.md", 0.8),
            make_rerank_result("doc2.md", 0.9),
            make_rerank_result("doc3.md", 0.7),
        ]

        blended = blend_scores(rrf_results, rerank_results)

        # Verify weighting for rank 0 (doc1): 0.75 * 1.0 + 0.25 * 0.8 = 0.95
        assert blended[0].score == pytest.approx(0.75 * 1.0 + 0.25 * 0.8)

    def test_rank_4_to_10_weighting(self):
        """Ranks 4-10 should use 60% RRF + 40% rerank."""
        # Create 10 results to test middle range
        rrf_results = [make_ranked_result(f"doc{i}.md", 1.0 - i * 0.05) for i in range(10)]
        rerank_results = [make_rerank_result(f"doc{i}.md", 0.9) for i in range(10)]

        blended = blend_scores(rrf_results, rerank_results)

        # Check rank 4 (index 3): 60% RRF + 40% rerank
        rrf_score_3 = rrf_results[3].score
        expected_3 = 0.60 * rrf_score_3 + 0.40 * 0.9
        assert blended[3].score == pytest.approx(expected_3, rel=0.1)

    def test_rank_11_plus_weighting(self):
        """Ranks 11+ should use 40% RRF + 60% rerank."""
        # Create 15 results to test tail range
        rrf_results = [make_ranked_result(f"doc{i}.md", 1.0 - i * 0.03) for i in range(15)]
        rerank_results = [make_rerank_result(f"doc{i}.md", 0.8) for i in range(15)]

        blended = blend_scores(rrf_results, rerank_results)

        # Check rank 11 (index 10): 40% RRF + 60% rerank
        rrf_score_10 = rrf_results[10].score
        expected_10 = 0.40 * rrf_score_10 + 0.60 * 0.8
        assert blended[10].score == pytest.approx(expected_10, rel=0.1)

    def test_results_resorted_by_blended_score(self):
        """Results should be re-sorted by blended score."""
        # RRF has doc1 > doc2, but rerank prefers doc2
        rrf_results = [
            make_ranked_result("doc1.md", 1.0),
            make_ranked_result("doc2.md", 0.5),
        ]
        rerank_results = [
            make_rerank_result("doc1.md", 0.1),  # Very low rerank
            make_rerank_result("doc2.md", 1.0),  # Very high rerank
        ]

        blended = blend_scores(rrf_results, rerank_results)

        # After blending, order may change
        scores = [r.score for r in blended]
        assert scores == sorted(scores, reverse=True)

    def test_preserves_metadata(self):
        """Blending should preserve document metadata."""
        rrf_results = [
            RankedResult(
                file="test.md",
                display_path="test.md",
                title="Test Title",
                body="Test body content",
                score=0.8,
                fts_score=0.9,
                vec_score=0.7,
                rerank_score=None,
            )
        ]
        rerank_results = [make_rerank_result("test.md", 0.85)]

        blended = blend_scores(rrf_results, rerank_results)

        assert blended[0].file == "test.md"
        assert blended[0].title == "Test Title"
        assert blended[0].body == "Test body content"
        assert blended[0].fts_score == 0.9
        assert blended[0].vec_score == 0.7
        assert blended[0].rerank_score == 0.85


class TestWeightedScore:
    """Tests for weighted_score function."""

    def test_equal_weights(self):
        """Equal weights should give average."""
        result = weighted_score(0.8, 0.4, fts_weight=1.0, vec_weight=1.0)

        expected = (0.8 + 0.4) / 2.0
        assert result == pytest.approx(expected)

    def test_fts_only_weight(self):
        """FTS-only weight should return fts_score."""
        result = weighted_score(0.8, 0.4, fts_weight=1.0, vec_weight=0.0)

        assert result == pytest.approx(0.8)

    def test_vec_only_weight(self):
        """Vec-only weight should return vec_score."""
        result = weighted_score(0.8, 0.4, fts_weight=0.0, vec_weight=1.0)

        assert result == pytest.approx(0.4)

    def test_unequal_weights(self):
        """Unequal weights should favor higher weight."""
        result = weighted_score(0.8, 0.4, fts_weight=3.0, vec_weight=1.0)

        expected = (0.8 * 3.0 + 0.4 * 1.0) / (3.0 + 1.0)
        assert result == pytest.approx(expected)

    def test_zero_weights(self):
        """Zero weights should return 0."""
        result = weighted_score(0.8, 0.4, fts_weight=0.0, vec_weight=0.0)

        assert result == 0.0

    def test_default_weights(self):
        """Default weights should be equal (1.0 each)."""
        result = weighted_score(0.6, 0.8)

        expected = (0.6 + 0.8) / 2.0
        assert result == pytest.approx(expected)


class TestConfidenceScore:
    """Tests for confidence_score function."""

    def test_identical_scores_boost_confidence(self):
        """Identical scores should boost confidence."""
        result = confidence_score(0.8, 0.8)

        # When scores agree perfectly, boost by agreement * 0.1
        assert result > 0.8

    def test_scores_within_threshold_boost(self):
        """Scores within threshold should boost confidence."""
        result = confidence_score(0.8, 0.7, threshold=0.2)

        # Difference is 0.1, within 0.2 threshold
        assert result > 0.8

    def test_scores_outside_threshold_reduce(self):
        """Scores outside threshold should reduce confidence."""
        result = confidence_score(0.8, 0.3, threshold=0.2)

        # Difference is 0.5, outside 0.2 threshold
        # Confidence reduced: primary * agreement
        assert result < 0.8

    def test_primary_score_dominates(self):
        """Primary score should be the base for confidence."""
        result1 = confidence_score(0.9, 0.5)
        result2 = confidence_score(0.5, 0.9)

        # Higher primary should generally give higher result
        assert result1 > result2

    def test_confidence_capped_at_1(self):
        """Confidence should not exceed 1.0."""
        result = confidence_score(0.95, 0.95)

        assert result <= 1.0

    def test_default_threshold(self):
        """Default threshold should be 0.2."""
        # Difference of 0.15 is within default 0.2 threshold
        result = confidence_score(0.8, 0.65)

        assert result >= 0.8

    def test_zero_scores(self):
        """Zero scores should handle gracefully."""
        result = confidence_score(0.0, 0.0)

        # Should not error; agreement is 1.0
        assert result >= 0.0

    def test_agreement_calculation(self):
        """Agreement should be 1 - abs(difference)."""
        # Difference of 0.3 -> agreement of 0.7
        result = confidence_score(0.8, 0.5, threshold=0.1)

        # Outside threshold, so confidence = primary * agreement
        expected = 0.8 * 0.7
        assert result == pytest.approx(expected)


class TestScoringEdgeCases:
    """Edge case tests for scoring functions."""

    def test_normalize_very_small_scores(self):
        """Should handle very small scores."""
        results = [
            make_ranked_result("doc1.md", 0.0001),
            make_ranked_result("doc2.md", 0.00005),
        ]

        normalized = normalize_scores(results)

        assert normalized[0].score == 1.0
        assert normalized[1].score == 0.5

    def test_normalize_very_large_scores(self):
        """Should handle very large scores."""
        results = [
            make_ranked_result("doc1.md", 10000.0),
            make_ranked_result("doc2.md", 5000.0),
        ]

        normalized = normalize_scores(results)

        assert normalized[0].score == 1.0
        assert normalized[1].score == 0.5

    def test_blend_mismatched_lengths(self):
        """Blend should handle when rerank has fewer results."""
        rrf_results = [
            make_ranked_result("doc1.md", 0.9),
            make_ranked_result("doc2.md", 0.8),
            make_ranked_result("doc3.md", 0.7),
        ]
        rerank_results = [
            make_rerank_result("doc1.md", 0.85),
        ]

        blended = blend_scores(rrf_results, rerank_results)

        # All three should be present
        assert len(blended) == 3
        # doc2 and doc3 should use default 0.5 rerank score
        doc2 = next(r for r in blended if r.file == "doc2.md")
        assert doc2.rerank_score == 0.5

    def test_weighted_score_negative_values(self):
        """Weighted score should handle negative inputs (if they occur)."""
        result = weighted_score(-0.5, 0.5, fts_weight=1.0, vec_weight=1.0)

        expected = (-0.5 + 0.5) / 2.0
        assert result == pytest.approx(expected)
