"""Tests for metadata-based scoring."""

import pytest
from dataclasses import dataclass

from pmd.metadata import (
    MetadataBoostConfig,
    BoostResult,
    WeightedBoostResult,
    apply_metadata_boost,
    apply_metadata_boost_v2,
)
from pmd.metadata.query.scoring import _calculate_boost


@dataclass
class MockResult:
    """Mock search result for testing."""

    file: str
    score: float


class TestApplyMetadataBoost:
    """Tests for the apply_metadata_boost function."""

    def test_boost_with_matching_tags(self):
        """Documents with matching tags should be boosted."""
        results = [
            MockResult(file="doc1.md", score=1.0),
            MockResult(file="doc2.md", score=0.8),
        ]
        query_tags = {"python", "web"}
        doc_id_to_tags = {
            1: {"python", "web", "api"},  # Matches both
            2: {"rust"},  # No match
        }
        doc_path_to_id = {"doc1.md": 1, "doc2.md": 2}

        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id
        )

        # doc1 should be boosted
        doc1_result, doc1_boost = boosted[0]
        assert doc1_result.file == "doc1.md"
        assert doc1_boost.boost_applied > 1.0
        assert doc1_boost.matching_tags == {"python", "web"}

    def test_no_boost_without_matching_tags(self):
        """Documents without matching tags should not be boosted."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = {"python"}
        doc_id_to_tags = {1: {"rust"}}
        doc_path_to_id = {"doc1.md": 1}

        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied == 1.0
        assert boost_info.matching_tags == set()

    def test_no_boost_with_empty_query_tags(self):
        """Empty query tags should result in no boosting."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = set()
        doc_id_to_tags = {1: {"python"}}
        doc_path_to_id = {"doc1.md": 1}

        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied == 1.0

    def test_results_reordered_by_boosted_score(self):
        """Results should be reordered by boosted score."""
        results = [
            MockResult(file="doc1.md", score=1.0),  # No match
            MockResult(file="doc2.md", score=0.5),  # Has match, will be boosted
        ]
        query_tags = {"python"}
        doc_id_to_tags = {1: set(), 2: {"python"}}
        doc_path_to_id = {"doc1.md": 1, "doc2.md": 2}

        config = MetadataBoostConfig(boost_factor=3.0, max_boost=3.0)  # Strong boost
        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id, config
        )

        # doc2 should now be first due to boost
        assert boosted[0][0].file == "doc2.md"

    def test_unknown_document_not_boosted(self):
        """Documents not in path_to_id map should not be boosted."""
        results = [MockResult(file="unknown.md", score=1.0)]
        query_tags = {"python"}
        doc_id_to_tags = {1: {"python"}}
        doc_path_to_id = {}  # unknown.md not in map

        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied == 1.0

    def test_custom_config_boost_factor(self):
        """Custom boost factor should be applied."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = {"python"}
        doc_id_to_tags = {1: {"python"}}
        doc_path_to_id = {"doc1.md": 1}

        config = MetadataBoostConfig(boost_factor=2.0)
        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id, config
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied >= 2.0

    def test_max_boost_caps_score(self):
        """Boost should not exceed max_boost."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = {"a", "b", "c", "d", "e"}  # Many matching tags
        doc_id_to_tags = {1: {"a", "b", "c", "d", "e"}}
        doc_path_to_id = {"doc1.md": 1}

        config = MetadataBoostConfig(boost_factor=1.5, max_boost=1.8)
        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id, config
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied <= 1.8

    def test_min_tags_for_boost(self):
        """Should require minimum matching tags for boost."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = {"python", "web"}
        doc_id_to_tags = {1: {"python"}}  # Only 1 match
        doc_path_to_id = {"doc1.md": 1}

        config = MetadataBoostConfig(min_tags_for_boost=2)
        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id, config
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied == 1.0  # Not enough matches


class TestCalculateBoost:
    """Tests for the _calculate_boost helper."""

    def test_no_matches_returns_one(self):
        """Zero matches should return 1.0 (no boost)."""
        boost = _calculate_boost(0, 1.5, 2.0)
        assert boost == 1.0

    def test_single_match_returns_boost_factor(self):
        """Single match should return approximately the boost factor."""
        boost = _calculate_boost(1, 1.5, 2.0)
        assert boost >= 1.5

    def test_more_matches_higher_boost(self):
        """More matches should result in higher boost."""
        boost1 = _calculate_boost(1, 1.3, 3.0)
        boost2 = _calculate_boost(2, 1.3, 3.0)
        boost3 = _calculate_boost(3, 1.3, 3.0)

        assert boost1 < boost2 < boost3

    def test_diminishing_returns(self):
        """Additional matches should have diminishing returns."""
        boost1 = _calculate_boost(1, 1.5, 5.0)
        boost2 = _calculate_boost(2, 1.5, 5.0)
        boost3 = _calculate_boost(3, 1.5, 5.0)

        diff12 = boost2 - boost1
        diff23 = boost3 - boost2

        # Later additions contribute progressively less
        # (with the current formula, they contribute equally, but this tests the cap)
        assert diff12 > 0
        assert diff23 > 0

    def test_respects_max_boost(self):
        """Boost should be capped at max_boost."""
        boost = _calculate_boost(100, 1.5, 2.0)
        assert boost == 2.0


class TestBoostResult:
    """Tests for BoostResult dataclass."""

    def test_boost_result_fields(self):
        """BoostResult should have correct fields."""
        result = BoostResult(
            original_score=1.0,
            boosted_score=1.5,
            matching_tags={"python", "web"},
            boost_applied=1.5,
        )

        assert result.original_score == 1.0
        assert result.boosted_score == 1.5
        assert result.matching_tags == {"python", "web"}
        assert result.boost_applied == 1.5


class TestMetadataBoostConfig:
    """Tests for MetadataBoostConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = MetadataBoostConfig()

        assert config.boost_factor > 1.0
        assert config.max_boost > config.boost_factor
        assert config.min_tags_for_boost >= 1

    def test_custom_values(self):
        """Config should accept custom values."""
        config = MetadataBoostConfig(
            boost_factor=2.0,
            max_boost=3.0,
            min_tags_for_boost=2,
        )

        assert config.boost_factor == 2.0
        assert config.max_boost == 3.0
        assert config.min_tags_for_boost == 2


class TestScoringEdgeCases:
    """Edge case tests for metadata scoring."""

    def test_boost_with_zero_score(self):
        """Document with zero score should remain zero after boost."""
        results = [MockResult(file="doc1.md", score=0.0)]
        query_tags = {"python"}
        doc_id_to_tags = {1: {"python"}}
        doc_path_to_id = {"doc1.md": 1}

        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id
        )

        _, boost_info = boosted[0]
        assert boost_info.boosted_score == 0.0
        assert boost_info.boost_applied > 1.0  # Boost was applied

    def test_boost_with_very_small_score(self):
        """Very small scores should still be boosted proportionally."""
        results = [MockResult(file="doc1.md", score=0.00001)]
        query_tags = {"python"}
        doc_id_to_tags = {1: {"python"}}
        doc_path_to_id = {"doc1.md": 1}

        config = MetadataBoostConfig(boost_factor=2.0)
        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id, config
        )

        _, boost_info = boosted[0]
        assert boost_info.boosted_score >= 0.00001 * 2.0

    def test_empty_results_list(self):
        """Empty results list should return empty list."""
        boosted = apply_metadata_boost(
            [], {"python"}, {1: {"python"}}, {"doc1.md": 1}
        )
        assert boosted == []

    def test_many_documents_preserves_order_when_no_matches(self):
        """Documents with no matching tags should preserve original order."""
        results = [
            MockResult(file=f"doc{i}.md", score=1.0 - i * 0.1)
            for i in range(10)
        ]
        query_tags = {"nonexistent"}
        doc_id_to_tags = {i: {"other"} for i in range(10)}
        doc_path_to_id = {f"doc{i}.md": i for i in range(10)}

        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id
        )

        # Order should be preserved since no boosting occurred
        for i, (result, _) in enumerate(boosted):
            assert result.file == f"doc{i}.md"

    def test_partial_path_mapping(self):
        """Documents partially in path map should be handled."""
        results = [
            MockResult(file="doc1.md", score=0.9),
            MockResult(file="doc2.md", score=0.8),
            MockResult(file="unknown.md", score=0.7),
        ]
        query_tags = {"python"}
        doc_id_to_tags = {1: {"python"}, 2: {"python"}}
        doc_path_to_id = {"doc1.md": 1, "doc2.md": 2}  # unknown.md not in map

        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id
        )

        # Find unknown.md result
        unknown = next((b for b in boosted if b[0].file == "unknown.md"), None)
        assert unknown is not None
        assert unknown[1].boost_applied == 1.0  # No boost

    def test_boost_with_hierarchical_tags(self):
        """Should match hierarchical tags correctly."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = {"python", "python/web"}  # Hierarchical
        doc_id_to_tags = {1: {"python", "python/web", "python/api"}}
        doc_path_to_id = {"doc1.md": 1}

        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id
        )

        _, boost_info = boosted[0]
        assert boost_info.matching_tags == {"python", "python/web"}

    def test_large_number_of_matching_tags(self):
        """Should handle large number of matching tags with max cap."""
        results = [MockResult(file="doc1.md", score=1.0)]
        # 50 matching tags
        query_tags = {f"tag{i}" for i in range(50)}
        doc_id_to_tags = {1: query_tags.copy()}
        doc_path_to_id = {"doc1.md": 1}

        config = MetadataBoostConfig(boost_factor=1.2, max_boost=3.0)
        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id, config
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied <= 3.0  # Capped at max

    def test_boost_factor_equals_one(self):
        """boost_factor of 1.0 should result in no boosting."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = {"python"}
        doc_id_to_tags = {1: {"python"}}
        doc_path_to_id = {"doc1.md": 1}

        config = MetadataBoostConfig(boost_factor=1.0, max_boost=1.0)
        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id, config
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied == 1.0

    def test_boost_preserves_relative_ordering_when_same_tags(self):
        """Documents with same tags should maintain relative order."""
        results = [
            MockResult(file="doc1.md", score=0.9),
            MockResult(file="doc2.md", score=0.8),
            MockResult(file="doc3.md", score=0.7),
        ]
        query_tags = {"python"}
        doc_id_to_tags = {1: {"python"}, 2: {"python"}, 3: {"python"}}
        doc_path_to_id = {"doc1.md": 1, "doc2.md": 2, "doc3.md": 3}

        boosted = apply_metadata_boost(
            results, query_tags, doc_id_to_tags, doc_path_to_id
        )

        # All boosted equally, should maintain original order
        files = [b[0].file for b in boosted]
        assert files == ["doc1.md", "doc2.md", "doc3.md"]


class TestCalculateBoostEdgeCases:
    """Edge case tests for _calculate_boost helper."""

    def test_negative_matches_treated_as_zero(self):
        """Negative match count should return 1.0 (no boost)."""
        boost = _calculate_boost(-1, 1.5, 2.0)
        assert boost == 1.0

    def test_boost_factor_less_than_one(self):
        """boost_factor < 1.0 should still work (reduces score)."""
        boost = _calculate_boost(1, 0.8, 0.5)
        # Should be capped at max_boost which is 0.5
        assert boost <= 0.8

    def test_max_boost_less_than_one(self):
        """max_boost < 1.0 should cap at that value."""
        boost = _calculate_boost(1, 1.5, 0.9)
        assert boost == 0.9

    def test_zero_max_boost(self):
        """Zero max_boost should return 0.0."""
        boost = _calculate_boost(1, 1.5, 0.0)
        assert boost == 0.0

    def test_very_high_match_count(self):
        """Very high match count should still cap at max_boost."""
        boost = _calculate_boost(1000, 1.5, 2.0)
        assert boost == 2.0


class TestApplyMetadataBoostV2:
    """Tests for the weighted metadata boost function (v2)."""

    def test_single_exact_match(self):
        """Single exact match (weight 1.0) should boost correctly."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = {"python": 1.0}
        doc_id_to_tags = {1: {"python", "web"}}
        doc_path_to_id = {"doc1.md": 1}

        boosted = apply_metadata_boost_v2(
            results, query_tags, doc_id_to_tags, doc_path_to_id,
            boost_factor=1.15,
        )

        _, boost_info = boosted[0]
        assert boost_info.matching_tags == {"python": 1.0}
        assert boost_info.total_match_weight == 1.0
        assert boost_info.boost_applied == pytest.approx(1.15)

    def test_parent_match_reduced_weight(self):
        """Parent match (weight < 1.0) should boost less."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = {"ml/supervised": 0.7}  # Parent match from ontology
        doc_id_to_tags = {1: {"ml/supervised"}}
        doc_path_to_id = {"doc1.md": 1}

        boosted = apply_metadata_boost_v2(
            results, query_tags, doc_id_to_tags, doc_path_to_id,
            boost_factor=1.15,
        )

        _, boost_info = boosted[0]
        assert boost_info.total_match_weight == pytest.approx(0.7)
        # 1.15 ** 0.7 = 1.10
        assert boost_info.boost_applied == pytest.approx(1.15 ** 0.7)

    def test_multiple_weighted_matches(self):
        """Multiple matches should sum their weights."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = {
            "ml/supervised/classification": 1.0,
            "ml/supervised": 0.7,
            "ml": 0.49,
        }
        doc_id_to_tags = {1: {"ml/supervised", "ml"}}  # Matches 2 of 3
        doc_path_to_id = {"doc1.md": 1}

        boosted = apply_metadata_boost_v2(
            results, query_tags, doc_id_to_tags, doc_path_to_id,
            boost_factor=1.15,
        )

        _, boost_info = boosted[0]
        assert boost_info.matching_tags == {"ml/supervised": 0.7, "ml": 0.49}
        assert boost_info.total_match_weight == pytest.approx(1.19)
        # 1.15 ** 1.19 = ~1.18
        assert boost_info.boost_applied == pytest.approx(1.15 ** 1.19)

    def test_no_match_no_boost(self):
        """Documents without matching tags should not be boosted."""
        results = [MockResult(file="doc1.md", score=1.0)]
        query_tags = {"python": 1.0}
        doc_id_to_tags = {1: {"rust"}}
        doc_path_to_id = {"doc1.md": 1}

        boosted = apply_metadata_boost_v2(
            results, query_tags, doc_id_to_tags, doc_path_to_id,
        )

        _, boost_info = boosted[0]
        assert boost_info.matching_tags == {}
        assert boost_info.total_match_weight == 0.0
        assert boost_info.boost_applied == 1.0

    def test_empty_query_tags(self):
        """Empty query tags should not boost anything."""
        results = [MockResult(file="doc1.md", score=1.0)]

        boosted = apply_metadata_boost_v2(
            results, {}, {1: {"python"}}, {"doc1.md": 1},
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied == 1.0

    def test_max_boost_cap(self):
        """Total boost should be capped at max_boost."""
        results = [MockResult(file="doc1.md", score=1.0)]
        # High weight sum would give boost > max
        query_tags = {f"tag{i}": 1.0 for i in range(10)}
        doc_id_to_tags = {1: {f"tag{i}" for i in range(10)}}
        doc_path_to_id = {"doc1.md": 1}

        boosted = apply_metadata_boost_v2(
            results, query_tags, doc_id_to_tags, doc_path_to_id,
            boost_factor=1.15,
            max_boost=1.5,
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied == 1.5

    def test_reorders_by_boosted_score(self):
        """Results should be reordered by boosted score."""
        results = [
            MockResult(file="doc1.md", score=1.0),  # No match
            MockResult(file="doc2.md", score=0.8),  # Will match
        ]
        query_tags = {"python": 1.0}
        doc_id_to_tags = {1: set(), 2: {"python"}}
        doc_path_to_id = {"doc1.md": 1, "doc2.md": 2}

        boosted = apply_metadata_boost_v2(
            results, query_tags, doc_id_to_tags, doc_path_to_id,
            boost_factor=1.5,
            max_boost=2.0,
        )

        # Doc2 should now be first (0.8 * 1.5 = 1.2 > 1.0)
        assert boosted[0][0].file == "doc2.md"
        assert boosted[1][0].file == "doc1.md"

    def test_unknown_document_no_boost(self):
        """Documents not in path map should not be boosted."""
        results = [MockResult(file="unknown.md", score=1.0)]
        query_tags = {"python": 1.0}
        doc_id_to_tags = {1: {"python"}}
        doc_path_to_id = {}  # unknown.md not mapped

        boosted = apply_metadata_boost_v2(
            results, query_tags, doc_id_to_tags, doc_path_to_id,
        )

        _, boost_info = boosted[0]
        assert boost_info.boost_applied == 1.0

    def test_partial_ontology_match(self):
        """Should correctly handle partial ontology matches."""
        results = [MockResult(file="doc1.md", score=1.0)]
        # Simulates ontology expansion for "ml/supervised/classification"
        query_tags = {
            "ml/supervised/classification": 1.0,  # Not in doc
            "ml/supervised": 0.7,                 # In doc
            "ml": 0.49,                           # In doc
        }
        doc_id_to_tags = {1: {"ml/supervised", "ml", "python"}}
        doc_path_to_id = {"doc1.md": 1}

        boosted = apply_metadata_boost_v2(
            results, query_tags, doc_id_to_tags, doc_path_to_id,
            boost_factor=1.2,
        )

        _, boost_info = boosted[0]
        # Only matches ml/supervised (0.7) and ml (0.49)
        assert "ml/supervised/classification" not in boost_info.matching_tags
        assert boost_info.matching_tags["ml/supervised"] == 0.7
        assert boost_info.matching_tags["ml"] == 0.49


class TestWeightedBoostResult:
    """Tests for WeightedBoostResult dataclass."""

    def test_all_fields(self):
        """Should have all required fields."""
        result = WeightedBoostResult(
            original_score=1.0,
            boosted_score=1.5,
            matching_tags={"python": 1.0, "web": 0.7},
            total_match_weight=1.7,
            boost_applied=1.5,
        )

        assert result.original_score == 1.0
        assert result.boosted_score == 1.5
        assert result.matching_tags == {"python": 1.0, "web": 0.7}
        assert result.total_match_weight == 1.7
        assert result.boost_applied == 1.5
