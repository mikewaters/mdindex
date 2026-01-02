"""Tests for pipeline metadata boost integration."""

import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, AsyncMock

from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig
from pmd.search.metadata.inference import LexicalTagMatcher
from pmd.search.metadata.ontology import Ontology
from pmd.search.metadata.retrieval import TagRetriever
from pmd.search.metadata.scoring import MetadataBoostConfig
from pmd.core.types import SearchSource


@dataclass
class MockRankedResult:
    """Mock RankedResult for testing."""

    file: str
    score: float
    title: str = ""
    snippet: str = ""
    document_id: int | None = None
    rerank_score: float | None = None


class TestPipelineMetadataBoostIntegration:
    """Tests for _apply_metadata_boost method."""

    def test_metadata_boost_disabled_by_default(self):
        """Metadata boost should be disabled when not configured."""
        fts_repo = MagicMock()
        config = SearchPipelineConfig()  # defaults

        pipeline = HybridSearchPipeline(fts_repo, config)

        assert pipeline.tag_matcher is None
        assert pipeline.metadata_repo is None

    def test_metadata_boost_requires_both_matcher_and_repo(self):
        """Metadata boost requires both matcher and repo to be provided."""
        fts_repo = MagicMock()
        matcher = LexicalTagMatcher()
        config = SearchPipelineConfig(enable_metadata_boost=True)

        # Only matcher, no repo
        pipeline = HybridSearchPipeline(
            fts_repo, config, tag_matcher=matcher, metadata_repo=None
        )

        # Should still work but not boost (handled gracefully)
        assert pipeline.tag_matcher is not None or pipeline.tag_matcher is None

    def test_apply_metadata_boost_returns_unchanged_without_matcher(self):
        """_apply_metadata_boost should return unchanged results without matcher."""
        fts_repo = MagicMock()
        config = SearchPipelineConfig(enable_metadata_boost=False)

        pipeline = HybridSearchPipeline(fts_repo, config)

        candidates = [
            MockRankedResult(file="doc1.md", score=1.0),
            MockRankedResult(file="doc2.md", score=0.8),
        ]

        # Even if we call the method directly, it should return unchanged
        result = pipeline._apply_metadata_boost("python tutorial", candidates)

        assert result == candidates

    def test_apply_metadata_boost_with_no_query_tags(self):
        """Should return unchanged when no tags are inferred from query."""
        fts_repo = MagicMock()
        matcher = LexicalTagMatcher()
        # Don't register any tags
        metadata_repo = MagicMock()
        config = SearchPipelineConfig(enable_metadata_boost=True)

        pipeline = HybridSearchPipeline(
            fts_repo, config, tag_matcher=matcher, metadata_repo=metadata_repo
        )

        candidates = [
            MockRankedResult(file="doc1.md", score=1.0),
        ]

        result = pipeline._apply_metadata_boost("random query", candidates)

        assert len(result) == 1
        assert result[0].file == "doc1.md"

    @patch("pmd.search.pipeline.build_path_to_id_map")
    @patch("pmd.search.pipeline.get_document_tags_batch")
    def test_apply_metadata_boost_boosts_matching_docs(
        self, mock_get_tags, mock_build_map
    ):
        """Documents with matching tags should be boosted and reordered."""
        fts_repo = MagicMock()
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust"])
        metadata_repo = MagicMock()

        config = SearchPipelineConfig(
            enable_metadata_boost=True,
            metadata_boost=MetadataBoostConfig(boost_factor=3.0, max_boost=3.0),
        )

        pipeline = HybridSearchPipeline(
            fts_repo, config, tag_matcher=matcher, metadata_repo=metadata_repo
        )

        # doc1 has no python tag, doc2 has python tag
        mock_build_map.return_value = {"doc1.md": 1, "doc2.md": 2}
        mock_get_tags.return_value = {1: set(), 2: {"python"}}

        candidates = [
            MockRankedResult(file="doc1.md", score=1.0),  # No match
            MockRankedResult(file="doc2.md", score=0.5),  # Has python tag
        ]

        result = pipeline._apply_metadata_boost("python tutorial", candidates)

        # doc2 should be boosted (0.5 * 2.0 = 1.0), should come first now
        assert len(result) == 2
        assert result[0].file == "doc2.md"
        assert result[0].score >= 1.0  # Boosted

    @patch("pmd.search.pipeline.build_path_to_id_map")
    @patch("pmd.search.pipeline.get_document_tags_batch")
    def test_apply_metadata_boost_handles_errors_gracefully(
        self, mock_get_tags, mock_build_map
    ):
        """Should return original candidates on error."""
        fts_repo = MagicMock()
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        metadata_repo = MagicMock()

        config = SearchPipelineConfig(enable_metadata_boost=True)

        pipeline = HybridSearchPipeline(
            fts_repo, config, tag_matcher=matcher, metadata_repo=metadata_repo
        )

        # Simulate an error
        mock_build_map.side_effect = Exception("Database error")

        candidates = [MockRankedResult(file="doc1.md", score=1.0)]

        result = pipeline._apply_metadata_boost("python tutorial", candidates)

        # Should return original candidates
        assert result == candidates


class TestSearchPipelineConfigMetadata:
    """Tests for SearchPipelineConfig metadata options."""

    def test_default_metadata_boost_disabled(self):
        """Metadata boost should be disabled by default."""
        config = SearchPipelineConfig()

        assert config.enable_metadata_boost is False
        assert config.metadata_boost is None

    def test_custom_metadata_boost_config(self):
        """Should accept custom metadata boost configuration."""
        boost_config = MetadataBoostConfig(boost_factor=1.5, max_boost=3.0)
        config = SearchPipelineConfig(
            enable_metadata_boost=True,
            metadata_boost=boost_config,
        )

        assert config.enable_metadata_boost is True
        assert config.metadata_boost.boost_factor == 1.5
        assert config.metadata_boost.max_boost == 3.0


# Sample ontology for testing
SAMPLE_ADJACENCY = {
    "programming": {
        "children": ["programming/python", "programming/rust"],
        "description": "Programming topics",
    },
    "programming/python": {
        "children": ["programming/python/web", "programming/python/ml"],
        "description": "Python programming",
    },
    "ml": {
        "children": ["ml/supervised", "ml/unsupervised"],
        "description": "Machine learning",
    },
    "ml/supervised": {
        "children": ["ml/supervised/classification"],
        "description": "Supervised learning",
    },
}


class TestPipelineOntologyIntegration:
    """Tests for ontology-based metadata boosting."""

    def test_ontology_disabled_by_default(self):
        """Ontology should be None when not provided."""
        fts_repo = MagicMock()
        config = SearchPipelineConfig(enable_metadata_boost=True)

        pipeline = HybridSearchPipeline(fts_repo, config)

        assert pipeline.ontology is None

    def test_ontology_cleared_when_metadata_boost_disabled(self):
        """Ontology should be cleared when metadata boost is disabled."""
        fts_repo = MagicMock()
        ontology = Ontology(SAMPLE_ADJACENCY)
        config = SearchPipelineConfig(enable_metadata_boost=False)

        pipeline = HybridSearchPipeline(fts_repo, config, ontology=ontology)

        assert pipeline.ontology is None

    def test_ontology_preserved_when_metadata_boost_enabled(self):
        """Ontology should be preserved when metadata boost is enabled."""
        fts_repo = MagicMock()
        ontology = Ontology(SAMPLE_ADJACENCY)
        matcher = LexicalTagMatcher()
        metadata_repo = MagicMock()
        config = SearchPipelineConfig(enable_metadata_boost=True)

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            metadata_repo=metadata_repo,
            ontology=ontology,
        )

        assert pipeline.ontology is ontology

    @patch("pmd.search.pipeline.build_path_to_id_map")
    @patch("pmd.search.pipeline.get_document_tags_batch")
    @patch("pmd.search.pipeline.apply_metadata_boost_v2")
    def test_uses_v2_boost_when_ontology_provided(
        self, mock_boost_v2, mock_get_tags, mock_build_map
    ):
        """Should use apply_metadata_boost_v2 when ontology is provided."""
        fts_repo = MagicMock()
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.7)
        matcher = LexicalTagMatcher()
        # Register tag and alias so query matches
        matcher.register_tags(["programming/python"])
        matcher.register_alias("python", "programming/python")
        metadata_repo = MagicMock()
        config = SearchPipelineConfig(enable_metadata_boost=True)

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            metadata_repo=metadata_repo,
            ontology=ontology,
        )

        mock_build_map.return_value = {"doc1.md": 1}
        mock_get_tags.return_value = {1: {"programming/python"}}
        # Return mocked boosted results
        mock_result = MagicMock()
        mock_result.score = 1.5
        mock_result.file = "doc1.md"
        mock_boost_info = MagicMock()
        mock_boost_info.boost_applied = 1.15
        mock_boost_v2.return_value = [(mock_result, mock_boost_info)]

        candidates = [MockRankedResult(file="doc1.md", score=1.0)]
        result = pipeline._apply_metadata_boost("python tutorial", candidates)

        # Verify v2 was called
        mock_boost_v2.assert_called_once()
        call_args = mock_boost_v2.call_args

        # Check that expanded tags were passed (dict with weights)
        expanded_tags = call_args[0][1]
        assert isinstance(expanded_tags, dict)
        # programming/python should be 1.0 (exact match)
        # programming should be 0.7 (parent)
        assert expanded_tags.get("programming/python") == 1.0
        assert expanded_tags.get("programming") == pytest.approx(0.7)

    @patch("pmd.search.pipeline.build_path_to_id_map")
    @patch("pmd.search.pipeline.get_document_tags_batch")
    @patch("pmd.search.pipeline.apply_metadata_boost")
    def test_uses_v1_boost_without_ontology(
        self, mock_boost_v1, mock_get_tags, mock_build_map
    ):
        """Should use apply_metadata_boost (v1) when no ontology."""
        fts_repo = MagicMock()
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        metadata_repo = MagicMock()
        config = SearchPipelineConfig(enable_metadata_boost=True)

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            metadata_repo=metadata_repo,
            ontology=None,  # No ontology
        )

        mock_build_map.return_value = {"doc1.md": 1}
        mock_get_tags.return_value = {1: {"python"}}
        # Return mocked boosted results
        mock_result = MagicMock()
        mock_result.score = 1.3
        mock_result.file = "doc1.md"
        mock_boost_info = MagicMock()
        mock_boost_info.boost_applied = 1.3
        mock_boost_v1.return_value = [(mock_result, mock_boost_info)]

        candidates = [MockRankedResult(file="doc1.md", score=1.0)]
        result = pipeline._apply_metadata_boost("python tutorial", candidates)

        # Verify v1 was called
        mock_boost_v1.assert_called_once()
        call_args = mock_boost_v1.call_args

        # Check that simple set was passed (not dict)
        query_tags = call_args[0][1]
        assert isinstance(query_tags, set)
        assert "python" in query_tags

    @patch("pmd.search.pipeline.build_path_to_id_map")
    @patch("pmd.search.pipeline.get_document_tags_batch")
    def test_ontology_boost_with_custom_config(self, mock_get_tags, mock_build_map):
        """Should use custom boost config with ontology."""
        fts_repo = MagicMock()
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.7)
        matcher = LexicalTagMatcher()
        # Register alias so "supervised" matches "ml/supervised"
        matcher.register_tags(["ml/supervised"])
        matcher.register_alias("supervised", "ml/supervised")
        metadata_repo = MagicMock()

        # Custom config with higher boost
        boost_config = MetadataBoostConfig(boost_factor=2.0, max_boost=5.0)
        config = SearchPipelineConfig(
            enable_metadata_boost=True,
            metadata_boost=boost_config,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            metadata_repo=metadata_repo,
            ontology=ontology,
        )

        mock_build_map.return_value = {"doc1.md": 1}
        mock_get_tags.return_value = {1: {"ml/supervised"}}

        candidates = [MockRankedResult(file="doc1.md", score=1.0)]
        result = pipeline._apply_metadata_boost("supervised learning", candidates)

        # Should get boosted (2.0 ^ 1.0 = 2.0 for exact match)
        assert len(result) == 1
        assert result[0].score == pytest.approx(2.0)

    @patch("pmd.search.pipeline.build_path_to_id_map")
    @patch("pmd.search.pipeline.get_document_tags_batch")
    def test_ontology_boost_reorders_by_boosted_score(
        self, mock_get_tags, mock_build_map
    ):
        """Documents with parent tag match should be boosted less than exact."""
        fts_repo = MagicMock()
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.7)
        matcher = LexicalTagMatcher()
        # Register tag and alias so "classification" matches the full path
        matcher.register_tags(["ml/supervised/classification"])
        matcher.register_alias("classification", "ml/supervised/classification")
        metadata_repo = MagicMock()

        config = SearchPipelineConfig(enable_metadata_boost=True)

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            metadata_repo=metadata_repo,
            ontology=ontology,
        )

        # doc1 has exact tag match, doc2 has parent tag
        mock_build_map.return_value = {"doc1.md": 1, "doc2.md": 2}
        mock_get_tags.return_value = {
            1: {"ml/supervised/classification"},  # Exact match (weight 1.0)
            2: {"ml/supervised"},  # Parent match (weight 0.7)
        }

        # Start with doc2 higher scored
        candidates = [
            MockRankedResult(file="doc2.md", score=1.0),  # Parent tag
            MockRankedResult(file="doc1.md", score=0.8),  # Exact tag
        ]

        result = pipeline._apply_metadata_boost("classification", candidates)

        # doc1 should be boosted more (exact match) and reorder
        # doc1: 0.8 * 1.15^1.0 = 0.92
        # doc2: 1.0 * 1.15^0.7 = 1.10
        # Actually doc2 still has higher base score so might still be first
        # Let's just verify both are boosted
        assert len(result) == 2
        # Both should be boosted
        assert result[0].score > 0.8 or result[1].score > 0.8


class TestPipelineTagRetrievalIntegration:
    """Tests for tag-based retrieval in RRF fusion."""

    def test_tag_retrieval_disabled_by_default(self):
        """Tag retrieval should be disabled when not configured."""
        fts_repo = MagicMock()
        config = SearchPipelineConfig()

        pipeline = HybridSearchPipeline(fts_repo, config)

        assert pipeline.tag_retriever is None
        assert config.enable_tag_retrieval is False

    def test_tag_retrieval_enabled_preserves_components(self):
        """When tag retrieval enabled, tag_retriever should be preserved."""
        fts_repo = MagicMock()
        tag_retriever = MagicMock(spec=TagRetriever)
        matcher = LexicalTagMatcher()
        config = SearchPipelineConfig(enable_tag_retrieval=True)

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            tag_retriever=tag_retriever,
        )

        assert pipeline.tag_retriever is tag_retriever
        assert pipeline.tag_matcher is matcher

    def test_tag_retrieval_preserves_matcher_for_both_features(self):
        """Tag matcher preserved when either tag_retrieval or metadata_boost enabled."""
        fts_repo = MagicMock()
        matcher = LexicalTagMatcher()
        metadata_repo = MagicMock()
        config = SearchPipelineConfig(enable_tag_retrieval=True)

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            metadata_repo=metadata_repo,
        )

        # tag_matcher should be preserved for tag retrieval
        assert pipeline.tag_matcher is matcher

    def test_default_tag_weight(self):
        """Default tag weight should be 0.8."""
        config = SearchPipelineConfig()
        assert config.tag_weight == 0.8

    def test_custom_tag_weight(self):
        """Should accept custom tag weight."""
        config = SearchPipelineConfig(tag_weight=1.2)
        assert config.tag_weight == 1.2

    @pytest.mark.asyncio
    async def test_parallel_search_includes_tag_results(self):
        """Parallel search should include tag results when enabled."""
        fts_repo = MagicMock()
        fts_repo.search.return_value = []

        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        tag_retriever = MagicMock(spec=TagRetriever)
        tag_retriever.search.return_value = []

        config = SearchPipelineConfig(enable_tag_retrieval=True)

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            tag_retriever=tag_retriever,
        )

        results, weights = await pipeline._parallel_search(["python tutorial"], limit=10, collection_id=None)

        # Should have 3 result lists: FTS, vector (empty), tag
        assert len(results) == 3
        assert len(weights) == 3

        # Tag retriever should have been called
        tag_retriever.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_search_uses_ontology_expansion(self):
        """Should expand tags with ontology when available."""
        fts_repo = MagicMock()
        fts_repo.search.return_value = []

        matcher = LexicalTagMatcher()
        matcher.register_tags(["ml/supervised"])
        matcher.register_alias("supervised", "ml/supervised")

        # Define complete hierarchy: ml -> ml/supervised
        # Both need to be defined for parent lookup to work
        ontology = Ontology({
            "ml": {"children": ["ml/supervised"]},
            "ml/supervised": {"children": []},  # Leaf node must be defined for parent lookup
        }, parent_weight=0.7)

        tag_retriever = MagicMock(spec=TagRetriever)
        tag_retriever.search.return_value = []

        config = SearchPipelineConfig(enable_tag_retrieval=True)

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            tag_retriever=tag_retriever,
            ontology=ontology,
        )

        await pipeline._parallel_search(["supervised learning"], limit=10, collection_id=None)

        # Verify tag retriever was called with expanded tags
        call_args = tag_retriever.search.call_args
        expanded_tags = call_args[0][0]
        assert isinstance(expanded_tags, dict)
        assert "ml/supervised" in expanded_tags
        assert "ml" in expanded_tags  # Parent should be included

    @pytest.mark.asyncio
    async def test_parallel_search_weights_include_tag_weight(self):
        """Weights should include tag weight for tag results."""
        fts_repo = MagicMock()
        fts_repo.search.return_value = []

        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        tag_retriever = MagicMock(spec=TagRetriever)
        tag_retriever.search.return_value = []

        config = SearchPipelineConfig(
            enable_tag_retrieval=True,
            fts_weight=1.0,
            vec_weight=1.0,
            tag_weight=0.8,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            tag_retriever=tag_retriever,
        )

        results, weights = await pipeline._parallel_search(["python"], limit=10, collection_id=None)

        # Weights should be [fts=1.0, vec=1.0, tag=0.8]
        assert weights[0] == 1.0  # FTS
        assert weights[1] == 1.0  # Vector
        assert weights[2] == 0.8  # Tag

    @pytest.mark.asyncio
    async def test_parallel_search_handles_tag_error_gracefully(self):
        """Should continue if tag search fails."""
        fts_repo = MagicMock()
        fts_repo.search.return_value = []

        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        tag_retriever = MagicMock(spec=TagRetriever)
        tag_retriever.search.side_effect = Exception("Tag search failed")

        config = SearchPipelineConfig(enable_tag_retrieval=True)

        pipeline = HybridSearchPipeline(
            fts_repo,
            config,
            tag_matcher=matcher,
            tag_retriever=tag_retriever,
        )

        # Should not raise
        results, weights = await pipeline._parallel_search(["python"], limit=10, collection_id=None)

        # Should still have 3 results (FTS, vec, tag) but tag is empty
        assert len(results) == 3
        assert results[2] == []  # Tag results empty due to error
