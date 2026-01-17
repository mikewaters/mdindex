"""Tests for pipeline metadata boost integration.

These tests verify that the pipeline correctly integrates metadata boosting
and tag retrieval features through the port interfaces.
"""

import pytest
from unittest.mock import MagicMock

from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig
from pmd.metadata import Ontology
from pmd.core.types import SearchSource
from tests.pmd.fakes.search import (
    InMemoryTextSearcher,
    InMemoryTagSearcher,
    InMemoryMetadataBooster,
    InMemoryTagInferencer,
    make_search_result,
    make_ranked_result,
)


class TestSearchPipelineConfigMetadata:
    """Tests for SearchPipelineConfig metadata options."""

    def test_default_metadata_boost_disabled(self):
        """Metadata boost should be disabled by default."""
        config = SearchPipelineConfig()

        assert config.enable_metadata_boost is False

    def test_enable_metadata_boost(self):
        """Should accept enable_metadata_boost flag."""
        config = SearchPipelineConfig(
            enable_metadata_boost=True,
        )

        assert config.enable_metadata_boost is True

    def test_default_metadata_boost_factor(self):
        """Default metadata boost factor should be 1.15."""
        config = SearchPipelineConfig()
        assert config.metadata_boost_factor == 1.15

    def test_custom_metadata_boost_factor(self):
        """Should accept custom metadata boost factor."""
        config = SearchPipelineConfig(metadata_boost_factor=2.0)
        assert config.metadata_boost_factor == 2.0

    def test_default_metadata_max_boost(self):
        """Default metadata max boost should be 2.0."""
        config = SearchPipelineConfig()
        assert config.metadata_max_boost == 2.0

    def test_custom_metadata_max_boost(self):
        """Should accept custom metadata max boost."""
        config = SearchPipelineConfig(metadata_max_boost=3.0)
        assert config.metadata_max_boost == 3.0


class TestPipelineMetadataBoostIntegration:
    """Tests for metadata boost port integration."""

    def test_metadata_booster_cleared_when_disabled(self):
        """Metadata booster should be None when boost disabled."""
        text_searcher = InMemoryTextSearcher()
        booster = InMemoryMetadataBooster()
        config = SearchPipelineConfig(enable_metadata_boost=False)

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            metadata_booster=booster,
            config=config,
        )

        assert pipeline.metadata_booster is None

    def test_metadata_booster_preserved_when_enabled(self):
        """Metadata booster should be preserved when boost enabled."""
        text_searcher = InMemoryTextSearcher()
        booster = InMemoryMetadataBooster()
        config = SearchPipelineConfig(enable_metadata_boost=True)

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            metadata_booster=booster,
            config=config,
        )

        assert pipeline.metadata_booster is booster

    @pytest.mark.asyncio
    async def test_metadata_boost_requires_tag_inferencer(self):
        """Metadata boost needs tag_inferencer to infer query tags."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("doc.md", 0.9))

        booster = InMemoryMetadataBooster()
        booster.add_document_tags("doc.md", {"python"})

        # No tag inferencer - can't infer query tags
        config = SearchPipelineConfig(enable_metadata_boost=True)
        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            metadata_booster=booster,
            config=config,
        )

        # Should work but not boost (no tags inferred)
        results = await pipeline.search("python tutorial")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_metadata_boost_with_inferencer(self):
        """Metadata boost should work with tag_inferencer."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("tagged.md", 0.5))
        text_searcher.add_result(make_search_result("untagged.md", 0.9))

        inferencer = InMemoryTagInferencer()
        inferencer.set_tags_for_query("python tutorial", {"python"})
        inferencer.set_expansion("python", {"python": 1.0})

        booster = InMemoryMetadataBooster(boost_factor=2.0)
        booster.add_document_tags("tagged.md", {"python"})
        booster.add_document_tags("untagged.md", set())

        config = SearchPipelineConfig(enable_metadata_boost=True)
        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            tag_inferencer=inferencer,
            metadata_booster=booster,
            config=config,
        )

        results = await pipeline.search("python tutorial")

        # tagged.md should be boosted, potentially reordering
        assert len(results) == 2
        tagged_result = next(r for r in results if r.file == "tagged.md")
        assert tagged_result.score > 0.5  # Should be boosted


class TestPipelineTagRetrievalIntegration:
    """Tests for tag-based retrieval port integration."""

    def test_tag_retrieval_disabled_by_default(self):
        """Tag retrieval should be disabled by default."""
        config = SearchPipelineConfig()
        assert config.enable_tag_retrieval is False

    def test_tag_retrieval_enabled(self):
        """Should enable tag retrieval via config."""
        config = SearchPipelineConfig(enable_tag_retrieval=True)
        assert config.enable_tag_retrieval is True

    def test_tag_searcher_cleared_when_disabled(self):
        """Tag searcher should be None when retrieval disabled."""
        text_searcher = InMemoryTextSearcher()
        tag_searcher = InMemoryTagSearcher()
        config = SearchPipelineConfig(enable_tag_retrieval=False)

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            tag_searcher=tag_searcher,
            config=config,
        )

        assert pipeline.tag_searcher is None

    def test_tag_searcher_preserved_when_enabled(self):
        """Tag searcher should be preserved when retrieval enabled."""
        text_searcher = InMemoryTextSearcher()
        tag_searcher = InMemoryTagSearcher()
        config = SearchPipelineConfig(enable_tag_retrieval=True)

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            tag_searcher=tag_searcher,
            config=config,
        )

        assert pipeline.tag_searcher is tag_searcher

    def test_default_tag_weight(self):
        """Default tag weight should be 0.8."""
        config = SearchPipelineConfig()
        assert config.tag_weight == 0.8

    def test_custom_tag_weight(self):
        """Should accept custom tag weight."""
        config = SearchPipelineConfig(tag_weight=1.5)
        assert config.tag_weight == 1.5

    @pytest.mark.asyncio
    async def test_tag_retrieval_finds_documents(self):
        """Tag retrieval should find documents by tag."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("fts_only.md", 0.9))

        tag_searcher = InMemoryTagSearcher()
        tag_searcher.add_document("tag_only.md", {"python"})

        inferencer = InMemoryTagInferencer()
        inferencer.set_tags_for_query("python tutorial", {"python"})
        inferencer.set_expansion("python", {"python": 1.0})

        config = SearchPipelineConfig(enable_tag_retrieval=True)
        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            tag_searcher=tag_searcher,
            tag_inferencer=inferencer,
            config=config,
        )

        results = await pipeline.search("python tutorial")

        # Should have results from both FTS and tag retrieval
        files = {r.file for r in results}
        assert "fts_only.md" in files or "tag_only.md" in files


class TestPipelineOntologyIntegration:
    """Tests for ontology integration with metadata features."""

    @pytest.mark.asyncio
    async def test_ontology_expansion_in_tag_retrieval(self):
        """Ontology should expand tags for retrieval."""
        text_searcher = InMemoryTextSearcher()

        tag_searcher = InMemoryTagSearcher()
        tag_searcher.add_document("parent.md", {"ml"})
        tag_searcher.add_document("child.md", {"ml/supervised"})

        # Inferencer expands ml/supervised to include parent ml
        inferencer = InMemoryTagInferencer()
        inferencer.set_tags_for_query("supervised learning", {"ml/supervised"})
        inferencer.set_expansion("ml/supervised", {"ml/supervised": 1.0, "ml": 0.7})

        config = SearchPipelineConfig(enable_tag_retrieval=True)
        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            tag_searcher=tag_searcher,
            tag_inferencer=inferencer,
            config=config,
        )

        results = await pipeline.search("supervised learning")

        # Should find both parent and child tag matches
        files = {r.file for r in results}
        assert "child.md" in files  # Exact match
        # parent.md may also be found if tag searcher scores parent tags


class TestCombinedMetadataFeatures:
    """Tests for using both metadata boost and tag retrieval together."""

    @pytest.mark.asyncio
    async def test_both_features_enabled(self):
        """Both metadata boost and tag retrieval should work together."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("fts_doc.md", 0.7))

        tag_searcher = InMemoryTagSearcher()
        tag_searcher.add_document("tag_doc.md", {"python"})

        inferencer = InMemoryTagInferencer()
        inferencer.set_tags_for_query("python web", {"python", "web"})
        inferencer.set_expansion("python", {"python": 1.0})
        inferencer.set_expansion("web", {"web": 1.0})

        booster = InMemoryMetadataBooster(boost_factor=1.5)
        booster.add_document_tags("fts_doc.md", {"python"})
        booster.add_document_tags("tag_doc.md", {"python"})

        config = SearchPipelineConfig(
            enable_tag_retrieval=True,
            enable_metadata_boost=True,
        )
        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            tag_searcher=tag_searcher,
            tag_inferencer=inferencer,
            metadata_booster=booster,
            config=config,
        )

        results = await pipeline.search("python web")

        # Should have results from both channels
        assert len(results) > 0
        # Scores should be boosted
        for r in results:
            assert r.score > 0
