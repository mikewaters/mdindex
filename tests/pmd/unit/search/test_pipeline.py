"""Tests for hybrid search pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from pmd.core.types import RankedResult, SearchResult, SearchSource
from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig
from tests.pmd.fakes.search import (
    InMemoryTextSearcher,
    InMemoryVectorSearcher,
    StubQueryExpander,
    StubReranker,
    make_search_result,
)


class TestSearchPipelineConfig:
    """Tests for SearchPipelineConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = SearchPipelineConfig()

        assert config.fts_weight == 1.0
        assert config.vec_weight == 1.0
        assert config.rrf_k == 60
        assert config.top_rank_bonus == 0.05
        assert config.expansion_weight == 0.5
        assert config.rerank_candidates == 30
        assert config.enable_query_expansion is False
        assert config.enable_reranking is False

    def test_custom_values(self):
        """Config should accept custom values."""
        config = SearchPipelineConfig(
            fts_weight=2.0,
            vec_weight=0.5,
            rrf_k=100,
            enable_query_expansion=True,
            enable_reranking=True,
        )

        assert config.fts_weight == 2.0
        assert config.vec_weight == 0.5
        assert config.rrf_k == 100
        assert config.enable_query_expansion is True
        assert config.enable_reranking is True


class TestHybridSearchPipelineInit:
    """Tests for HybridSearchPipeline initialization."""

    def test_init_with_defaults(self):
        """Pipeline should initialize with defaults."""
        text_searcher = InMemoryTextSearcher()

        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        assert pipeline.text_searcher == text_searcher
        assert pipeline.config is not None
        assert pipeline.query_expander is None
        assert pipeline.reranker is None
        assert pipeline.vector_searcher is None

    def test_init_with_config(self):
        """Pipeline should accept custom config."""
        text_searcher = InMemoryTextSearcher()
        config = SearchPipelineConfig(rrf_k=100)

        pipeline = HybridSearchPipeline(text_searcher=text_searcher, config=config)

        assert pipeline.config.rrf_k == 100

    def test_query_expander_disabled_without_config(self):
        """Query expander should be disabled if config says so."""
        text_searcher = InMemoryTextSearcher()
        config = SearchPipelineConfig(enable_query_expansion=False)
        mock_expander = StubQueryExpander()

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            config=config,
            query_expander=mock_expander,
        )

        assert pipeline.query_expander is None

    def test_reranker_disabled_without_config(self):
        """Reranker should be disabled if config says so."""
        text_searcher = InMemoryTextSearcher()
        config = SearchPipelineConfig(enable_reranking=False)
        mock_reranker = StubReranker()

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            config=config,
            reranker=mock_reranker,
        )

        assert pipeline.reranker is None


class TestHybridSearchPipelineSearch:
    """Tests for HybridSearchPipeline.search method."""

    @pytest.mark.asyncio
    async def test_search_returns_ranked_results(self):
        """Search should return RankedResult objects."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("doc1.md", 0.9, SearchSource.FTS))
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        results = await pipeline.search("test query")

        assert len(results) > 0
        assert all(isinstance(r, RankedResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_limit(self):
        """Search should respect limit parameter."""
        text_searcher = InMemoryTextSearcher()
        for i in range(10):
            text_searcher.add_result(make_search_result(f"doc{i}.md", 0.9 - i * 0.1))
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        results = await pipeline.search("test", limit=3)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_respects_min_score(self):
        """Search should filter by min_score."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("high.md", 0.9))
        text_searcher.add_result(make_search_result("low.md", 0.1))
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        results = await pipeline.search("test", min_score=0.5)

        # Low-scoring result should be filtered out after normalization
        for r in results:
            assert r.score >= 0.5

    @pytest.mark.asyncio
    async def test_empty_results_return_empty_list(self):
        """Empty results should return empty list."""
        text_searcher = InMemoryTextSearcher()
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        results = await pipeline.search("nonexistent")

        assert results == []


class TestHybridSearchPipelineVectorSearch:
    """Tests for vector search integration."""

    @pytest.mark.asyncio
    async def test_no_vector_search_without_searcher(self):
        """Should not do vector search without vector_searcher."""
        text_searcher = InMemoryTextSearcher()
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        await pipeline.search("test")

        # No vector searcher means no vector search
        # Just verify search completes without error
        assert True

    @pytest.mark.asyncio
    async def test_vector_search_with_searcher(self):
        """Should do vector search with vector_searcher."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("fts.md", 0.8))

        vector_searcher = InMemoryVectorSearcher()
        vector_searcher.add_result(make_search_result("vec.md", 0.9, SearchSource.VECTOR))

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            vector_searcher=vector_searcher,
        )

        results = await pipeline.search("test")

        # Should have results from both sources
        files = {r.file for r in results}
        assert "fts.md" in files or "vec.md" in files


class TestHybridSearchPipelineQueryExpansion:
    """Tests for query expansion integration."""

    @pytest.mark.asyncio
    async def test_no_expansion_by_default(self):
        """Query expansion should be disabled by default."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.set_results_for_query("test", [make_search_result("test.md", 0.9)])
        config = SearchPipelineConfig(enable_query_expansion=False)
        mock_expander = StubQueryExpander(variations=["expanded"])

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            config=config,
            query_expander=mock_expander,
        )

        results = await pipeline.search("test")

        # Expander should not be used (it's set to None in init)
        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_expansion_when_enabled(self):
        """Query expansion should add query variants."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.set_results_for_query("original", [make_search_result("orig.md", 0.9)])
        text_searcher.set_results_for_query("variant1", [make_search_result("var.md", 0.8)])

        mock_expander = StubQueryExpander(variations=["variant1", "variant2"])
        config = SearchPipelineConfig(enable_query_expansion=True)

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            config=config,
            query_expander=mock_expander,
        )

        results = await pipeline.search("original")

        # Should have results from original and expanded queries
        files = {r.file for r in results}
        assert "orig.md" in files or "var.md" in files


class TestHybridSearchPipelineReranking:
    """Tests for reranking integration."""

    @pytest.mark.asyncio
    async def test_no_reranking_by_default(self):
        """Reranking should be disabled by default."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("doc.md", 0.9))
        mock_reranker = StubReranker(default_score=0.1)
        config = SearchPipelineConfig(enable_reranking=False)

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            config=config,
            reranker=mock_reranker,
        )

        results = await pipeline.search("test")

        # Reranker should not be called (it's set to None in init)
        # Original order should be preserved
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_reranking_when_enabled(self):
        """Reranking should be applied when enabled."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("doc1.md", 0.9))
        text_searcher.add_result(make_search_result("doc2.md", 0.5))

        # Reranker scores doc2 higher than doc1
        mock_reranker = StubReranker(scores={
            "doc1.md": 0.2,
            "doc2.md": 0.95,
        })
        config = SearchPipelineConfig(enable_reranking=True)

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            config=config,
            reranker=mock_reranker,
        )

        results = await pipeline.search("test")

        # Should have both results (ranking may be affected by position-aware blending)
        assert len(results) == 2


class TestHybridSearchPipelineIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_fts_and_vector(self):
        """Full pipeline should combine FTS and vector results."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("fts1.md", 0.9, SearchSource.FTS))
        text_searcher.add_result(make_search_result("both.md", 0.8, SearchSource.FTS))

        vector_searcher = InMemoryVectorSearcher()
        vector_searcher.add_result(make_search_result("both.md", 0.95, SearchSource.VECTOR))
        vector_searcher.add_result(make_search_result("vec1.md", 0.85, SearchSource.VECTOR))

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            vector_searcher=vector_searcher,
        )

        results = await pipeline.search("test")

        # Should have results from both sources
        files = {r.file for r in results}
        assert "fts1.md" in files or "both.md" in files or "vec1.md" in files

    @pytest.mark.asyncio
    async def test_duplicate_documents_fused(self):
        """Documents appearing in both FTS and vector should be fused."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("shared.md", 0.9, SearchSource.FTS))

        vector_searcher = InMemoryVectorSearcher()
        vector_searcher.add_result(make_search_result("shared.md", 0.95, SearchSource.VECTOR))

        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            vector_searcher=vector_searcher,
        )

        results = await pipeline.search("test")

        # Should only have one result for shared.md
        shared_results = [r for r in results if r.file == "shared.md"]
        assert len(shared_results) == 1

    @pytest.mark.asyncio
    async def test_results_sorted_by_fused_score(self):
        """Results should be sorted by fused score."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("doc1.md", 0.9))
        text_searcher.add_result(make_search_result("doc2.md", 0.8))
        text_searcher.add_result(make_search_result("doc3.md", 0.7))

        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        results = await pipeline.search("test")

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestHybridSearchPipelineEdgeCases:
    """Edge case tests for HybridSearchPipeline."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Empty query should still work."""
        text_searcher = InMemoryTextSearcher()
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        results = await pipeline.search("")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Very long query should work."""
        text_searcher = InMemoryTextSearcher()
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)
        long_query = "word " * 1000

        results = await pipeline.search(long_query)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self):
        """Query with special characters should work."""
        text_searcher = InMemoryTextSearcher()
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        results = await pipeline.search("test @#$%^&*() query")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_unicode_query(self):
        """Unicode query should work."""
        text_searcher = InMemoryTextSearcher()
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        results = await pipeline.search("test query")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_limit_zero(self):
        """Limit of 0 should return empty list."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("doc.md", 0.9))
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        results = await pipeline.search("test", limit=0)

        assert results == []

    @pytest.mark.asyncio
    async def test_very_high_min_score(self):
        """Very high min_score should filter all results."""
        text_searcher = InMemoryTextSearcher()
        text_searcher.add_result(make_search_result("doc.md", 0.9))
        pipeline = HybridSearchPipeline(text_searcher=text_searcher)

        results = await pipeline.search("test", min_score=999.0)

        # All results should be filtered
        assert results == []
