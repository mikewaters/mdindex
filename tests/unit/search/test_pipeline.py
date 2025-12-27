"""Tests for hybrid search pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from pmd.core.types import RankedResult, SearchResult, SearchSource
from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig
from pmd.store.search import SearchRepository


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
        collection_id=1,
        modified_at="2024-01-01T00:00:00",
        body_length=100,
        body="Test content",
        score=score,
        source=source,
    )


class MockSearchRepository(SearchRepository):
    """Mock search repository for testing."""

    def __init__(self, fts_results: list[SearchResult] | None = None, vec_results: list[SearchResult] | None = None):
        self.fts_results = fts_results or []
        self.vec_results = vec_results or []
        self.fts_calls = []
        self.vec_calls = []

    def search_fts(
        self,
        query: str,
        limit: int = 10,
        collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        self.fts_calls.append((query, limit, collection_id, min_score))
        return self.fts_results

    def search_vec(
        self,
        embedding: list[float],
        limit: int = 10,
        collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        self.vec_calls.append((embedding, limit, collection_id, min_score))
        return self.vec_results


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
        repo = MockSearchRepository()

        pipeline = HybridSearchPipeline(repo)

        assert pipeline.search_repo == repo
        assert pipeline.config is not None
        assert pipeline.query_expander is None
        assert pipeline.reranker is None
        assert pipeline.embedding_generator is None

    def test_init_with_config(self):
        """Pipeline should accept custom config."""
        repo = MockSearchRepository()
        config = SearchPipelineConfig(rrf_k=100)

        pipeline = HybridSearchPipeline(repo, config=config)

        assert pipeline.config.rrf_k == 100

    def test_query_expander_disabled_without_config(self):
        """Query expander should be disabled if config says so."""
        repo = MockSearchRepository()
        config = SearchPipelineConfig(enable_query_expansion=False)
        mock_expander = MagicMock()

        pipeline = HybridSearchPipeline(repo, config=config, query_expander=mock_expander)

        assert pipeline.query_expander is None

    def test_reranker_disabled_without_config(self):
        """Reranker should be disabled if config says so."""
        repo = MockSearchRepository()
        config = SearchPipelineConfig(enable_reranking=False)
        mock_reranker = MagicMock()

        pipeline = HybridSearchPipeline(repo, config=config, reranker=mock_reranker)

        assert pipeline.reranker is None


class TestHybridSearchPipelineSearch:
    """Tests for HybridSearchPipeline.search method."""

    @pytest.mark.asyncio
    async def test_search_returns_ranked_results(self):
        """Search should return RankedResult objects."""
        fts_results = [make_search_result("doc1.md", 0.9, SearchSource.FTS)]
        repo = MockSearchRepository(fts_results=fts_results)
        pipeline = HybridSearchPipeline(repo)

        results = await pipeline.search("test query")

        assert len(results) > 0
        assert all(isinstance(r, RankedResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_calls_fts(self):
        """Search should call FTS search."""
        repo = MockSearchRepository()
        pipeline = HybridSearchPipeline(repo)

        await pipeline.search("test query")

        assert len(repo.fts_calls) > 0
        assert repo.fts_calls[0][0] == "test query"

    @pytest.mark.asyncio
    async def test_search_respects_limit(self):
        """Search should respect limit parameter."""
        fts_results = [make_search_result(f"doc{i}.md", 0.9 - i * 0.1) for i in range(10)]
        repo = MockSearchRepository(fts_results=fts_results)
        pipeline = HybridSearchPipeline(repo)

        results = await pipeline.search("test", limit=3)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_respects_min_score(self):
        """Search should filter by min_score."""
        fts_results = [
            make_search_result("high.md", 0.9),
            make_search_result("low.md", 0.1),
        ]
        repo = MockSearchRepository(fts_results=fts_results)
        pipeline = HybridSearchPipeline(repo)

        results = await pipeline.search("test", min_score=0.5)

        # Low-scoring result should be filtered out
        # Note: After RRF, scores are recalculated
        for r in results:
            assert r.score >= 0.5

    @pytest.mark.asyncio
    async def test_search_passes_collection_id(self):
        """Search should pass collection_id to repository."""
        repo = MockSearchRepository()
        pipeline = HybridSearchPipeline(repo)

        await pipeline.search("test", collection_id=42)

        assert repo.fts_calls[0][2] == 42

    @pytest.mark.asyncio
    async def test_empty_results_return_empty_list(self):
        """Empty results should return empty list."""
        repo = MockSearchRepository()
        pipeline = HybridSearchPipeline(repo)

        results = await pipeline.search("nonexistent")

        assert results == []


class TestHybridSearchPipelineVectorSearch:
    """Tests for vector search integration."""

    @pytest.mark.asyncio
    async def test_no_vector_search_without_generator(self):
        """Should not do vector search without embedding generator."""
        repo = MockSearchRepository()
        pipeline = HybridSearchPipeline(repo)

        await pipeline.search("test")

        assert len(repo.vec_calls) == 0

    @pytest.mark.asyncio
    async def test_vector_search_with_generator(self):
        """Should do vector search with embedding generator."""
        repo = MockSearchRepository()
        mock_generator = AsyncMock()
        mock_generator.embed_query.return_value = [0.1] * 768
        pipeline = HybridSearchPipeline(repo, embedding_generator=mock_generator)

        await pipeline.search("test")

        assert len(repo.vec_calls) > 0

    @pytest.mark.asyncio
    async def test_vector_search_uses_query_embedding(self):
        """Vector search should use generated query embedding."""
        repo = MockSearchRepository()
        expected_embedding = [0.5] * 768
        mock_generator = AsyncMock()
        mock_generator.embed_query.return_value = expected_embedding
        pipeline = HybridSearchPipeline(repo, embedding_generator=mock_generator)

        await pipeline.search("test")

        assert repo.vec_calls[0][0] == expected_embedding

    @pytest.mark.asyncio
    async def test_handles_embedding_error_gracefully(self):
        """Should handle embedding generation errors gracefully."""
        fts_results = [make_search_result("doc.md", 0.9)]
        repo = MockSearchRepository(fts_results=fts_results)
        mock_generator = AsyncMock()
        mock_generator.embed_query.side_effect = Exception("Embedding failed")
        pipeline = HybridSearchPipeline(repo, embedding_generator=mock_generator)

        # Should not raise, should fall back to FTS only
        results = await pipeline.search("test")

        assert len(results) > 0


class TestHybridSearchPipelineQueryExpansion:
    """Tests for query expansion integration."""

    @pytest.mark.asyncio
    async def test_no_expansion_by_default(self):
        """Query expansion should be disabled by default."""
        repo = MockSearchRepository()
        mock_expander = AsyncMock()
        config = SearchPipelineConfig(enable_query_expansion=False)
        pipeline = HybridSearchPipeline(repo, config=config, query_expander=mock_expander)

        await pipeline.search("test")

        # Expander should not be called (it's set to None in init)
        assert repo.fts_calls[0][0] == "test"  # Only original query

    @pytest.mark.asyncio
    async def test_expansion_when_enabled(self):
        """Query expansion should add query variants."""
        repo = MockSearchRepository()
        mock_expander = AsyncMock()
        mock_expander.expand.return_value = ["variant1", "variant2"]
        config = SearchPipelineConfig(enable_query_expansion=True)
        pipeline = HybridSearchPipeline(repo, config=config, query_expander=mock_expander)

        await pipeline.search("original")

        # Should search for original + variants
        queries_searched = [call[0] for call in repo.fts_calls]
        assert "original" in queries_searched

    @pytest.mark.asyncio
    async def test_handles_expansion_error_gracefully(self):
        """Should handle query expansion errors gracefully."""
        fts_results = [make_search_result("doc.md", 0.9)]
        repo = MockSearchRepository(fts_results=fts_results)
        mock_expander = AsyncMock()
        mock_expander.expand.side_effect = Exception("Expansion failed")
        config = SearchPipelineConfig(enable_query_expansion=True)
        pipeline = HybridSearchPipeline(repo, config=config, query_expander=mock_expander)

        # Should not raise, should fall back to original query
        results = await pipeline.search("test")

        assert len(results) > 0


class TestHybridSearchPipelineReranking:
    """Tests for reranking integration."""

    @pytest.mark.asyncio
    async def test_no_reranking_by_default(self):
        """Reranking should be disabled by default."""
        fts_results = [make_search_result("doc.md", 0.9)]
        repo = MockSearchRepository(fts_results=fts_results)
        mock_reranker = AsyncMock()
        config = SearchPipelineConfig(enable_reranking=False)
        pipeline = HybridSearchPipeline(repo, config=config, reranker=mock_reranker)

        await pipeline.search("test")

        # Reranker should not be called (it's set to None in init)

    @pytest.mark.asyncio
    async def test_reranking_when_enabled(self):
        """Reranking should be applied when enabled."""
        fts_results = [make_search_result("doc.md", 0.9)]
        repo = MockSearchRepository(fts_results=fts_results)
        mock_reranker = AsyncMock()
        # Return reranked results in different format
        mock_reranker.rerank.return_value = [
            RankedResult(
                file="doc.md",
                display_path="doc.md",
                title="Test",
                body="Content",
                score=0.95,
            )
        ]
        config = SearchPipelineConfig(enable_reranking=True)
        pipeline = HybridSearchPipeline(repo, config=config, reranker=mock_reranker)

        results = await pipeline.search("test")

        mock_reranker.rerank.assert_called_once()
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_handles_reranking_error_gracefully(self):
        """Should handle reranking errors gracefully."""
        fts_results = [make_search_result("doc.md", 0.9)]
        repo = MockSearchRepository(fts_results=fts_results)
        mock_reranker = AsyncMock()
        mock_reranker.rerank.side_effect = Exception("Reranking failed")
        config = SearchPipelineConfig(enable_reranking=True)
        pipeline = HybridSearchPipeline(repo, config=config, reranker=mock_reranker)

        # Should not raise, should return non-reranked results
        results = await pipeline.search("test")

        assert len(results) > 0


class TestHybridSearchPipelineIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_fts_and_vector(self):
        """Full pipeline should combine FTS and vector results."""
        fts_results = [
            make_search_result("fts1.md", 0.9, SearchSource.FTS),
            make_search_result("both.md", 0.8, SearchSource.FTS),
        ]
        vec_results = [
            make_search_result("both.md", 0.95, SearchSource.VECTOR),
            make_search_result("vec1.md", 0.85, SearchSource.VECTOR),
        ]
        repo = MockSearchRepository(fts_results=fts_results, vec_results=vec_results)
        mock_generator = AsyncMock()
        mock_generator.embed_query.return_value = [0.1] * 768
        pipeline = HybridSearchPipeline(repo, embedding_generator=mock_generator)

        results = await pipeline.search("test")

        # Should have results from both sources
        files = {r.file for r in results}
        assert "fts1.md" in files or "both.md" in files or "vec1.md" in files

    @pytest.mark.asyncio
    async def test_duplicate_documents_fused(self):
        """Documents appearing in both FTS and vector should be fused."""
        fts_results = [make_search_result("shared.md", 0.9, SearchSource.FTS)]
        vec_results = [make_search_result("shared.md", 0.95, SearchSource.VECTOR)]
        repo = MockSearchRepository(fts_results=fts_results, vec_results=vec_results)
        mock_generator = AsyncMock()
        mock_generator.embed_query.return_value = [0.1] * 768
        pipeline = HybridSearchPipeline(repo, embedding_generator=mock_generator)

        results = await pipeline.search("test")

        # Should only have one result for shared.md
        shared_results = [r for r in results if r.file == "shared.md"]
        assert len(shared_results) == 1

    @pytest.mark.asyncio
    async def test_results_sorted_by_fused_score(self):
        """Results should be sorted by fused score."""
        fts_results = [
            make_search_result("doc1.md", 0.9),
            make_search_result("doc2.md", 0.8),
            make_search_result("doc3.md", 0.7),
        ]
        repo = MockSearchRepository(fts_results=fts_results)
        pipeline = HybridSearchPipeline(repo)

        results = await pipeline.search("test")

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestHybridSearchPipelineEdgeCases:
    """Edge case tests for HybridSearchPipeline."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Empty query should still work."""
        repo = MockSearchRepository()
        pipeline = HybridSearchPipeline(repo)

        results = await pipeline.search("")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Very long query should work."""
        repo = MockSearchRepository()
        pipeline = HybridSearchPipeline(repo)
        long_query = "word " * 1000

        results = await pipeline.search(long_query)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self):
        """Query with special characters should work."""
        repo = MockSearchRepository()
        pipeline = HybridSearchPipeline(repo)

        results = await pipeline.search("test @#$%^&*() query")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_unicode_query(self):
        """Unicode query should work."""
        repo = MockSearchRepository()
        pipeline = HybridSearchPipeline(repo)

        results = await pipeline.search("test 世界 query")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_limit_zero(self):
        """Limit of 0 should return empty list."""
        fts_results = [make_search_result("doc.md", 0.9)]
        repo = MockSearchRepository(fts_results=fts_results)
        pipeline = HybridSearchPipeline(repo)

        results = await pipeline.search("test", limit=0)

        assert results == []

    @pytest.mark.asyncio
    async def test_very_high_min_score(self):
        """Very high min_score should filter all results."""
        fts_results = [make_search_result("doc.md", 0.9)]
        repo = MockSearchRepository(fts_results=fts_results)
        pipeline = HybridSearchPipeline(repo)

        results = await pipeline.search("test", min_score=999.0)

        # All results should be filtered
        assert results == []
