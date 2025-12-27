"""Tests for DocumentReranker."""

import pytest

from pmd.llm.reranker import DocumentReranker
from pmd.core.types import RankedResult, RerankDocumentResult, RerankResult

from .conftest import MockLLMProvider, make_rerank_document_result


def make_rerank_result_from_relevances(
    filepaths: list[str], relevances: list[bool]
) -> RerankResult:
    """Create a RerankResult from filepaths and relevance flags."""
    results = [
        make_rerank_document_result(fp, rel) for fp, rel in zip(filepaths, relevances)
    ]
    return RerankResult(results=results, model="test-model")


@pytest.fixture
def reranker(mock_llm_provider: MockLLMProvider) -> DocumentReranker:
    """Create document reranker for testing."""
    return DocumentReranker(mock_llm_provider)


class TestDocumentRerankerInit:
    """Tests for DocumentReranker initialization."""

    def test_stores_llm_provider(self, reranker, mock_llm_provider):
        """Should store LLM provider."""
        assert reranker.llm == mock_llm_provider

    def test_gets_default_model(self, reranker):
        """Should get default reranker model from provider."""
        assert reranker.model == "mock-reranker-model"


class TestDocumentRerankerRerank:
    """Tests for DocumentReranker.rerank method."""

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self, reranker):
        """rerank should return empty list for empty input."""
        result = await reranker.rerank("query", [])

        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_returns_ranked_results(
        self, sample_ranked_results: list[RankedResult]
    ):
        """rerank should return RankedResult objects."""
        rerank_result = make_rerank_result_from_relevances(
            ["doc1.md", "doc2.md", "doc3.md"], [True, False, True]
        )
        mock_provider = MockLLMProvider(rerank_result=rerank_result)
        reranker = DocumentReranker(mock_provider)

        result = await reranker.rerank("query", sample_ranked_results)

        assert len(result) == 3
        assert all(isinstance(r, RankedResult) for r in result)

    @pytest.mark.asyncio
    async def test_rerank_calls_llm(
        self, sample_ranked_results: list[RankedResult], mock_llm_provider
    ):
        """rerank should call LLM with documents."""
        reranker = DocumentReranker(mock_llm_provider)

        await reranker.rerank("search query", sample_ranked_results)

        assert len(mock_llm_provider.rerank_calls) == 1
        query, docs, model = mock_llm_provider.rerank_calls[0]
        assert query == "search query"
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_rerank_updates_scores(self, sample_ranked_results: list[RankedResult]):
        """rerank should update result scores based on relevance."""
        rerank_result = make_rerank_result_from_relevances(
            ["doc1.md", "doc2.md", "doc3.md"], [True, True, False]
        )
        mock_provider = MockLLMProvider(rerank_result=rerank_result)
        reranker = DocumentReranker(mock_provider)

        result = await reranker.rerank("query", sample_ranked_results)

        # Relevant docs should have higher scores
        doc1 = next(r for r in result if r.file == "doc1.md")
        doc3 = next(r for r in result if r.file == "doc3.md")
        assert doc1.score > doc3.score

    @pytest.mark.asyncio
    async def test_rerank_sets_rerank_score(self, sample_ranked_results: list[RankedResult]):
        """rerank should set rerank_score attribute."""
        rerank_result = make_rerank_result_from_relevances(
            ["doc1.md", "doc2.md", "doc3.md"], [True, False, True]
        )
        mock_provider = MockLLMProvider(rerank_result=rerank_result)
        reranker = DocumentReranker(mock_provider)

        result = await reranker.rerank("query", sample_ranked_results)

        for r in result:
            assert r.rerank_score is not None

    @pytest.mark.asyncio
    async def test_rerank_sorts_by_score(self, sample_ranked_results: list[RankedResult]):
        """rerank should sort results by blended score."""
        rerank_result = make_rerank_result_from_relevances(
            ["doc1.md", "doc2.md", "doc3.md"], [False, True, True]
        )
        mock_provider = MockLLMProvider(rerank_result=rerank_result)
        reranker = DocumentReranker(mock_provider)

        result = await reranker.rerank("query", sample_ranked_results)

        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self, sample_ranked_results: list[RankedResult]):
        """rerank should respect top_k parameter."""
        rerank_result = make_rerank_result_from_relevances(
            ["doc1.md", "doc2.md", "doc3.md"], [True, True, True]
        )
        mock_provider = MockLLMProvider(rerank_result=rerank_result)
        reranker = DocumentReranker(mock_provider)

        result = await reranker.rerank("query", sample_ranked_results, top_k=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_rerank_preserves_metadata(self, sample_ranked_results: list[RankedResult]):
        """rerank should preserve document metadata."""
        rerank_result = make_rerank_result_from_relevances(
            ["doc1.md", "doc2.md", "doc3.md"], [True, True, True]
        )
        mock_provider = MockLLMProvider(rerank_result=rerank_result)
        reranker = DocumentReranker(mock_provider)

        result = await reranker.rerank("query", sample_ranked_results)

        doc1 = next(r for r in result if r.file == "doc1.md")
        assert doc1.title == "Document 1"
        assert "Python" in doc1.body
        assert doc1.fts_score == 0.85

    @pytest.mark.asyncio
    async def test_rerank_handles_missing_rerank_result(
        self, sample_ranked_results: list[RankedResult]
    ):
        """rerank should handle documents not in rerank results."""
        # Only return results for 2 of 3 documents
        rerank_result = RerankResult(
            results=[
                make_rerank_document_result("doc1.md", True),
                make_rerank_document_result("doc2.md", False),
            ],
            model="test-model",
        )
        mock_provider = MockLLMProvider(rerank_result=rerank_result)
        reranker = DocumentReranker(mock_provider)

        result = await reranker.rerank("query", sample_ranked_results)

        # All 3 should still be in results
        assert len(result) == 3


class TestDocumentRerankerScoreDocument:
    """Tests for DocumentReranker.score_document method."""

    @pytest.mark.asyncio
    async def test_score_document_returns_score(self, sample_ranked_results):
        """score_document should return relevance score."""
        rerank_result = make_rerank_result_from_relevances(["doc1.md"], [True])
        mock_provider = MockLLMProvider(rerank_result=rerank_result)
        reranker = DocumentReranker(mock_provider)

        score = await reranker.score_document("query", sample_ranked_results[0])

        assert isinstance(score, float)
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_score_document_returns_default_on_empty(self, sample_ranked_results):
        """score_document should return 0.5 when no results."""
        rerank_result = RerankResult(results=[], model="test")
        mock_provider = MockLLMProvider(rerank_result=rerank_result)
        reranker = DocumentReranker(mock_provider)

        score = await reranker.score_document("query", sample_ranked_results[0])

        assert score == 0.5


class TestDocumentRerankerBlendScores:
    """Tests for _blend_scores static method."""

    def test_blend_scores_weighted(self):
        """_blend_scores should use 60/40 weighting."""
        result = DocumentReranker._blend_scores(1.0, 0.0)

        # 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        assert result == pytest.approx(0.6)

    def test_blend_scores_equal(self):
        """_blend_scores with equal inputs should return same value."""
        result = DocumentReranker._blend_scores(0.5, 0.5)

        assert result == pytest.approx(0.5)

    def test_blend_scores_high_rerank(self):
        """_blend_scores with high rerank should boost score."""
        result = DocumentReranker._blend_scores(0.5, 1.0)

        # 0.6 * 0.5 + 0.4 * 1.0 = 0.7
        assert result == pytest.approx(0.7)


class TestDocumentRerankerCalculateConfidence:
    """Tests for calculate_confidence static method."""

    def test_calculate_confidence_all_relevant(self):
        """Confidence should be high when all relevant."""
        results = [
            make_rerank_document_result("doc1.md", True),
            make_rerank_document_result("doc2.md", True),
        ]

        confidence = DocumentReranker.calculate_confidence(results)

        assert confidence == 1.0

    def test_calculate_confidence_none_relevant(self):
        """Confidence should be low when none relevant."""
        results = [
            make_rerank_document_result("doc1.md", False),
            make_rerank_document_result("doc2.md", False),
        ]

        confidence = DocumentReranker.calculate_confidence(results)

        assert confidence == 0.0

    def test_calculate_confidence_mixed(self):
        """Confidence should reflect proportion relevant."""
        results = [
            make_rerank_document_result("doc1.md", True),
            make_rerank_document_result("doc2.md", False),
            make_rerank_document_result("doc3.md", True),
            make_rerank_document_result("doc4.md", False),
        ]

        confidence = DocumentReranker.calculate_confidence(results)

        assert confidence == pytest.approx(0.5)

    def test_calculate_confidence_empty(self):
        """Confidence should be 0.5 for empty results."""
        confidence = DocumentReranker.calculate_confidence([])

        assert confidence == 0.5


class TestDocumentRerankerIntegration:
    """Integration tests for DocumentReranker."""

    @pytest.mark.asyncio
    async def test_full_reranking_flow(self):
        """Test complete reranking with realistic scenario."""
        candidates = [
            RankedResult(
                file="relevant.md",
                display_path="relevant.md",
                title="Relevant Document",
                body="This document is about Python programming.",
                score=0.7,
            ),
            RankedResult(
                file="irrelevant.md",
                display_path="irrelevant.md",
                title="Irrelevant Document",
                body="This document is about cooking recipes.",
                score=0.8,
            ),
        ]

        rerank_result = make_rerank_result_from_relevances(
            ["relevant.md", "irrelevant.md"], [True, False]
        )
        mock_provider = MockLLMProvider(rerank_result=rerank_result)
        reranker = DocumentReranker(mock_provider)

        result = await reranker.rerank("Python programming", candidates)

        # Relevant doc should now be ranked first
        assert result[0].file == "relevant.md"
        assert result[0].rerank_score > result[1].rerank_score
