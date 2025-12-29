"""Document reranking using LLM for relevance judgment.

This module provides LLM-based document reranking for search results. The reranker
evaluates each candidate document's relevance to the query and returns relevance
scores that can be blended with initial retrieval scores.

Typical usage in a search pipeline:

    from pmd.llm.reranker import DocumentReranker
    from pmd.search.scoring import blend_scores

    reranker = DocumentReranker(llm_provider)

    # Get raw relevance scores from the LLM
    rerank_scores = await reranker.get_rerank_scores(query, candidates)

    # Blend with RRF scores using position-aware weighting
    final_results = blend_scores(candidates, rerank_scores)

The separation of scoring and blending allows the pipeline to use the
position-aware blending strategy from `pmd.search.scoring.blend_scores`.

See Also:
    - `pmd.search.scoring.blend_scores`: Position-aware score blending
    - `pmd.search.pipeline.HybridSearchPipeline`: Full search pipeline
"""

from ..core.types import RankedResult, RerankDocumentResult, RerankResult
from .base import LLMProvider


class DocumentReranker:
    """Reranks search results using LLM for relevance judgment.

    The reranker evaluates candidate documents against the query using an LLM
    to determine relevance. It provides two main methods:

    - `get_rerank_scores()`: Returns raw LLM relevance scores for pipeline use
    - `rerank()`: Convenience method that applies simple 60/40 score blending

    For production pipelines, prefer `get_rerank_scores()` combined with
    `pmd.search.scoring.blend_scores()` for position-aware blending.

    Example:
        >>> reranker = DocumentReranker(llm_provider)
        >>> scores = await reranker.get_rerank_scores(query, candidates)
        >>> # Use with blend_scores for position-aware blending
        >>> from pmd.search.scoring import blend_scores
        >>> blended = blend_scores(candidates, scores)
    """

    def __init__(self, llm_provider: LLMProvider):
        """Initialize reranker.

        Args:
            llm_provider: LLM provider for ranking.
        """
        self.llm = llm_provider
        self.model = llm_provider.get_default_reranker_model()

    async def get_rerank_scores(
        self,
        query: str,
        candidates: list[RankedResult],
    ) -> list[RerankDocumentResult]:
        """Get raw reranking scores from LLM without blending.

        This method returns the raw LLM relevance scores for each candidate,
        suitable for use with `pmd.search.scoring.blend_scores()` which applies
        position-aware blending.

        This is the preferred method for search pipelines as it separates
        the concerns of scoring (LLM) and blending (pipeline).

        Args:
            query: Search query.
            candidates: Initial ranked candidates from retrieval.

        Returns:
            List of RerankDocumentResult with relevance scores.
            Results maintain the same order as candidates.

        Example:
            >>> scores = await reranker.get_rerank_scores(query, candidates)
            >>> # Apply position-aware blending
            >>> from pmd.search.scoring import blend_scores
            >>> final = blend_scores(candidates, scores)

        See Also:
            - `pmd.search.scoring.blend_scores`: Position-aware score blending
        """
        if not candidates:
            return []

        # Prepare documents for reranking
        docs = [
            {
                "file": c.file,
                "body": c.body,
            }
            for c in candidates
        ]

        # Get reranking scores from LLM
        rerank_result = await self.llm.rerank(query, docs, model=self.model)

        return rerank_result.results

    async def rerank(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """Rerank candidate documents with simple score blending.

        This convenience method gets LLM relevance scores and applies a simple
        60% RRF / 40% reranker blending. For position-aware blending, use
        `get_rerank_scores()` with `pmd.search.scoring.blend_scores()`.

        Args:
            query: Search query.
            candidates: Initial ranked candidates.
            top_k: Optional limit on results (default: return all).

        Returns:
            Reranked results sorted by blended relevance score.

        Note:
            This method uses position-independent 60/40 blending.
            For production pipelines, consider using `get_rerank_scores()`
            with `blend_scores()` for position-aware blending that trusts
            top results more and relies on the reranker for borderline cases.

        See Also:
            - `get_rerank_scores`: Raw scores for custom blending
            - `pmd.search.scoring.blend_scores`: Position-aware blending
        """
        if not candidates:
            return []

        # Get raw rerank scores
        rerank_results = await self.get_rerank_scores(query, candidates)

        # Create mapping of file -> rerank score
        rerank_map = {r.file: r for r in rerank_results}

        # Update candidates with rerank scores using simple blending
        reranked = []
        for candidate in candidates:
            rerank_doc = rerank_map.get(candidate.file)

            if rerank_doc:
                # Simple 60/40 blend (position-independent)
                final_score = self._blend_scores(candidate.score, rerank_doc.score)

                reranked.append(
                    RankedResult(
                        file=candidate.file,
                        display_path=candidate.display_path,
                        title=candidate.title,
                        body=candidate.body,
                        score=final_score,
                        fts_score=candidate.fts_score,
                        vec_score=candidate.vec_score,
                        rerank_score=rerank_doc.score,
                    )
                )
            else:
                # Keep original if reranking failed
                reranked.append(candidate)

        # Sort by reranked score
        reranked.sort(key=lambda r: r.score, reverse=True)

        # Return top K if specified
        if top_k:
            return reranked[:top_k]

        return reranked

    async def score_document(
        self,
        query: str,
        document: RankedResult,
    ) -> float:
        """Score a single document's relevance to query.

        Args:
            query: Search query.
            document: Document to score.

        Returns:
            Relevance score (0-1).
        """
        docs = [{"file": document.file, "body": document.body}]

        result = await self.llm.rerank(query, docs, model=self.model)

        if result.results:
            return result.results[0].score

        return 0.5  # Default neutral score on error

    @staticmethod
    def _blend_scores(rrf_score: float, rerank_score: float) -> float:
        """Blend RRF and reranker scores.

        Uses position-independent weighting: 60% RRF, 40% reranker.
        (Position-aware blending is handled in the search pipeline)

        Args:
            rrf_score: RRF fusion score.
            rerank_score: LLM reranker score.

        Returns:
            Blended score.
        """
        return 0.6 * rrf_score + 0.4 * rerank_score

    @staticmethod
    def calculate_confidence(rerank_results: list[RerankDocumentResult]) -> float:
        """Calculate confidence of reranking results.

        Based on how many documents were rated as relevant.

        Args:
            rerank_results: Results from reranking.

        Returns:
            Confidence score (0-1).
        """
        if not rerank_results:
            return 0.5

        relevant_count = sum(1 for r in rerank_results if r.relevant)
        return min(1.0, relevant_count / len(rerank_results))
