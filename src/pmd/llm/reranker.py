"""Document reranking using LLM for relevance judgment."""

from ..core.types import RankedResult, RerankDocumentResult, RerankResult
from .base import LLMProvider


class DocumentReranker:
    """Reranks search results using LLM for relevance judgment."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize reranker.

        Args:
            llm_provider: LLM provider for ranking.
        """
        self.llm = llm_provider
        self.model = llm_provider.get_default_reranker_model()

    async def rerank(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """Rerank candidate documents by relevance to query.

        Args:
            query: Search query.
            candidates: Initial ranked candidates.
            top_k: Optional limit on results (default: return all).

        Returns:
            Reranked results sorted by relevance.
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

        # Create mapping of file -> rerank score
        rerank_map = {r.file: r for r in rerank_result.results}

        # Update candidates with rerank scores
        reranked = []
        for candidate in candidates:
            rerank_doc = rerank_map.get(candidate.file)

            if rerank_doc:
                # Blend RRF score with reranker score
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
