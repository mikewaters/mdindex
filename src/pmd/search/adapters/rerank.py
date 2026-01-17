"""Reranking adapter.

Wraps DocumentReranker to implement the Reranker protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pmd.app.protocols import RerankScore

if TYPE_CHECKING:
    from pmd.core.types import RankedResult
    from pmd.llm.reranker import DocumentReranker


class LLMRerankerAdapter:
    """Adapter that wraps DocumentReranker for the Reranker protocol.

    This adapter wraps the existing DocumentReranker and converts its
    output to the RerankScore type expected by the pipeline.

    Example:
        >>> from pmd.llm.reranker import DocumentReranker
        >>> doc_reranker = DocumentReranker(llm_provider)
        >>> adapter = LLMRerankerAdapter(doc_reranker)
        >>> scores = await adapter.rerank("python tutorial", candidates)
    """

    def __init__(self, reranker: "DocumentReranker"):
        """Initialize with document reranker.

        Args:
            reranker: DocumentReranker instance.
        """
        self._reranker = reranker

    async def rerank(
        self,
        query: str,
        candidates: list["RankedResult"],
    ) -> list[RerankScore]:
        """Rerank candidate documents by relevance.

        Args:
            query: Search query.
            candidates: Candidate documents from retrieval/fusion.

        Returns:
            List of RerankScore objects in same order as candidates.
        """
        if not candidates:
            return []

        # Get raw rerank scores from the underlying reranker
        rerank_results = await self._reranker.get_rerank_scores(query, candidates)

        # Convert to RerankScore protocol type
        return [
            RerankScore(
                file=result.file,
                score=result.score,
                relevant=result.relevant,
                confidence=result.confidence,
            )
            for result in rerank_results
        ]
