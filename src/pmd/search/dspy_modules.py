"""DSPy modules for search logic.

This module provides DSPy-based implementations of query expansion and
document reranking, replacing manual prompt engineering with DSPy's
declarative signatures and modules.

These modules work with any PMD LLM provider (MLX, LiteLLM, etc.) through
the PMDClient wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy
from loguru import logger

if TYPE_CHECKING:
    from pmd.llm import LLMProvider
    from pmd.core.types import RankedResult, RerankDocumentResult


# =============================================================================
# Query Expansion
# =============================================================================


class QueryExpansionSignature(dspy.Signature):
    """Generate diverse search query variations.

    Given an original search query, produce alternative phrasings that
    capture the same intent but use different wording, synonyms, or
    related concepts.
    """

    query: str = dspy.InputField(desc="The original search query")
    num_variations: int = dspy.InputField(desc="Number of variations to generate")
    variations: list[str] = dspy.OutputField(
        desc="List of alternative query phrasings"
    )


class DSPyQueryExpander(dspy.Module):
    """DSPy module for query expansion.

    Uses chain-of-thought reasoning to generate semantically similar
    query variations for improved search recall.

    Example:
        from pmd.llm import create_llm_provider, create_dspy_client
        from pmd.search.dspy_modules import DSPyQueryExpander

        provider = create_llm_provider(config)
        lm = create_dspy_client(provider)
        dspy.configure(lm=lm)

        expander = DSPyQueryExpander()
        result = expander(query="machine learning basics", num_variations=3)
        print(result.variations)  # ["ML fundamentals", "intro to machine learning", ...]
    """

    def __init__(self):
        """Initialize the query expander module."""
        super().__init__()
        self.expand = dspy.ChainOfThought(QueryExpansionSignature)

    def forward(self, query: str, num_variations: int = 2) -> dspy.Prediction:
        """Generate query variations.

        Args:
            query: Original search query.
            num_variations: Number of variations to generate.

        Returns:
            Prediction with 'variations' field containing alternative queries.
        """
        return self.expand(query=query, num_variations=num_variations)


class QueryExpanderAdapter:
    """Adapter wrapping DSPyQueryExpander for use in existing pipelines.

    This adapter provides the same interface as the original QueryExpander
    class, making it a drop-in replacement in the search pipeline.

    Example:
        from pmd.llm import create_llm_provider, create_dspy_client
        from pmd.search.dspy_modules import QueryExpanderAdapter

        provider = create_llm_provider(config)
        expander = QueryExpanderAdapter(provider)
        variations = await expander.expand("machine learning", num_variations=2)
    """

    def __init__(self, provider: "LLMProvider"):
        """Initialize the adapter.

        Args:
            provider: PMD LLM provider instance.
        """
        from pmd.llm import create_dspy_client

        self._provider = provider
        self._lm = create_dspy_client(provider)
        self._module = DSPyQueryExpander()

    async def expand(self, query: str, num_variations: int = 2) -> list[str]:
        """Generate query variations.

        Args:
            query: Original search query.
            num_variations: Number of variations to generate.

        Returns:
            List of query strings: [original, variation1, variation2, ...].
        """
        variations = [query]  # Always include original

        try:
            with dspy.context(lm=self._lm):
                result = self._module(query=query, num_variations=num_variations)

            if result and result.variations:
                # Handle both list and string responses
                if isinstance(result.variations, list):
                    variations.extend(result.variations[:num_variations])
                elif isinstance(result.variations, str):
                    # Parse newline-separated variations
                    parsed = [
                        v.strip()
                        for v in result.variations.split("\n")
                        if v.strip()
                    ]
                    variations.extend(parsed[:num_variations])

        except Exception as e:
            logger.warning(f"DSPy query expansion failed: {e}")

        return variations[:num_variations + 1]


# =============================================================================
# Document Reranking
# =============================================================================


class DocumentRelevanceSignature(dspy.Signature):
    """Judge whether a document is relevant to a query.

    Evaluate the document content against the search query and determine
    if the document would help answer the query.
    """

    query: str = dspy.InputField(desc="The search query")
    document: str = dspy.InputField(desc="The document content to evaluate")
    relevant: bool = dspy.OutputField(
        desc="True if document is relevant to query, False otherwise"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0"
    )


class DSPyReranker(dspy.Module):
    """DSPy module for document reranking.

    Evaluates document relevance using chain-of-thought reasoning to
    determine if documents match the search query.

    Example:
        from pmd.llm import create_llm_provider, create_dspy_client
        from pmd.search.dspy_modules import DSPyReranker

        provider = create_llm_provider(config)
        lm = create_dspy_client(provider)
        dspy.configure(lm=lm)

        reranker = DSPyReranker()
        result = reranker(query="python async", document="Guide to asyncio...")
        print(result.relevant, result.confidence)
    """

    def __init__(self):
        """Initialize the reranker module."""
        super().__init__()
        self.judge = dspy.ChainOfThought(DocumentRelevanceSignature)

    def forward(self, query: str, document: str) -> dspy.Prediction:
        """Judge document relevance.

        Args:
            query: Search query.
            document: Document content to evaluate.

        Returns:
            Prediction with 'relevant' and 'confidence' fields.
        """
        return self.judge(query=query, document=document)


class DocumentRerankerAdapter:
    """Adapter wrapping DSPyReranker for use in existing pipelines.

    This adapter provides the same interface as the original DocumentReranker
    class, making it a drop-in replacement in the search pipeline.

    Example:
        from pmd.llm import create_llm_provider
        from pmd.search.dspy_modules import DocumentRerankerAdapter

        provider = create_llm_provider(config)
        reranker = DocumentRerankerAdapter(provider)
        scores = await reranker.get_rerank_scores(query, candidates)
    """

    def __init__(self, provider: "LLMProvider"):
        """Initialize the adapter.

        Args:
            provider: PMD LLM provider instance.
        """
        from pmd.llm import create_dspy_client

        self._provider = provider
        self._lm = create_dspy_client(provider)
        self._module = DSPyReranker()

    async def get_rerank_scores(
        self,
        query: str,
        candidates: list["RankedResult"],
    ) -> list["RerankDocumentResult"]:
        """Get reranking scores for candidates.

        Args:
            query: Search query.
            candidates: Initial ranked candidates from retrieval.

        Returns:
            List of RerankDocumentResult with relevance scores.
        """
        from pmd.core.types import RerankDocumentResult

        if not candidates:
            return []

        results = []

        for candidate in candidates:
            try:
                with dspy.context(lm=self._lm):
                    prediction = self._module(
                        query=query,
                        document=candidate.body[:1000],  # Truncate for efficiency
                    )

                relevant = bool(prediction.relevant)
                confidence = float(prediction.confidence) if prediction.confidence else 0.5

                # Clamp confidence to [0, 1]
                confidence = max(0.0, min(1.0, confidence))

                # Convert to score: relevant docs get higher scores
                score = 0.5 + 0.5 * confidence if relevant else 0.5 * (1 - confidence)

                results.append(
                    RerankDocumentResult(
                        file=candidate.file,
                        relevant=relevant,
                        confidence=confidence,
                        score=score,
                        raw_token="yes" if relevant else "no",
                        logprob=None,
                    )
                )

            except Exception as e:
                logger.warning(f"DSPy reranking failed for {candidate.file}: {e}")
                # Default to neutral score on error
                results.append(
                    RerankDocumentResult(
                        file=candidate.file,
                        relevant=False,
                        confidence=0.5,
                        score=0.5,
                        raw_token="error",
                        logprob=None,
                    )
                )

        return results


# =============================================================================
# Factory Functions
# =============================================================================


def create_query_expander(
    provider: "LLMProvider",
    use_dspy: bool = True,
) -> Any:
    """Create a query expander instance.

    Args:
        provider: PMD LLM provider.
        use_dspy: If True, use DSPy-based expander. If False, use legacy.

    Returns:
        Query expander instance (DSPy adapter or legacy QueryExpander).
    """
    if use_dspy:
        return QueryExpanderAdapter(provider)
    else:
        from pmd.llm.query_expansion import QueryExpander
        return QueryExpander(provider)


def create_reranker(
    provider: "LLMProvider",
    use_dspy: bool = True,
) -> Any:
    """Create a document reranker instance.

    Args:
        provider: PMD LLM provider.
        use_dspy: If True, use DSPy-based reranker. If False, use legacy.

    Returns:
        Reranker instance (DSPy adapter or legacy DocumentReranker).
    """
    if use_dspy:
        return DocumentRerankerAdapter(provider)
    else:
        from pmd.llm.reranker import DocumentReranker
        return DocumentReranker(provider)
