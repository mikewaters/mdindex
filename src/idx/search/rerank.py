"""idx.search.rerank - LLM-as-judge reranking for search results.

Provides LLM-based reranking using LlamaIndex's LLMRerank node postprocessor.
Supports configurable LLM providers with lazy initialization.

Example usage:
    from idx.search.rerank import Reranker
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    reranker = Reranker()

    # Rerank nodes
    reranked_nodes = reranker.rerank(
        nodes=nodes_with_scores,
        query="machine learning concepts",
        top_n=5,
    )
"""

from typing import TYPE_CHECKING

from idx.core.logging import get_logger

if TYPE_CHECKING:
    from llama_index.core.postprocessor import LLMRerank
    from llama_index.core.schema import NodeWithScore
    from llama_index.llms.openai import OpenAI as OpenAILLM

__all__ = ["Reranker"]

logger = get_logger(__name__)


class Reranker:
    """LLM-as-judge reranker using LlamaIndex LLMRerank postprocessor.

    Wraps LlamaIndex's LLMRerank node postprocessor with lazy initialization.
    The LLM provider is configurable, with OpenAI as the default.

    The reranker applies LLM-based relevance judgment to a list of candidate
    nodes, reordering them by predicted relevance to the query. This is
    typically applied after fusion (RRF) and before final result limiting.

    Attributes:
        _llm: Cached LLM instance, initialized lazily.
        _reranker: Cached LLMRerank postprocessor, initialized lazily.
        _choice_batch_size: Number of nodes to evaluate per LLM call.
        _default_top_n: Default number of top results to return.

    Example:
        reranker = Reranker(choice_batch_size=5)

        # Rerank after hybrid search
        reranked = reranker.rerank(
            nodes=hybrid_results,
            query="async python patterns",
            top_n=10,
        )
    """

    def __init__(
        self,
        choice_batch_size: int = 5,
        default_top_n: int = 10,
        llm: "OpenAILLM | None" = None,
    ) -> None:
        """Initialize the Reranker.

        Args:
            choice_batch_size: Number of nodes to evaluate per LLM call.
                Smaller batches may produce more accurate rankings but
                require more LLM calls. Defaults to 5.
            default_top_n: Default number of top results to return when
                top_n is not specified in rerank(). Defaults to 10.
            llm: Optional pre-configured LLM instance. If None, creates
                an OpenAI LLM on first use.
        """
        self._choice_batch_size = choice_batch_size
        self._default_top_n = default_top_n
        self._llm = llm
        self._reranker: "LLMRerank | None" = None

    def _ensure_llm(self) -> "OpenAILLM":
        """Ensure the LLM is initialized.

        Lazily initializes an OpenAI LLM if one was not provided.

        Returns:
            Configured LLM instance.

        Raises:
            ImportError: If llama_index.llms.openai is not installed.
        """
        if self._llm is None:
            from llama_index.llms.openai import OpenAI as OpenAILLM

            logger.debug("Initializing OpenAI LLM for reranking")
            # Use a fast, capable model for reranking
            self._llm = OpenAILLM(model="gpt-4o-mini", temperature=0.0)
            logger.info("OpenAI LLM initialized for reranking")
        return self._llm

    def _ensure_reranker(self, top_n: int) -> "LLMRerank":
        """Ensure the LLMRerank postprocessor is initialized.

        Creates or updates the reranker with the specified top_n.
        The reranker is cached and reused if top_n matches.

        Args:
            top_n: Number of top results to return.

        Returns:
            Configured LLMRerank postprocessor.
        """
        # Recreate if top_n changed or not initialized
        if self._reranker is None or self._reranker.top_n != top_n:
            from llama_index.core.postprocessor import LLMRerank

            llm = self._ensure_llm()
            logger.debug(
                f"Creating LLMRerank: top_n={top_n}, choice_batch_size={self._choice_batch_size}"
            )
            self._reranker = LLMRerank(
                llm=llm,
                choice_batch_size=self._choice_batch_size,
                top_n=top_n,
            )
        return self._reranker

    def rerank(
        self,
        nodes: list["NodeWithScore"],
        query: str,
        top_n: int | None = None,
    ) -> list["NodeWithScore"]:
        """Rerank nodes using LLM-as-judge relevance scoring.

        Applies LLMRerank postprocessor to reorder nodes by their
        predicted relevance to the query. The LLM evaluates batches
        of nodes and assigns relevance scores.

        Args:
            nodes: List of NodeWithScore objects to rerank. Should be
                bounded by rerank_candidates (max chunks to send to LLM).
            query: The search query string for relevance evaluation.
            top_n: Number of top results to return. If None, uses
                default_top_n from initialization.

        Returns:
            List of NodeWithScore objects reordered by LLM relevance,
            limited to top_n results. Scores are updated to reflect
            the LLM's relevance assessment.

        Example:
            # Rerank top 20 candidates to get top 5
            reranked = reranker.rerank(
                nodes=candidates[:20],
                query="python async patterns",
                top_n=5,
            )
        """
        from llama_index.core.schema import QueryBundle

        if not nodes:
            logger.debug("No nodes to rerank, returning empty list")
            return []

        effective_top_n = top_n if top_n is not None else self._default_top_n

        logger.debug(
            f"Reranking {len(nodes)} nodes with query '{query[:50]}...', top_n={effective_top_n}"
        )

        reranker = self._ensure_reranker(effective_top_n)

        # Create query bundle for reranker
        query_bundle = QueryBundle(query_str=query)

        # Apply reranking
        reranked_nodes = reranker.postprocess_nodes(
            nodes=nodes,
            query_bundle=query_bundle,
        )

        logger.debug(f"Reranking complete: {len(nodes)} -> {len(reranked_nodes)} nodes")

        return reranked_nodes
