"""DSPy retriever wrapping PMD's HybridSearchPipeline.

This module provides a DSPy-compatible retriever that uses PMD's
hybrid search (FTS + vector) as a retrieval backend for RAG applications.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dspy
from loguru import logger

if TYPE_CHECKING:
    from pmd.search.pipeline import HybridSearchPipeline
    from pmd.core.types import RankedResult


@dataclass
class RetrievedPassage:
    """A retrieved passage from PMD search.

    Attributes:
        text: The passage text content.
        score: Relevance score.
        file: Source file path.
        title: Document title.
    """

    text: str
    score: float
    file: str
    title: str


class PMDRetriever(dspy.Retrieve):
    """DSPy retriever backed by PMD's HybridSearchPipeline.

    This retriever allows DSPy modules (like RAG) to use PMD's hybrid
    search capabilities (FTS + vector similarity) for context retrieval.

    Example:
        from pmd.search.pipeline import HybridSearchPipeline
        from pmd.search.dspy_retriever import PMDRetriever

        pipeline = HybridSearchPipeline(...)  # Setup your pipeline
        retriever = PMDRetriever(pipeline, k=5)

        # Use in a DSPy module
        class RAG(dspy.Module):
            def __init__(self, retriever):
                self.retriever = retriever
                self.generate = dspy.ChainOfThought("context, question -> answer")

            def forward(self, question):
                context = self.retriever(question)
                return self.generate(context=context, question=question)
    """

    def __init__(
        self,
        pipeline: "HybridSearchPipeline",
        k: int = 5,
        use_vector: bool = True,
        use_rerank: bool = False,
    ):
        """Initialize PMDRetriever.

        Args:
            pipeline: PMD HybridSearchPipeline instance.
            k: Number of passages to retrieve (default: 5).
            use_vector: Whether to use vector search (default: True).
            use_rerank: Whether to rerank results (default: False for speed).
        """
        super().__init__(k=k)
        self._pipeline = pipeline
        self._k = k
        self._use_vector = use_vector
        self._use_rerank = use_rerank

    def forward(
        self,
        query: str | list[str],
        k: int | None = None,
        **kwargs: Any,
    ) -> dspy.Prediction:
        """Retrieve relevant passages for a query.

        Args:
            query: Search query string or list of queries.
            k: Optional override for number of passages to retrieve.
            **kwargs: Additional arguments (passed to pipeline).

        Returns:
            Prediction with 'passages' field containing retrieved text.
        """
        # Handle list of queries by joining
        if isinstance(query, list):
            query = " ".join(query)

        k = k or self._k

        # Run async search in sync context
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._search(query, k),
                )
                results = future.result()
        except RuntimeError:
            results = asyncio.run(self._search(query, k))

        # Extract passages for DSPy
        passages = [r.text for r in results]

        return dspy.Prediction(passages=passages)

    async def _search(self, query: str, k: int) -> list[RetrievedPassage]:
        """Perform async search using the pipeline.

        Args:
            query: Search query.
            k: Number of results.

        Returns:
            List of RetrievedPassage objects.
        """
        try:
            results = await self._pipeline.search(
                query=query,
                limit=k,
                use_expansion=False,  # Don't expand for retrieval speed
                use_vector=self._use_vector,
                use_rerank=self._use_rerank,
            )

            return [
                RetrievedPassage(
                    text=r.body,
                    score=r.score,
                    file=r.file,
                    title=r.title,
                )
                for r in results
            ]

        except Exception as e:
            logger.warning(f"PMD retrieval failed: {e}")
            return []

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> list[RetrievedPassage]:
        """Synchronous retrieve method for direct use.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List of RetrievedPassage objects with full metadata.
        """
        k = k or self._k

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._search(query, k),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self._search(query, k))


class PMDRetrieverAsync:
    """Async version of PMDRetriever for use in async contexts.

    This retriever is for use in fully async applications where you
    don't want the overhead of thread pool execution.

    Example:
        retriever = PMDRetrieverAsync(pipeline, k=5)
        passages = await retriever.retrieve("machine learning basics")
    """

    def __init__(
        self,
        pipeline: "HybridSearchPipeline",
        k: int = 5,
        use_vector: bool = True,
        use_rerank: bool = False,
    ):
        """Initialize async retriever.

        Args:
            pipeline: PMD HybridSearchPipeline instance.
            k: Number of passages to retrieve.
            use_vector: Whether to use vector search.
            use_rerank: Whether to rerank results.
        """
        self._pipeline = pipeline
        self._k = k
        self._use_vector = use_vector
        self._use_rerank = use_rerank

    async def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> list[RetrievedPassage]:
        """Retrieve relevant passages asynchronously.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List of RetrievedPassage objects.
        """
        k = k or self._k

        try:
            results = await self._pipeline.search(
                query=query,
                limit=k,
                use_expansion=False,
                use_vector=self._use_vector,
                use_rerank=self._use_rerank,
            )

            return [
                RetrievedPassage(
                    text=r.body,
                    score=r.score,
                    file=r.file,
                    title=r.title,
                )
                for r in results
            ]

        except Exception as e:
            logger.warning(f"PMD async retrieval failed: {e}")
            return []

    async def __call__(
        self,
        query: str,
        k: int | None = None,
    ) -> list[str]:
        """Retrieve passages and return just the text.

        Args:
            query: Search query.
            k: Number of results.

        Returns:
            List of passage texts.
        """
        results = await self.retrieve(query, k)
        return [r.text for r in results]


def create_retriever(
    pipeline: "HybridSearchPipeline",
    k: int = 5,
    use_vector: bool = True,
    use_rerank: bool = False,
    async_mode: bool = False,
) -> PMDRetriever | PMDRetrieverAsync:
    """Create a PMD retriever for DSPy.

    Args:
        pipeline: PMD HybridSearchPipeline instance.
        k: Number of passages to retrieve.
        use_vector: Whether to use vector search.
        use_rerank: Whether to rerank results.
        async_mode: If True, return async retriever.

    Returns:
        PMDRetriever or PMDRetrieverAsync instance.
    """
    if async_mode:
        return PMDRetrieverAsync(
            pipeline=pipeline,
            k=k,
            use_vector=use_vector,
            use_rerank=use_rerank,
        )
    return PMDRetriever(
        pipeline=pipeline,
        k=k,
        use_vector=use_vector,
        use_rerank=use_rerank,
    )
