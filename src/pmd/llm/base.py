"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod

from ..core.types import EmbeddingResult, RerankResult


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def embed(
        self,
        text: str,
        model: str | None = None,
        is_query: bool = False,
    ) -> EmbeddingResult | None:
        """Generate embeddings for text.

        Args:
            text: Text to embed.
            model: Optional model override. Uses default if None.
            is_query: If True, embed as query. If False, embed as document.

        Returns:
            EmbeddingResult with embedding vector and model name, or None on failure.
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str | None:
        """Generate text completion.

        Args:
            prompt: Prompt text.
            model: Optional model override. Uses default if None.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0-1).

        Returns:
            Generated text or None on failure.
        """
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[dict],
        model: str | None = None,
    ) -> RerankResult:
        """Rerank documents by relevance to query.

        Args:
            query: Search query.
            documents: List of dicts with 'file' and 'body' keys.
            model: Optional model override. Uses default if None.

        Returns:
            RerankResult with relevance scores.
        """
        pass

    @abstractmethod
    async def model_exists(self, model: str) -> bool:
        """Check if model is available.

        Args:
            model: Model name to check.

        Returns:
            True if model is available, False otherwise.
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the LLM service is available.

        Returns:
            True if service is reachable, False otherwise.
        """
        pass

    @abstractmethod
    def get_default_embedding_model(self) -> str:
        """Get default embedding model name.

        Returns:
            Model name.
        """
        pass

    @abstractmethod
    def get_default_expansion_model(self) -> str:
        """Get default query expansion model name.

        Returns:
            Model name.
        """
        pass

    @abstractmethod
    def get_default_reranker_model(self) -> str:
        """Get default reranker model name.

        Returns:
            Model name.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the provider and release any resources.

        Should be called when the provider is no longer needed.
        """
        pass
