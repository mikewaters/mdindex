"""OpenRouter LLM provider implementation."""

import httpx

from ..core.config import OpenRouterConfig
from ..core.exceptions import OllamaConnectionError
from ..core.types import EmbeddingResult, RerankDocumentResult, RerankResult
from .base import LLMProvider


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider for accessing multiple models."""

    def __init__(self, config: OpenRouterConfig):
        """Initialize OpenRouter provider.

        Args:
            config: OpenRouterConfig with API credentials.

        Raises:
            ValueError: If API key is not configured.
        """
        if not config.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.config = config
        self._client = httpx.AsyncClient(
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "HTTP-Referer": "https://github.com/steveyegge/pmd",
                "X-Title": "PMD - Python Markdown Search",
            },
        )

    async def embed(
        self,
        text: str,
        model: str | None = None,
        is_query: bool = False,
    ) -> EmbeddingResult | None:
        """Generate embeddings using OpenRouter.

        Args:
            text: Text to embed.
            model: Optional model override.
            is_query: If True, embed as query. If False, embed as document.

        Returns:
            EmbeddingResult or None on failure.
        """
        model = model or self.config.embedding_model

        try:
            response = await self._client.post(
                f"{self.config.base_url}/embeddings",
                json={
                    "model": model,
                    "input": text,
                },
            )

            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]
                return EmbeddingResult(embedding=embedding, model=model)

            return None
        except Exception:
            return None

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str | None:
        """Generate text using OpenRouter.

        Args:
            prompt: Prompt text.
            model: Optional model override.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text or None on failure.
        """
        model = model or self.config.expansion_model

        try:
            response = await self._client.post(
                f"{self.config.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]

            return None
        except Exception:
            return None

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        model: str | None = None,
    ) -> RerankResult:
        """Rerank documents using OpenRouter.

        Args:
            query: Search query.
            documents: List of dicts with 'file' and 'body' keys.
            model: Optional model override.

        Returns:
            RerankResult with relevance scores.
        """
        model = model or self.config.reranker_model
        results = []

        for doc in documents:
            system_prompt = (
                "You are a relevance judge. Given a query and a document, "
                "respond with ONLY 'Yes' if the document is relevant to the query, "
                "or 'No' if it is not relevant."
            )

            user_prompt = (
                f"Query: {query}\n\nDocument: {doc.get('body', '')[:1000]}"
            )

            try:
                response = await self._client.post(
                    f"{self.config.base_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["choices"][0]["message"]["content"].strip().lower()

                    relevant = answer.startswith("yes")
                    confidence = 0.9 if relevant else 0.1
                    score = 0.5 + 0.5 * confidence if relevant else 0.5 * (1 - confidence)

                    results.append(
                        RerankDocumentResult(
                            file=doc.get("file", ""),
                            relevant=relevant,
                            confidence=confidence,
                            score=score,
                            raw_token=answer,
                            logprob=None,
                        )
                    )
            except Exception:
                # Default to neutral score on error
                results.append(
                    RerankDocumentResult(
                        file=doc.get("file", ""),
                        relevant=False,
                        confidence=0.5,
                        score=0.5,
                        raw_token="error",
                        logprob=None,
                    )
                )

        return RerankResult(results=results, model=model)

    async def model_exists(self, model: str) -> bool:
        """Check if model is available via OpenRouter.

        Args:
            model: Model name to check.

        Returns:
            True if model is available.
        """
        try:
            response = await self._client.get(
                f"{self.config.base_url}/models",
            )

            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                return model in models

            return False
        except Exception:
            return False

    async def is_available(self) -> bool:
        """Check if OpenRouter API is available.

        Returns:
            True if service is reachable.
        """
        try:
            response = await self._client.get(
                f"{self.config.base_url}/models",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_default_embedding_model(self) -> str:
        """Get default embedding model."""
        return self.config.embedding_model

    def get_default_expansion_model(self) -> str:
        """Get default expansion model."""
        return self.config.expansion_model

    def get_default_reranker_model(self) -> str:
        """Get default reranker model."""
        return self.config.reranker_model

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
