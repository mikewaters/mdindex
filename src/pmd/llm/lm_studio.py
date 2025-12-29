"""LM Studio LLM provider implementation."""

import time

import httpx
from loguru import logger

from ..core.config import LMStudioConfig
from ..core.types import EmbeddingResult, RerankDocumentResult, RerankResult
from .base import LLMProvider


class LMStudioProvider(LLMProvider):
    """LM Studio local LLM provider (OpenAI-compatible API)."""

    def __init__(self, config: LMStudioConfig):
        """Initialize LM Studio provider.

        Args:
            config: LMStudioConfig with connection details.
        """
        self.config = config
        self._client = httpx.AsyncClient(timeout=config.timeout)
        logger.debug(f"LM Studio provider initialized: base_url={config.base_url}")

    async def embed(
        self,
        text: str,
        model: str | None = None,
        is_query: bool = False,
    ) -> EmbeddingResult | None:
        """Generate embeddings using LM Studio.

        Args:
            text: Text to embed.
            model: Optional model override.
            is_query: If True, embed as query. If False, embed as document.

        Returns:
            EmbeddingResult or None on failure.
        """
        model = model or self.config.embedding_model
        text_preview = text[:50] + "..." if len(text) > 50 else text
        logger.debug(f"Embedding text ({len(text)} chars, is_query={is_query}): {text_preview!r}")
        start_time = time.perf_counter()

        try:
            response = await self._client.post(
                f"{self.config.base_url}/v1/embeddings",
                json={
                    "model": model,
                    "input": text,
                },
            )

            elapsed = (time.perf_counter() - start_time) * 1000
            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]
                logger.debug(f"Embedding generated: dim={len(embedding)}, {elapsed:.1f}ms")
                return EmbeddingResult(embedding=embedding, model=model)

            logger.warning(f"Embedding failed: HTTP {response.status_code}, {elapsed:.1f}ms")
            return None
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"Embedding failed after {elapsed:.1f}ms: {e}")
            return None

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str | None:
        """Generate text using LM Studio (OpenAI-compatible).

        Args:
            prompt: Prompt text.
            model: Optional model override.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text or None on failure.
        """
        model = model or self.config.expansion_model
        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        logger.debug(f"Generating text (max_tokens={max_tokens}, temp={temperature}): {prompt_preview!r}")
        start_time = time.perf_counter()

        try:
            response = await self._client.post(
                f"{self.config.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )

            elapsed = (time.perf_counter() - start_time) * 1000
            if response.status_code == 200:
                data = response.json()
                result = data["choices"][0]["message"]["content"]
                resp_preview = result[:50] + "..." if result and len(result) > 50 else result
                logger.debug(f"Generated in {elapsed:.1f}ms: {resp_preview!r}")
                return result

            logger.warning(f"Generation failed: HTTP {response.status_code}, {elapsed:.1f}ms")
            return None
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"Generation failed after {elapsed:.1f}ms: {e}")
            return None

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        model: str | None = None,
    ) -> RerankResult:
        """Rerank documents using LM Studio.

        Args:
            query: Search query.
            documents: List of dicts with 'file' and 'body' keys.
            model: Optional model override.

        Returns:
            RerankResult with relevance scores.
        """
        model = model or self.config.reranker_model
        logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]!r}...")
        start_time = time.perf_counter()
        results = []

        for i, doc in enumerate(documents):
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
                    f"{self.config.base_url}/v1/chat/completions",
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
                    logger.debug(f"  [{i+1}/{len(documents)}] {doc.get('file', '')}: {answer} (score={score:.2f})")
                else:
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
                    logger.debug(f"  [{i+1}/{len(documents)}] {doc.get('file', '')}: HTTP {response.status_code}, neutral")
            except Exception as e:
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
                logger.warning(f"  [{i+1}/{len(documents)}] {doc.get('file', '')}: error: {e}")

        elapsed = (time.perf_counter() - start_time) * 1000
        relevant_count = sum(1 for r in results if r.relevant)
        logger.debug(f"Reranking complete: {relevant_count}/{len(results)} relevant, {elapsed:.1f}ms")

        return RerankResult(results=results, model=model)

    async def model_exists(self, model: str) -> bool:
        """Check if model is available in LM Studio.

        Args:
            model: Model name to check.

        Returns:
            True if model is available.
        """
        try:
            response = await self._client.get(
                f"{self.config.base_url}/v1/models",
            )

            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                return model in models

            return False
        except Exception:
            return False

    async def is_available(self) -> bool:
        """Check if LM Studio is available.

        Returns:
            True if service is reachable.
        """
        logger.debug(f"Checking LM Studio availability at {self.config.base_url}")
        try:
            response = await self._client.get(
                f"{self.config.base_url}/v1/models",
                timeout=5.0,
            )
            available = response.status_code == 200
            logger.debug(f"LM Studio available: {available}")
            return available
        except Exception as e:
            logger.debug(f"LM Studio not available: {e}")
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
