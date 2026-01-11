"""LiteLLM universal LLM provider implementation.

LiteLLM provides a unified interface to 100+ LLM providers including
OpenAI, Anthropic, Google, Azure, Cohere, and many more.

See: https://docs.litellm.ai/
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import litellm
from loguru import logger

from ..core.types import EmbeddingResult, RerankDocumentResult, RerankResult
from .base import LLMProvider

if TYPE_CHECKING:
    from ..core.config import LiteLLMConfig


class LiteLLMProvider(LLMProvider):
    """Universal LLM provider using LiteLLM.

    LiteLLM abstracts away provider-specific API differences, allowing
    seamless switching between OpenAI, Anthropic, Google, Azure, and 100+ providers.

    Example:
        config = LiteLLMConfig(model="gpt-4o-mini", embedding_model="text-embedding-3-small")
        provider = LiteLLMProvider(config)
        result = await provider.generate("Hello, world!")
    """

    def __init__(self, config: "LiteLLMConfig"):
        """Initialize LiteLLM provider.

        Args:
            config: LiteLLMConfig with model settings.
        """
        self._config = config
        self._model = config.model
        self._embedding_model = config.embedding_model
        self._reranker_model = config.reranker_model or config.model
        self._timeout = config.timeout
        self._api_base = config.api_base or None

        # Configure LiteLLM
        litellm.request_timeout = config.timeout

        logger.debug(
            f"LiteLLM provider initialized: model={self._model}, "
            f"embedding_model={self._embedding_model}"
        )

    async def embed(
        self,
        text: str,
        model: str | None = None,
        is_query: bool = False,
    ) -> EmbeddingResult | None:
        """Generate embeddings using LiteLLM.

        Args:
            text: Text to embed.
            model: Optional model override.
            is_query: If True, embed as query. If False, embed as document.

        Returns:
            EmbeddingResult or None on failure.
        """
        model = model or self._embedding_model
        text_preview = text[:50] + "..." if len(text) > 50 else text
        logger.debug(f"Embedding text ({len(text)} chars, is_query={is_query}): {text_preview!r}")
        start_time = time.perf_counter()

        try:
            response = await litellm.aembedding(
                model=model,
                input=[text],
                api_base=self._api_base,
                timeout=self._timeout,
            )

            elapsed = (time.perf_counter() - start_time) * 1000
            embedding = response.data[0]["embedding"]
            logger.debug(f"Embedding generated: dim={len(embedding)}, {elapsed:.1f}ms")
            return EmbeddingResult(embedding=embedding, model=model)

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
        """Generate text using LiteLLM.

        Args:
            prompt: Prompt text.
            model: Optional model override.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text or None on failure.
        """
        model = model or self._model
        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        logger.debug(f"Generating text (max_tokens={max_tokens}, temp={temperature}): {prompt_preview!r}")
        start_time = time.perf_counter()

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                api_base=self._api_base,
                timeout=self._timeout,
            )

            elapsed = (time.perf_counter() - start_time) * 1000
            result = response.choices[0].message.content
            resp_preview = result[:50] + "..." if result and len(result) > 50 else result
            logger.debug(f"Generated in {elapsed:.1f}ms: {resp_preview!r}")
            return result

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
        """Rerank documents using LiteLLM.

        Uses the LLM as a relevance judge to rerank documents.

        Args:
            query: Search query.
            documents: List of dicts with 'file' and 'body' keys.
            model: Optional model override.

        Returns:
            RerankResult with relevance scores.
        """
        model = model or self._reranker_model
        logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]!r}...")
        start_time = time.perf_counter()
        results = []

        for i, doc in enumerate(documents):
            system_prompt = (
                "You are a relevance judge. Given a query and a document, "
                "respond with ONLY 'Yes' if the document is relevant to the query, "
                "or 'No' if it is not relevant."
            )

            user_prompt = f"Query: {query}\n\nDocument: {doc.get('body', '')[:1000]}"

            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=1,
                    temperature=0.0,
                    api_base=self._api_base,
                    timeout=self._timeout,
                )

                answer = response.choices[0].message.content.strip().lower()
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
        """Check if model is available via LiteLLM.

        LiteLLM doesn't have a direct model listing API, so we try a minimal
        completion request to verify the model works.

        Args:
            model: Model name to check.

        Returns:
            True if model is available.
        """
        try:
            await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                api_base=self._api_base,
                timeout=10.0,
            )
            return True
        except Exception:
            return False

    async def is_available(self) -> bool:
        """Check if LiteLLM can reach the configured provider.

        Returns:
            True if service is reachable.
        """
        logger.debug(f"Checking LiteLLM availability for model {self._model}")
        try:
            # Try a minimal completion to verify connectivity
            await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                api_base=self._api_base,
                timeout=10.0,
            )
            logger.debug("LiteLLM available: True")
            return True
        except Exception as e:
            logger.debug(f"LiteLLM not available: {e}")
            return False

    def get_default_embedding_model(self) -> str:
        """Get default embedding model."""
        return self._embedding_model

    def get_default_expansion_model(self) -> str:
        """Get default expansion model."""
        return self._model

    def get_default_reranker_model(self) -> str:
        """Get default reranker model."""
        return self._reranker_model

    async def close(self) -> None:
        """Close the provider.

        LiteLLM manages its own connection pool, so nothing to close.
        """
        pass
