"""MLX LLM provider for local inference on Apple Silicon."""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

from loguru import logger

from ..core.config import MLXConfig
from ..core.types import EmbeddingResult, RerankDocumentResult, RerankResult
from .base import LLMProvider

if TYPE_CHECKING:
    from mlx_lm import generate, load


class MLXProvider(LLMProvider):
    """MLX local LLM provider for Apple Silicon Macs.

    Uses mlx-lm for text generation (query expansion, reranking)
    and mlx-embeddings for embedding generation.

    Models are downloaded from HuggingFace. Authentication can be provided via:
    - HF_TOKEN environment variable
    - huggingface-cli login (or hf auth login)
    """

    def __init__(self, config: MLXConfig):
        """Initialize MLX provider.

        Args:
            config: MLXConfig with model settings.

        Raises:
            RuntimeError: If not running on macOS/Apple Silicon.
        """
        if sys.platform != "darwin":
            raise RuntimeError("MLX provider requires macOS with Apple Silicon")

        self.config = config

        # Text generation model (mlx-lm)
        self._model = None
        self._tokenizer = None
        self._model_loaded = False

        # Embedding model (mlx-embeddings)
        self._embedding_model = None
        self._embedding_tokenizer = None
        self._embedding_model_loaded = False

        # Load immediately if not lazy loading
        if not config.lazy_load:
            self._ensure_model_loaded()
            self._ensure_embedding_model_loaded()

    def _get_hf_token(self) -> str | None:
        """Get HuggingFace token from environment or cached login.

        Checks in order:
        1. HF_TOKEN environment variable
        2. Cached token from huggingface-cli login

        Returns:
            Token string or None if not available.
        """
        # Check environment variable first
        if token := os.environ.get("HF_TOKEN"):
            return token

        # Try to get token from huggingface_hub cached login
        try:
            from huggingface_hub import HfFolder
            return HfFolder.get_token()
        except Exception:
            return None

    def _ensure_model_loaded(self) -> None:
        """Load the text generation model if not already loaded."""
        if self._model_loaded:
            return

        logger.info(f"Loading text generation model: {self.config.model}")
        start_time = time.perf_counter()

        from mlx_lm import load

        # Get HF token for model download
        token = self._get_hf_token()

        # mlx_lm.load accepts tokenizer_config dict for token
        if token:
            logger.debug("Using HuggingFace token for model download")
            self._model, self._tokenizer = load(
                self.config.model,
                tokenizer_config={"token": token},
            )
        else:
            self._model, self._tokenizer = load(self.config.model)

        self._model_loaded = True
        elapsed = time.perf_counter() - start_time
        logger.info(f"Model loaded in {elapsed:.2f}s: {self.config.model}")

    def _ensure_embedding_model_loaded(self) -> None:
        """Load the embedding model if not already loaded."""
        if self._embedding_model_loaded:
            return

        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        start_time = time.perf_counter()

        from mlx_embeddings import load as load_embeddings

        # Get HF token for model download
        token = self._get_hf_token()

        # mlx_embeddings.load returns (model, tokenizer)
        # Token is passed via tokenizer_config for HuggingFace auth
        tokenizer_config = {"token": token} if token else {}
        if token:
            logger.debug("Using HuggingFace token for embedding model download")

        self._embedding_model, self._embedding_tokenizer = load_embeddings(
            self.config.embedding_model,
            tokenizer_config=tokenizer_config,
        )

        self._embedding_model_loaded = True
        elapsed = time.perf_counter() - start_time
        logger.info(f"Embedding model loaded in {elapsed:.2f}s: {self.config.embedding_model}")

    async def embed(
        self,
        text: str,
        model: str | None = None,
        is_query: bool = False,
    ) -> EmbeddingResult | None:
        """Generate embeddings using mlx-embeddings.

        Args:
            text: Text to embed.
            model: Ignored (uses configured embedding model).
            is_query: If True, formats text as query for asymmetric models.

        Returns:
            EmbeddingResult with embedding vector, or None on failure.
        """
        text_preview = text[:50] + "..." if len(text) > 50 else text
        logger.debug(f"Embedding text ({len(text)} chars, is_query={is_query}): {text_preview!r}")
        start_time = time.perf_counter()

        try:
            self._ensure_embedding_model_loaded()

            from mlx_embeddings import generate as generate_embeddings

            # Some embedding models (like E5) expect query/passage prefixes
            if is_query:
                formatted_text = f"query: {text}"
            else:
                formatted_text = f"passage: {text}"

            # Generate embedding using mlx_embeddings.generate
            # Returns a BaseModelOutput with pooler_output for sentence embedding
            result = generate_embeddings(
                self._embedding_model,
                self._embedding_tokenizer,
                formatted_text,
            )

            # Extract sentence embedding from pooler_output
            # Shape is (batch_size, embedding_dim), we take first element
            if hasattr(result, "pooler_output"):
                embedding = result.pooler_output.tolist()[0]
            elif hasattr(result, "text_embeds"):
                embedding = result.text_embeds.tolist()[0]
            else:
                # Fallback: try last_hidden_state mean pooling
                embedding = result.last_hidden_state.mean(axis=1).tolist()[0]

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Embedding generated: dim={len(embedding)}, {elapsed:.1f}ms")

            return EmbeddingResult(
                embedding=embedding,
                model=self.config.embedding_model,
            )

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
        """Generate text using MLX.

        Args:
            prompt: Prompt text.
            model: Ignored (uses configured model).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text or None on failure.
        """
        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        logger.debug(f"Generating text (max_tokens={max_tokens}, temp={temperature}): {prompt_preview!r}")
        start_time = time.perf_counter()

        try:
            self._ensure_model_loaded()

            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            # Format as chat message for instruction-tuned models
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template if available
            if hasattr(self._tokenizer, "apply_chat_template"):
                formatted_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted_prompt = prompt

            # Create sampler with temperature settings
            sampler = make_sampler(temp=temperature)

            response = generate(
                self._model,
                self._tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )

            result = response.strip() if response else None
            elapsed = (time.perf_counter() - start_time) * 1000
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
        """Rerank documents using MLX for relevance judgment.

        Args:
            query: Search query.
            documents: List of dicts with 'file' and 'body' keys.
            model: Ignored (uses configured model).

        Returns:
            RerankResult with relevance scores.
        """
        logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]!r}...")
        start_time = time.perf_counter()
        results = []

        for i, doc in enumerate(documents):
            prompt = (
                "You are a relevance judge. Given a query and a document, "
                "respond with ONLY 'Yes' if the document is relevant to the query, "
                "or 'No' if it is not relevant.\n\n"
                f"Query: {query}\n\n"
                f"Document: {doc.get('body', '')[:1000]}\n\n"
                "Is this document relevant? Answer with Yes or No only:"
            )

            try:
                response = await self.generate(
                    prompt,
                    max_tokens=5,
                    temperature=0.0,
                )

                if response:
                    answer = response.strip().lower()
                    relevant = answer.startswith("yes")
                    confidence = 0.9 if relevant else 0.1
                    score = 0.5 + 0.5 * confidence if relevant else 0.5 * (1 - confidence)

                    results.append(
                        RerankDocumentResult(
                            file=doc.get("file", ""),
                            relevant=relevant,
                            confidence=confidence,
                            score=score,
                            raw_token=answer[:10],
                            logprob=None,
                        )
                    )
                    logger.debug(f"  [{i+1}/{len(documents)}] {doc.get('file', '')}: {answer} (score={score:.2f})")
                else:
                    # Default neutral on failure
                    results.append(self._neutral_result(doc))
                    logger.debug(f"  [{i+1}/{len(documents)}] {doc.get('file', '')}: no response, neutral")

            except Exception as e:
                results.append(self._neutral_result(doc))
                logger.warning(f"  [{i+1}/{len(documents)}] {doc.get('file', '')}: error: {e}")

        elapsed = (time.perf_counter() - start_time) * 1000
        relevant_count = sum(1 for r in results if r.relevant)
        logger.debug(f"Reranking complete: {relevant_count}/{len(results)} relevant, {elapsed:.1f}ms")

        return RerankResult(results=results, model=self.config.model)

    def _neutral_result(self, doc: dict) -> RerankDocumentResult:
        """Create a neutral rerank result for error cases."""
        return RerankDocumentResult(
            file=doc.get("file", ""),
            relevant=False,
            confidence=0.5,
            score=0.5,
            raw_token="error",
            logprob=None,
        )

    async def model_exists(self, model: str) -> bool:
        """Check if model is available.

        For MLX, we just check if the configured model can be loaded.

        Args:
            model: Model name to check.

        Returns:
            True if model appears to be valid HuggingFace model ID.
        """
        # Basic validation - check if it looks like a valid model ID
        return "/" in model or model == self.config.model

    async def is_available(self) -> bool:
        """Check if MLX is available.

        Returns:
            True if running on macOS and mlx-lm is importable.
        """
        if sys.platform != "darwin":
            return False

        try:
            import mlx_lm
            return True
        except ImportError:
            return False

    def get_default_embedding_model(self) -> str:
        """Get default embedding model name.

        Returns:
            Embedding model name (from mlx-embeddings).
        """
        return self.config.embedding_model

    def get_default_expansion_model(self) -> str:
        """Get default query expansion model name.

        Returns:
            Model name.
        """
        return self.config.model

    def get_default_reranker_model(self) -> str:
        """Get default reranker model name.

        Returns:
            Model name.
        """
        return self.config.model

    def unload_model(self) -> None:
        """Unload the text generation model to free memory."""
        self._model = None
        self._tokenizer = None
        self._model_loaded = False

    def unload_embedding_model(self) -> None:
        """Unload the embedding model to free memory."""
        self._embedding_model = None
        self._embedding_tokenizer = None
        self._embedding_model_loaded = False

    def unload_all(self) -> None:
        """Unload all models to free memory."""
        self.unload_model()
        self.unload_embedding_model()

    async def close(self) -> None:
        """Close the provider and release resources.

        For MLX, this unloads all models to free memory.
        """
        self.unload_all()
