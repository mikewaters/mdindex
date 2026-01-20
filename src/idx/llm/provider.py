"""idx.llm.provider - MLX LLM provider for local inference.

Provides a simplified MLX provider that wraps mlx-lm for text generation.
Used for LLM-as-judge reranking on Apple Silicon.

Example:
    from idx.llm.provider import MLXProvider

    provider = MLXProvider()
    response = await provider.generate("Is this relevant? Yes or No:")
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from idx.core.logging import get_logger
from idx.core.settings import get_settings

if TYPE_CHECKING:
    pass

__all__ = [
    "MLXProvider",
    "LLMProviderError",
]

logger = get_logger(__name__)


class LLMProviderError(Exception):
    """Error from LLM provider operations."""

    pass


class MLXProvider:
    """MLX local LLM provider for Apple Silicon.

    Uses mlx-lm for text generation (reranking, query expansion).
    Models are lazy-loaded on first use.

    Attributes:
        model_name: HuggingFace model name/path.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize MLX provider.

        Args:
            model_name: HuggingFace model name. If not provided,
                uses IDX_TRANSFORMERS_MODEL from settings.

        Raises:
            LLMProviderError: If not running on macOS/Apple Silicon.
        """
        if sys.platform != "darwin":
            raise LLMProviderError("MLX provider requires macOS with Apple Silicon")

        settings = get_settings()
        self.model_name = model_name or settings.transformers_model

        # Lazy-loaded model and tokenizer
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._loaded:
            return

        import time

        logger.info(f"Loading MLX model: {self.model_name}")
        start = time.perf_counter()

        try:
            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_name)
            self._loaded = True

            elapsed = time.perf_counter() - start
            logger.info(f"Model loaded in {elapsed:.2f}s")

        except ImportError as e:
            raise LLMProviderError(
                "mlx-lm not installed. Install with: pip install mlx-lm"
            ) from e
        except Exception as e:
            raise LLMProviderError(f"Failed to load model: {e}") from e

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 10,
        temperature: float = 0.0,
    ) -> str:
        """Generate text using MLX.

        Args:
            prompt: User prompt text.
            system: Optional system prompt (for chat models).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 for deterministic).

        Returns:
            Generated text.

        Raises:
            LLMProviderError: If generation fails.
        """
        import time

        self._ensure_loaded()

        start = time.perf_counter()
        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        logger.debug(f"Generating (max_tokens={max_tokens}): {prompt_preview!r}")

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            # Format as chat message for instruction-tuned models
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Apply chat template if available
            if hasattr(self._tokenizer, "apply_chat_template"):
                formatted_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback for models without chat template
                formatted_prompt = prompt
                if system:
                    formatted_prompt = f"{system}\n\n{prompt}"

            # Create sampler
            sampler = make_sampler(temp=temperature)

            response = generate(
                self._model,
                self._tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )

            result = response.strip() if response else ""
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"Generated in {elapsed:.1f}ms: {result!r}")

            return result

        except Exception as e:
            raise LLMProviderError(f"Generation failed: {e}") from e

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.debug("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._loaded

    @staticmethod
    def is_available() -> bool:
        """Check if MLX is available on this system.

        Returns:
            True if on macOS and mlx-lm is importable.
        """
        if sys.platform != "darwin":
            return False

        try:
            import mlx_lm  # noqa: F401

            return True
        except ImportError:
            return False
