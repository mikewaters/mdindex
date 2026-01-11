"""DSPy client wrapper for PMD LLM providers.

This module provides a DSPy-compatible language model that wraps PMD's
LLMProvider implementations, allowing DSPy modules to use our MLX,
LiteLLM, or other providers seamlessly.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import dspy

if TYPE_CHECKING:
    from .base import LLMProvider


class PMDClient(dspy.BaseLM):
    """DSPy LM wrapper for PMD LLMProvider.

    This class bridges PMD's LLMProvider interface with DSPy's LM interface,
    allowing DSPy modules (signatures, chain-of-thought, etc.) to use our
    local MLX models or remote API providers.

    Example:
        from pmd.llm import create_llm_provider, PMDClient
        from pmd.core import Config

        provider = create_llm_provider(Config())
        lm = PMDClient(provider)
        dspy.configure(lm=lm)

        # Now DSPy modules will use our provider
        response = lm("What is 2+2?")
    """

    def __init__(
        self,
        provider: "LLMProvider",
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        """Initialize PMDClient.

        Args:
            provider: PMD LLMProvider instance (MLXProvider, LiteLLMProvider, etc.)
            temperature: Default sampling temperature.
            max_tokens: Default maximum tokens to generate.
        """
        self._provider = provider
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Call parent init with model name
        model_name = f"pmd/{provider.get_default_expansion_model()}"
        super().__init__(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @property
    def provider(self) -> "LLMProvider":
        """Get the underlying LLM provider."""
        return self._provider

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Synchronous forward pass (runs async code in event loop).

        Args:
            prompt: Simple string prompt.
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional generation parameters.

        Returns:
            List of response dicts in OpenAI format.
        """
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, run in thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.aforward(prompt=prompt, messages=messages, **kwargs),
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(
                self.aforward(prompt=prompt, messages=messages, **kwargs)
            )

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Async forward pass using the PMD provider.

        Args:
            prompt: Simple string prompt.
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional generation parameters.

        Returns:
            List of response dicts in OpenAI format.
        """
        # Extract parameters with defaults
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        # Build prompt from messages if not provided directly
        if prompt is None and messages:
            prompt = self._messages_to_prompt(messages)
        elif prompt is None:
            raise ValueError("Either prompt or messages must be provided")

        # Generate using our provider
        result = await self._provider.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Return in DSPy-expected format (list of OpenAI-style responses)
        if result is None:
            result = ""

        return [self._make_response(result)]

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Formatted prompt string.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        return "\n\n".join(parts)

    def _make_response(self, text: str) -> dict[str, Any]:
        """Create an OpenAI-style response dict.

        Args:
            text: Generated text.

        Returns:
            Response dict compatible with DSPy expectations.
        """
        return {
            "choices": [
                {
                    "message": {
                        "content": text,
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ],
            "model": self.model,
            "usage": {},
        }

    def __deepcopy__(self, memo: dict) -> "PMDClient":
        """Handle deep copy for DSPy serialization.

        DSPy may deep-copy programs; returning self prevents issues
        with non-copyable provider resources.
        """
        return self

    def dump_state(self) -> dict[str, Any]:
        """Serialize state for DSPy program saving."""
        return {
            "model": self.model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from serialized form.

        Note: This only restores parameters, not the provider instance.
        The provider must be re-created separately.
        """
        self._temperature = state.get("temperature", self._temperature)
        self._max_tokens = state.get("max_tokens", self._max_tokens)


def create_dspy_client(provider: "LLMProvider", **kwargs: Any) -> PMDClient:
    """Create a DSPy client from a PMD LLM provider.

    Convenience function for creating and configuring a PMDClient.

    Args:
        provider: PMD LLMProvider instance.
        **kwargs: Additional configuration (temperature, max_tokens).

    Returns:
        Configured PMDClient ready for use with DSPy.

    Example:
        provider = create_llm_provider(config)
        lm = create_dspy_client(provider, temperature=0.5)
        dspy.configure(lm=lm)
    """
    return PMDClient(provider, **kwargs)
