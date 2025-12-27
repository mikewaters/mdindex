"""LLM provider factory for instantiating the right provider."""

from ..core.config import Config
from ..core.exceptions import OllamaConnectionError
from .base import LLMProvider
from .lm_studio import LMStudioProvider
from .openrouter import OpenRouterProvider


def create_llm_provider(config: Config) -> LLMProvider:
    """Create and return the configured LLM provider.

    Args:
        config: Application configuration.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If provider is not recognized or not properly configured.
    """
    provider_name = config.llm_provider.lower().strip()

    if provider_name == "lm-studio":
        return LMStudioProvider(config.lm_studio)
    elif provider_name == "openrouter":
        return OpenRouterProvider(config.openrouter)
    elif provider_name == "ollama":
        # Legacy Ollama support would go here
        raise NotImplementedError("Ollama support coming in future release")
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            "Supported providers: lm-studio, openrouter"
        )


def get_provider_name(config: Config) -> str:
    """Get human-readable provider name.

    Args:
        config: Application configuration.

    Returns:
        Provider display name.
    """
    provider = config.llm_provider.lower().strip()
    names = {
        "lm-studio": "LM Studio",
        "openrouter": "OpenRouter",
        "ollama": "Ollama",
    }
    return names.get(provider, provider)
