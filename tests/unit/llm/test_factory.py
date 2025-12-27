"""Tests for LLM provider factory."""

import pytest

from pmd.llm.factory import create_llm_provider, get_provider_name
from pmd.llm.lm_studio import LMStudioProvider
from pmd.llm.openrouter import OpenRouterProvider
from pmd.core.config import Config, LMStudioConfig, OpenRouterConfig


class TestCreateLLMProvider:
    """Tests for create_llm_provider function."""

    def test_creates_lm_studio_provider(self):
        """Should create LMStudioProvider for lm-studio."""
        config = Config(llm_provider="lm-studio")

        provider = create_llm_provider(config)

        assert isinstance(provider, LMStudioProvider)

    def test_creates_lm_studio_case_insensitive(self):
        """Should handle case-insensitive provider name."""
        config = Config(llm_provider="LM-STUDIO")

        provider = create_llm_provider(config)

        assert isinstance(provider, LMStudioProvider)

    def test_creates_lm_studio_with_whitespace(self):
        """Should handle whitespace in provider name."""
        config = Config(llm_provider="  lm-studio  ")

        provider = create_llm_provider(config)

        assert isinstance(provider, LMStudioProvider)

    def test_creates_openrouter_provider(self):
        """Should create OpenRouterProvider for openrouter."""
        config = Config(llm_provider="openrouter")
        config.openrouter.api_key = "test-key"

        provider = create_llm_provider(config)

        assert isinstance(provider, OpenRouterProvider)

    def test_openrouter_requires_api_key(self):
        """Should raise if OpenRouter API key is missing."""
        config = Config(llm_provider="openrouter")
        config.openrouter.api_key = ""

        with pytest.raises(ValueError, match="API_KEY"):
            create_llm_provider(config)

    def test_ollama_not_implemented(self):
        """Should raise NotImplementedError for ollama."""
        config = Config(llm_provider="ollama")

        with pytest.raises(NotImplementedError, match="Ollama"):
            create_llm_provider(config)

    def test_unknown_provider_raises(self):
        """Should raise ValueError for unknown provider."""
        config = Config(llm_provider="unknown-provider")

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider(config)

    def test_error_message_lists_providers(self):
        """Error message should list supported providers."""
        config = Config(llm_provider="invalid")

        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(config)

        error_msg = str(exc_info.value)
        assert "lm-studio" in error_msg
        assert "openrouter" in error_msg


class TestCreateLLMProviderConfigs:
    """Tests for provider configuration passing."""

    def test_passes_lm_studio_config(self):
        """Should pass LMStudioConfig to provider."""
        config = Config(llm_provider="lm-studio")
        config.lm_studio.base_url = "http://custom:1234"
        config.lm_studio.embedding_model = "custom-embed"

        provider = create_llm_provider(config)

        assert provider.config.base_url == "http://custom:1234"
        assert provider.config.embedding_model == "custom-embed"

    def test_passes_openrouter_config(self):
        """Should pass OpenRouterConfig to provider."""
        config = Config(llm_provider="openrouter")
        config.openrouter.api_key = "secret-key"
        config.openrouter.embedding_model = "custom/embed"

        provider = create_llm_provider(config)

        assert provider.config.api_key == "secret-key"
        assert provider.config.embedding_model == "custom/embed"


class TestGetProviderName:
    """Tests for get_provider_name function."""

    def test_lm_studio_name(self):
        """Should return 'LM Studio' for lm-studio."""
        config = Config(llm_provider="lm-studio")

        name = get_provider_name(config)

        assert name == "LM Studio"

    def test_openrouter_name(self):
        """Should return 'OpenRouter' for openrouter."""
        config = Config(llm_provider="openrouter")

        name = get_provider_name(config)

        assert name == "OpenRouter"

    def test_ollama_name(self):
        """Should return 'Ollama' for ollama."""
        config = Config(llm_provider="ollama")

        name = get_provider_name(config)

        assert name == "Ollama"

    def test_unknown_returns_raw(self):
        """Should return raw provider name for unknown."""
        config = Config(llm_provider="custom-provider")

        name = get_provider_name(config)

        assert name == "custom-provider"

    def test_handles_case_and_whitespace(self):
        """Should handle case and whitespace."""
        config = Config(llm_provider="  LM-STUDIO  ")

        name = get_provider_name(config)

        assert name == "LM Studio"


class TestCreateLLMProviderDefaults:
    """Tests for default provider configuration."""

    def test_default_lm_studio_config(self):
        """Default LM Studio config should have sensible defaults."""
        config = Config(llm_provider="lm-studio")

        provider = create_llm_provider(config)

        assert provider.config.base_url == "http://localhost:1234"
        assert provider.config.timeout > 0

    def test_default_models_accessible(self):
        """Provider should expose default models."""
        config = Config(llm_provider="lm-studio")

        provider = create_llm_provider(config)

        assert provider.get_default_embedding_model() is not None
        assert provider.get_default_expansion_model() is not None
        assert provider.get_default_reranker_model() is not None
