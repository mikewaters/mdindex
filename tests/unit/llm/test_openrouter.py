"""Tests for OpenRouter LLM provider."""

import pytest
import respx
from httpx import Response

from pmd.llm.openrouter import OpenRouterProvider
from pmd.llm.base import LLMProvider
from pmd.core.config import OpenRouterConfig
from pmd.core.types import EmbeddingResult, RerankResult

from .conftest import make_embedding_response, make_chat_response, make_models_response


class TestOpenRouterProviderInit:
    """Tests for OpenRouterProvider initialization."""

    def test_implements_llm_provider(self, openrouter_config):
        """OpenRouterProvider should implement LLMProvider."""
        provider = OpenRouterProvider(openrouter_config)
        assert isinstance(provider, LLMProvider)

    def test_stores_config(self, openrouter_config):
        """Should store configuration."""
        provider = OpenRouterProvider(openrouter_config)
        assert provider.config == openrouter_config

    def test_raises_on_missing_api_key(self):
        """Should raise ValueError if API key is missing."""
        config = OpenRouterConfig(api_key="")

        with pytest.raises(ValueError, match="API_KEY"):
            OpenRouterProvider(config)

    def test_creates_http_client_with_auth(self, openrouter_config):
        """Should create HTTP client with authorization header."""
        provider = OpenRouterProvider(openrouter_config)

        auth_header = provider._client.headers.get("authorization")
        assert auth_header == "Bearer test-api-key-12345"

    def test_creates_http_client_with_referer(self, openrouter_config):
        """Should set HTTP-Referer header."""
        provider = OpenRouterProvider(openrouter_config)

        referer = provider._client.headers.get("http-referer")
        assert "github.com" in referer

    def test_creates_http_client_with_title(self, openrouter_config):
        """Should set X-Title header."""
        provider = OpenRouterProvider(openrouter_config)

        title = provider._client.headers.get("x-title")
        assert "PMD" in title


class TestOpenRouterProviderDefaultModels:
    """Tests for default model getters."""

    def test_get_default_embedding_model(self, openrouter_config):
        """Should return configured embedding model."""
        provider = OpenRouterProvider(openrouter_config)
        assert provider.get_default_embedding_model() == "nomic-ai/nomic-embed-text"

    def test_get_default_expansion_model(self, openrouter_config):
        """Should return configured expansion model."""
        provider = OpenRouterProvider(openrouter_config)
        assert provider.get_default_expansion_model() == "qwen/qwen-1.5-0.5b"

    def test_get_default_reranker_model(self, openrouter_config):
        """Should return configured reranker model."""
        provider = OpenRouterProvider(openrouter_config)
        assert provider.get_default_reranker_model() == "qwen/qwen-1.5-0.5b"


class TestOpenRouterEmbed:
    """Tests for OpenRouterProvider.embed method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_success(self, openrouter_config):
        """embed should return EmbeddingResult on success."""
        embedding = [0.1, 0.2, 0.3] * 256
        respx.post("https://openrouter.io/api/v1/embeddings").mock(
            return_value=Response(200, json=make_embedding_response(embedding))
        )

        provider = OpenRouterProvider(openrouter_config)
        result = await provider.embed("test text")

        assert result is not None
        assert isinstance(result, EmbeddingResult)
        assert result.embedding == embedding

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_uses_correct_endpoint(self, openrouter_config):
        """embed should use OpenRouter embeddings endpoint."""
        route = respx.post("https://openrouter.io/api/v1/embeddings").mock(
            return_value=Response(200, json=make_embedding_response())
        )

        provider = OpenRouterProvider(openrouter_config)
        await provider.embed("test")

        assert route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_returns_none_on_error(self, openrouter_config):
        """embed should return None on HTTP error."""
        respx.post("https://openrouter.io/api/v1/embeddings").mock(
            return_value=Response(401, json={"error": "Unauthorized"})
        )

        provider = OpenRouterProvider(openrouter_config)
        result = await provider.embed("test")

        assert result is None


class TestOpenRouterGenerate:
    """Tests for OpenRouterProvider.generate method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self, openrouter_config):
        """generate should return text on success."""
        respx.post("https://openrouter.io/api/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("Generated text"))
        )

        provider = OpenRouterProvider(openrouter_config)
        result = await provider.generate("prompt")

        assert result == "Generated text"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_uses_correct_endpoint(self, openrouter_config):
        """generate should use OpenRouter chat completions endpoint."""
        route = respx.post("https://openrouter.io/api/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("response"))
        )

        provider = OpenRouterProvider(openrouter_config)
        await provider.generate("prompt")

        assert route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_returns_none_on_error(self, openrouter_config):
        """generate should return None on HTTP error."""
        respx.post("https://openrouter.io/api/v1/chat/completions").mock(
            return_value=Response(429, json={"error": "Rate limited"})
        )

        provider = OpenRouterProvider(openrouter_config)
        result = await provider.generate("prompt")

        assert result is None


class TestOpenRouterRerank:
    """Tests for OpenRouterProvider.rerank method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_rerank_returns_result(self, openrouter_config):
        """rerank should return RerankResult."""
        respx.post("https://openrouter.io/api/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("Yes"))
        )

        provider = OpenRouterProvider(openrouter_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert isinstance(result, RerankResult)
        assert len(result.results) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_rerank_yes_is_relevant(self, openrouter_config):
        """rerank should mark 'Yes' response as relevant."""
        respx.post("https://openrouter.io/api/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("Yes"))
        )

        provider = OpenRouterProvider(openrouter_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert result.results[0].relevant is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_rerank_no_is_not_relevant(self, openrouter_config):
        """rerank should mark 'No' response as not relevant."""
        respx.post("https://openrouter.io/api/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("No"))
        )

        provider = OpenRouterProvider(openrouter_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert result.results[0].relevant is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_rerank_handles_error_gracefully(self, openrouter_config):
        """rerank should return neutral score on error."""
        respx.post("https://openrouter.io/api/v1/chat/completions").mock(
            side_effect=Exception("Network error")
        )

        provider = OpenRouterProvider(openrouter_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert result.results[0].score == 0.5


class TestOpenRouterModelExists:
    """Tests for OpenRouterProvider.model_exists method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_exists_true(self, openrouter_config):
        """model_exists should return True if model in list."""
        respx.get("https://openrouter.io/api/v1/models").mock(
            return_value=Response(
                200, json=make_models_response(["anthropic/claude-3", "openai/gpt-4"])
            )
        )

        provider = OpenRouterProvider(openrouter_config)
        result = await provider.model_exists("anthropic/claude-3")

        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_exists_false(self, openrouter_config):
        """model_exists should return False if model not in list."""
        respx.get("https://openrouter.io/api/v1/models").mock(
            return_value=Response(200, json=make_models_response(["model-a"]))
        )

        provider = OpenRouterProvider(openrouter_config)
        result = await provider.model_exists("nonexistent-model")

        assert result is False


class TestOpenRouterIsAvailable:
    """Tests for OpenRouterProvider.is_available method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_true(self, openrouter_config):
        """is_available should return True when service responds."""
        respx.get("https://openrouter.io/api/v1/models").mock(
            return_value=Response(200, json=make_models_response([]))
        )

        provider = OpenRouterProvider(openrouter_config)
        result = await provider.is_available()

        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_error(self, openrouter_config):
        """is_available should return False on HTTP error."""
        respx.get("https://openrouter.io/api/v1/models").mock(
            return_value=Response(503, json={"error": "Service unavailable"})
        )

        provider = OpenRouterProvider(openrouter_config)
        result = await provider.is_available()

        assert result is False


class TestOpenRouterClose:
    """Tests for OpenRouterProvider.close method."""

    @pytest.mark.asyncio
    async def test_close_closes_client(self, openrouter_config):
        """close should close the HTTP client."""
        provider = OpenRouterProvider(openrouter_config)

        await provider.close()

        assert provider._client.is_closed
