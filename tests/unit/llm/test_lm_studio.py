"""Tests for LM Studio LLM provider."""

import pytest
import respx
from httpx import Response

from pmd.llm.lm_studio import LMStudioProvider
from pmd.llm.base import LLMProvider
from pmd.core.config import LMStudioConfig
from pmd.core.types import EmbeddingResult, RerankResult

from .conftest import make_embedding_response, make_chat_response, make_models_response


class TestLMStudioProviderInit:
    """Tests for LMStudioProvider initialization."""

    def test_implements_llm_provider(self, lm_studio_config):
        """LMStudioProvider should implement LLMProvider."""
        provider = LMStudioProvider(lm_studio_config)
        assert isinstance(provider, LLMProvider)

    def test_stores_config(self, lm_studio_config):
        """Should store configuration."""
        provider = LMStudioProvider(lm_studio_config)
        assert provider.config == lm_studio_config

    def test_creates_http_client(self, lm_studio_config):
        """Should create HTTP client with timeout."""
        provider = LMStudioProvider(lm_studio_config)
        assert provider._client is not None
        # Client should have the configured timeout
        assert provider._client.timeout.connect == lm_studio_config.timeout


class TestLMStudioProviderDefaultModels:
    """Tests for default model getters."""

    def test_get_default_embedding_model(self, lm_studio_config):
        """Should return configured embedding model."""
        provider = LMStudioProvider(lm_studio_config)
        assert provider.get_default_embedding_model() == "nomic-embed-text"

    def test_get_default_expansion_model(self, lm_studio_config):
        """Should return configured expansion model."""
        provider = LMStudioProvider(lm_studio_config)
        assert provider.get_default_expansion_model() == "qwen2:0.5b"

    def test_get_default_reranker_model(self, lm_studio_config):
        """Should return configured reranker model."""
        provider = LMStudioProvider(lm_studio_config)
        assert provider.get_default_reranker_model() == "qwen2:0.5b"


class TestLMStudioEmbed:
    """Tests for LMStudioProvider.embed method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_success(self, lm_studio_config):
        """embed should return EmbeddingResult on success."""
        embedding = [0.1, 0.2, 0.3] * 256  # 768 dimensions
        respx.post("http://localhost:1234/v1/embeddings").mock(
            return_value=Response(200, json=make_embedding_response(embedding))
        )

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.embed("test text")

        assert result is not None
        assert isinstance(result, EmbeddingResult)
        assert result.embedding == embedding

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_uses_default_model(self, lm_studio_config):
        """embed should use default model when not specified."""
        route = respx.post("http://localhost:1234/v1/embeddings").mock(
            return_value=Response(200, json=make_embedding_response())
        )

        provider = LMStudioProvider(lm_studio_config)
        await provider.embed("test")

        request = route.calls[0].request
        import json
        body = json.loads(request.content)
        assert body["model"] == "nomic-embed-text"

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_uses_custom_model(self, lm_studio_config):
        """embed should use custom model when specified."""
        route = respx.post("http://localhost:1234/v1/embeddings").mock(
            return_value=Response(200, json=make_embedding_response())
        )

        provider = LMStudioProvider(lm_studio_config)
        await provider.embed("test", model="custom-model")

        request = route.calls[0].request
        import json
        body = json.loads(request.content)
        assert body["model"] == "custom-model"

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_sends_text(self, lm_studio_config):
        """embed should send text in request body."""
        route = respx.post("http://localhost:1234/v1/embeddings").mock(
            return_value=Response(200, json=make_embedding_response())
        )

        provider = LMStudioProvider(lm_studio_config)
        await provider.embed("Hello, world!")

        request = route.calls[0].request
        import json
        body = json.loads(request.content)
        assert body["input"] == "Hello, world!"

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_returns_none_on_error(self, lm_studio_config):
        """embed should return None on HTTP error."""
        respx.post("http://localhost:1234/v1/embeddings").mock(
            return_value=Response(500, json={"error": "Internal error"})
        )

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.embed("test")

        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_returns_none_on_network_error(self, lm_studio_config):
        """embed should return None on network error."""
        respx.post("http://localhost:1234/v1/embeddings").mock(side_effect=Exception("Network error"))

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.embed("test")

        assert result is None


class TestLMStudioGenerate:
    """Tests for LMStudioProvider.generate method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self, lm_studio_config):
        """generate should return text on success."""
        respx.post("http://localhost:1234/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("Generated text"))
        )

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.generate("prompt")

        assert result == "Generated text"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_uses_default_model(self, lm_studio_config):
        """generate should use expansion model by default."""
        route = respx.post("http://localhost:1234/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("response"))
        )

        provider = LMStudioProvider(lm_studio_config)
        await provider.generate("prompt")

        request = route.calls[0].request
        import json
        body = json.loads(request.content)
        assert body["model"] == "qwen2:0.5b"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_uses_custom_parameters(self, lm_studio_config):
        """generate should use custom max_tokens and temperature."""
        route = respx.post("http://localhost:1234/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("response"))
        )

        provider = LMStudioProvider(lm_studio_config)
        await provider.generate("prompt", max_tokens=100, temperature=0.5)

        request = route.calls[0].request
        import json
        body = json.loads(request.content)
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.5

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_returns_none_on_error(self, lm_studio_config):
        """generate should return None on HTTP error."""
        respx.post("http://localhost:1234/v1/chat/completions").mock(
            return_value=Response(500, json={"error": "Error"})
        )

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.generate("prompt")

        assert result is None


class TestLMStudioRerank:
    """Tests for LMStudioProvider.rerank method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_rerank_returns_result(self, lm_studio_config):
        """rerank should return RerankResult."""
        respx.post("http://localhost:1234/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("Yes"))
        )

        provider = LMStudioProvider(lm_studio_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert isinstance(result, RerankResult)
        assert len(result.results) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_rerank_yes_is_relevant(self, lm_studio_config):
        """rerank should mark 'Yes' response as relevant."""
        respx.post("http://localhost:1234/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("Yes"))
        )

        provider = LMStudioProvider(lm_studio_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert result.results[0].relevant is True
        assert result.results[0].score > 0.5

    @pytest.mark.asyncio
    @respx.mock
    async def test_rerank_no_is_not_relevant(self, lm_studio_config):
        """rerank should mark 'No' response as not relevant."""
        respx.post("http://localhost:1234/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_response("No"))
        )

        provider = LMStudioProvider(lm_studio_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert result.results[0].relevant is False
        assert result.results[0].score < 0.5

    @pytest.mark.asyncio
    @respx.mock
    async def test_rerank_multiple_documents(self, lm_studio_config):
        """rerank should handle multiple documents."""
        # Alternate Yes/No responses
        respx.post("http://localhost:1234/v1/chat/completions").mock(
            side_effect=[
                Response(200, json=make_chat_response("Yes")),
                Response(200, json=make_chat_response("No")),
                Response(200, json=make_chat_response("Yes")),
            ]
        )

        provider = LMStudioProvider(lm_studio_config)
        docs = [
            {"file": "doc1.md", "body": "content1"},
            {"file": "doc2.md", "body": "content2"},
            {"file": "doc3.md", "body": "content3"},
        ]
        result = await provider.rerank("query", docs)

        assert len(result.results) == 3
        assert result.results[0].relevant is True
        assert result.results[1].relevant is False
        assert result.results[2].relevant is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_rerank_handles_error_gracefully(self, lm_studio_config):
        """rerank should return neutral score on error."""
        respx.post("http://localhost:1234/v1/chat/completions").mock(
            side_effect=Exception("Network error")
        )

        provider = LMStudioProvider(lm_studio_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert len(result.results) == 1
        assert result.results[0].score == 0.5
        assert result.results[0].raw_token == "error"


class TestLMStudioModelExists:
    """Tests for LMStudioProvider.model_exists method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_exists_true(self, lm_studio_config):
        """model_exists should return True if model in list."""
        respx.get("http://localhost:1234/v1/models").mock(
            return_value=Response(200, json=make_models_response(["model-a", "model-b"]))
        )

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.model_exists("model-a")

        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_exists_false(self, lm_studio_config):
        """model_exists should return False if model not in list."""
        respx.get("http://localhost:1234/v1/models").mock(
            return_value=Response(200, json=make_models_response(["model-a", "model-b"]))
        )

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.model_exists("model-c")

        assert result is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_exists_error(self, lm_studio_config):
        """model_exists should return False on error."""
        respx.get("http://localhost:1234/v1/models").mock(
            return_value=Response(500, json={"error": "Error"})
        )

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.model_exists("any-model")

        assert result is False


class TestLMStudioIsAvailable:
    """Tests for LMStudioProvider.is_available method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_true(self, lm_studio_config):
        """is_available should return True when service responds."""
        respx.get("http://localhost:1234/v1/models").mock(
            return_value=Response(200, json=make_models_response([]))
        )

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.is_available()

        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_error(self, lm_studio_config):
        """is_available should return False on HTTP error."""
        respx.get("http://localhost:1234/v1/models").mock(
            return_value=Response(500, json={"error": "Error"})
        )

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.is_available()

        assert result is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_network_error(self, lm_studio_config):
        """is_available should return False on network error."""
        respx.get("http://localhost:1234/v1/models").mock(
            side_effect=Exception("Connection refused")
        )

        provider = LMStudioProvider(lm_studio_config)
        result = await provider.is_available()

        assert result is False


class TestLMStudioClose:
    """Tests for LMStudioProvider.close method."""

    @pytest.mark.asyncio
    async def test_close_closes_client(self, lm_studio_config):
        """close should close the HTTP client."""
        provider = LMStudioProvider(lm_studio_config)

        await provider.close()

        assert provider._client.is_closed
