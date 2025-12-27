"""Tests for LLMProvider abstract base class."""

import pytest
from abc import ABC

from pmd.llm.base import LLMProvider
from pmd.core.types import EmbeddingResult, RerankResult


class TestLLMProviderInterface:
    """Tests for LLMProvider abstract interface."""

    def test_is_abstract_class(self):
        """LLMProvider should be an abstract base class."""
        assert issubclass(LLMProvider, ABC)

    def test_cannot_instantiate(self):
        """LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_has_embed_method(self):
        """Should have embed abstract method."""
        assert hasattr(LLMProvider, "embed")
        assert getattr(LLMProvider.embed, "__isabstractmethod__", False)

    def test_has_generate_method(self):
        """Should have generate abstract method."""
        assert hasattr(LLMProvider, "generate")
        assert getattr(LLMProvider.generate, "__isabstractmethod__", False)

    def test_has_rerank_method(self):
        """Should have rerank abstract method."""
        assert hasattr(LLMProvider, "rerank")
        assert getattr(LLMProvider.rerank, "__isabstractmethod__", False)

    def test_has_model_exists_method(self):
        """Should have model_exists abstract method."""
        assert hasattr(LLMProvider, "model_exists")
        assert getattr(LLMProvider.model_exists, "__isabstractmethod__", False)

    def test_has_is_available_method(self):
        """Should have is_available abstract method."""
        assert hasattr(LLMProvider, "is_available")
        assert getattr(LLMProvider.is_available, "__isabstractmethod__", False)

    def test_has_get_default_embedding_model(self):
        """Should have get_default_embedding_model abstract method."""
        assert hasattr(LLMProvider, "get_default_embedding_model")
        assert getattr(LLMProvider.get_default_embedding_model, "__isabstractmethod__", False)

    def test_has_get_default_expansion_model(self):
        """Should have get_default_expansion_model abstract method."""
        assert hasattr(LLMProvider, "get_default_expansion_model")
        assert getattr(LLMProvider.get_default_expansion_model, "__isabstractmethod__", False)

    def test_has_get_default_reranker_model(self):
        """Should have get_default_reranker_model abstract method."""
        assert hasattr(LLMProvider, "get_default_reranker_model")
        assert getattr(LLMProvider.get_default_reranker_model, "__isabstractmethod__", False)


class TestMockLLMProvider:
    """Tests for MockLLMProvider fixture (validates test infrastructure)."""

    @pytest.mark.asyncio
    async def test_mock_implements_interface(self, mock_llm_provider):
        """Mock provider should implement LLMProvider interface."""
        assert isinstance(mock_llm_provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_mock_embed_returns_result(self, mock_llm_provider):
        """Mock embed should return configured result."""
        result = await mock_llm_provider.embed("test text")

        assert result is not None
        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 768

    @pytest.mark.asyncio
    async def test_mock_embed_tracks_calls(self, mock_llm_provider):
        """Mock embed should track calls."""
        await mock_llm_provider.embed("text1", model="model1", is_query=True)
        await mock_llm_provider.embed("text2")

        assert len(mock_llm_provider.embed_calls) == 2
        assert mock_llm_provider.embed_calls[0] == ("text1", "model1", True)
        assert mock_llm_provider.embed_calls[1] == ("text2", None, False)

    @pytest.mark.asyncio
    async def test_mock_generate_returns_none_by_default(self, mock_llm_provider):
        """Mock generate should return None by default."""
        result = await mock_llm_provider.generate("prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_mock_generate_tracks_calls(self, mock_llm_provider):
        """Mock generate should track calls."""
        await mock_llm_provider.generate("prompt1", model="m1", max_tokens=100, temperature=0.5)

        assert len(mock_llm_provider.generate_calls) == 1
        assert mock_llm_provider.generate_calls[0] == ("prompt1", "m1", 100, 0.5)

    @pytest.mark.asyncio
    async def test_mock_rerank_returns_result(self, mock_llm_provider):
        """Mock rerank should return configured result."""
        result = await mock_llm_provider.rerank("query", [{"file": "doc.md", "body": "content"}])

        assert isinstance(result, RerankResult)

    @pytest.mark.asyncio
    async def test_mock_is_available(self, mock_llm_provider):
        """Mock is_available should return configured value."""
        assert await mock_llm_provider.is_available() is True

    @pytest.mark.asyncio
    async def test_mock_model_exists(self, mock_llm_provider):
        """Mock model_exists should return configured value."""
        assert await mock_llm_provider.model_exists("any-model") is True

    def test_mock_get_default_models(self, mock_llm_provider):
        """Mock should return default model names."""
        assert mock_llm_provider.get_default_embedding_model() == "mock-embed-model"
        assert mock_llm_provider.get_default_expansion_model() == "mock-expansion-model"
        assert mock_llm_provider.get_default_reranker_model() == "mock-reranker-model"
