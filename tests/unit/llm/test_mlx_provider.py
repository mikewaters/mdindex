"""Tests for MLX LLM provider."""

import sys
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from pmd.core.config import MLXConfig
from pmd.core.types import RerankResult


# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX provider only available on macOS"
)


@pytest.fixture
def mlx_config() -> MLXConfig:
    """Create MLX config for testing."""
    return MLXConfig(
        model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        max_tokens=100,
        temperature=0.7,
        lazy_load=True,
    )


@pytest.fixture
def mock_mlx_lm():
    """Mock the mlx_lm and mlx_embeddings modules."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    mock_embedding_model = MagicMock()
    # Mock embedding result as numpy-like array
    import numpy as np
    mock_embedding_result = np.array([[0.1] * 384])  # Typical small embedding

    with patch.dict("sys.modules", {
        "mlx_lm": MagicMock(),
        "mlx_embeddings": MagicMock(),
    }):
        import sys
        mock_lm_module = sys.modules["mlx_lm"]
        mock_lm_module.load.return_value = (mock_model, mock_tokenizer)
        mock_lm_module.generate.return_value = "Generated response"

        mock_embed_module = sys.modules["mlx_embeddings"]
        mock_embed_module.load.return_value = mock_embedding_model
        mock_embed_module.embed.return_value = mock_embedding_result

        yield mock_lm_module, mock_embed_module, mock_model, mock_tokenizer, mock_embedding_model


class TestMLXProviderInit:
    """Tests for MLXProvider initialization."""

    def test_stores_config(self, mlx_config, mock_mlx_lm):
        """Should store configuration."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)

        assert provider.config == mlx_config

    def test_lazy_load_does_not_load_model(self, mlx_config, mock_mlx_lm):
        """Lazy load should not load model immediately."""
        from pmd.llm.mlx_provider import MLXProvider

        mlx_config.lazy_load = True
        provider = MLXProvider(mlx_config)

        assert provider._model is None
        assert provider._model_loaded is False

    def test_eager_load_loads_model(self, mlx_config, mock_mlx_lm):
        """Non-lazy load should load model immediately."""
        from pmd.llm.mlx_provider import MLXProvider

        mlx_config.lazy_load = False
        provider = MLXProvider(mlx_config)

        assert provider._model_loaded is True


class TestMLXProviderDefaultModels:
    """Tests for default model getters."""

    def test_get_default_embedding_model(self, mlx_config, mock_mlx_lm):
        """Should return configured embedding model."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)
        assert provider.get_default_embedding_model() == mlx_config.embedding_model

    def test_get_default_expansion_model(self, mlx_config, mock_mlx_lm):
        """Should return configured model."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)
        assert provider.get_default_expansion_model() == mlx_config.model

    def test_get_default_reranker_model(self, mlx_config, mock_mlx_lm):
        """Should return configured model."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)
        assert provider.get_default_reranker_model() == mlx_config.model


class TestMLXProviderEmbed:
    """Tests for MLXProvider.embed method."""

    @pytest.mark.asyncio
    async def test_embed_returns_embedding_result(self, mlx_config, mock_mlx_lm):
        """embed should return EmbeddingResult."""
        from pmd.llm.mlx_provider import MLXProvider
        from pmd.core.types import EmbeddingResult

        provider = MLXProvider(mlx_config)
        result = await provider.embed("test text")

        assert result is not None
        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 384  # Size from mock

    @pytest.mark.asyncio
    async def test_embed_loads_model_on_first_call(self, mlx_config, mock_mlx_lm):
        """embed should load embedding model on first call."""
        from pmd.llm.mlx_provider import MLXProvider

        _, mock_embed_module, _, _, _ = mock_mlx_lm
        provider = MLXProvider(mlx_config)

        assert not provider._embedding_model_loaded

        await provider.embed("test text")

        assert provider._embedding_model_loaded
        mock_embed_module.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_formats_query_text(self, mlx_config, mock_mlx_lm):
        """embed should add 'query:' prefix for queries."""
        from pmd.llm.mlx_provider import MLXProvider

        _, mock_embed_module, _, _, _ = mock_mlx_lm
        provider = MLXProvider(mlx_config)

        await provider.embed("search terms", is_query=True)

        # Check that embed was called with query prefix
        call_args = mock_embed_module.embed.call_args
        assert "query:" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_embed_formats_passage_text(self, mlx_config, mock_mlx_lm):
        """embed should add 'passage:' prefix for documents."""
        from pmd.llm.mlx_provider import MLXProvider

        _, mock_embed_module, _, _, _ = mock_mlx_lm
        provider = MLXProvider(mlx_config)

        await provider.embed("document content", is_query=False)

        # Check that embed was called with passage prefix
        call_args = mock_embed_module.embed.call_args
        assert "passage:" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_embed_returns_none_on_error(self, mlx_config, mock_mlx_lm):
        """embed should return None on error."""
        from pmd.llm.mlx_provider import MLXProvider

        _, mock_embed_module, _, _, _ = mock_mlx_lm
        mock_embed_module.embed.side_effect = Exception("Model error")

        provider = MLXProvider(mlx_config)
        result = await provider.embed("test")

        assert result is None

    @pytest.mark.asyncio
    async def test_embed_includes_model_name(self, mlx_config, mock_mlx_lm):
        """embed result should include model name."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)
        result = await provider.embed("test")

        assert result.model == mlx_config.embedding_model


class TestMLXProviderGenerate:
    """Tests for MLXProvider.generate method."""

    @pytest.mark.asyncio
    async def test_generate_returns_text(self, mlx_config, mock_mlx_lm):
        """generate should return generated text."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        mock_lm_module.generate.return_value = "Generated response"

        provider = MLXProvider(mlx_config)
        result = await provider.generate("test prompt")

        assert result == "Generated response"

    @pytest.mark.asyncio
    async def test_generate_loads_model_on_first_call(self, mlx_config, mock_mlx_lm):
        """generate should load model on first call."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        provider = MLXProvider(mlx_config)

        assert not provider._model_loaded

        await provider.generate("prompt")

        assert provider._model_loaded
        mock_lm_module.load.assert_called_once_with(mlx_config.model)

    @pytest.mark.asyncio
    async def test_generate_applies_chat_template(self, mlx_config, mock_mlx_lm):
        """generate should apply chat template for instruction models."""
        from pmd.llm.mlx_provider import MLXProvider

        _, _, _, mock_tokenizer, _ = mock_mlx_lm
        provider = MLXProvider(mlx_config)

        await provider.generate("test prompt")

        mock_tokenizer.apply_chat_template.assert_called()

    @pytest.mark.asyncio
    async def test_generate_returns_none_on_error(self, mlx_config, mock_mlx_lm):
        """generate should return None on error."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        mock_lm_module.generate.side_effect = Exception("Model error")

        provider = MLXProvider(mlx_config)
        result = await provider.generate("prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_strips_whitespace(self, mlx_config, mock_mlx_lm):
        """generate should strip whitespace from response."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        mock_lm_module.generate.return_value = "  response with spaces  \n"

        provider = MLXProvider(mlx_config)
        result = await provider.generate("prompt")

        assert result == "response with spaces"


class TestMLXProviderRerank:
    """Tests for MLXProvider.rerank method."""

    @pytest.mark.asyncio
    async def test_rerank_returns_result(self, mlx_config, mock_mlx_lm):
        """rerank should return RerankResult."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        mock_lm_module.generate.return_value = "Yes"

        provider = MLXProvider(mlx_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert isinstance(result, RerankResult)
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_rerank_yes_is_relevant(self, mlx_config, mock_mlx_lm):
        """rerank should mark 'Yes' response as relevant."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        mock_lm_module.generate.return_value = "Yes"

        provider = MLXProvider(mlx_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert result.results[0].relevant is True
        assert result.results[0].score > 0.5

    @pytest.mark.asyncio
    async def test_rerank_no_is_not_relevant(self, mlx_config, mock_mlx_lm):
        """rerank should mark 'No' response as not relevant."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        mock_lm_module.generate.return_value = "No"

        provider = MLXProvider(mlx_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert result.results[0].relevant is False
        assert result.results[0].score < 0.5

    @pytest.mark.asyncio
    async def test_rerank_multiple_documents(self, mlx_config, mock_mlx_lm):
        """rerank should handle multiple documents."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        mock_lm_module.generate.side_effect = ["Yes", "No", "Yes"]

        provider = MLXProvider(mlx_config)
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
    async def test_rerank_handles_error_gracefully(self, mlx_config, mock_mlx_lm):
        """rerank should return neutral score on error."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        mock_lm_module.generate.side_effect = Exception("Error")

        provider = MLXProvider(mlx_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert result.results[0].score == 0.5
        assert result.results[0].raw_token == "error"

    @pytest.mark.asyncio
    async def test_rerank_includes_model_name(self, mlx_config, mock_mlx_lm):
        """rerank result should include model name."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        mock_lm_module.generate.return_value = "Yes"

        provider = MLXProvider(mlx_config)
        docs = [{"file": "doc.md", "body": "content"}]
        result = await provider.rerank("query", docs)

        assert result.model == mlx_config.model


class TestMLXProviderModelExists:
    """Tests for MLXProvider.model_exists method."""

    @pytest.mark.asyncio
    async def test_model_exists_with_slash(self, mlx_config, mock_mlx_lm):
        """model_exists should return True for HF model IDs."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)
        result = await provider.model_exists("mlx-community/some-model")

        assert result is True

    @pytest.mark.asyncio
    async def test_model_exists_configured_model(self, mlx_config, mock_mlx_lm):
        """model_exists should return True for configured model."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)
        result = await provider.model_exists(mlx_config.model)

        assert result is True

    @pytest.mark.asyncio
    async def test_model_exists_invalid(self, mlx_config, mock_mlx_lm):
        """model_exists should return False for invalid model."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)
        result = await provider.model_exists("not-a-valid-id")

        assert result is False


class TestMLXProviderIsAvailable:
    """Tests for MLXProvider.is_available method."""

    @pytest.mark.asyncio
    async def test_is_available_on_macos(self, mlx_config, mock_mlx_lm):
        """is_available should return True on macOS with mlx-lm."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)
        result = await provider.is_available()

        assert result is True


class TestMLXProviderUnloadModel:
    """Tests for MLXProvider.unload_model method."""

    def test_unload_model(self, mlx_config, mock_mlx_lm):
        """unload_model should clear model from memory."""
        from pmd.llm.mlx_provider import MLXProvider

        mlx_config.lazy_load = False
        provider = MLXProvider(mlx_config)

        assert provider._model_loaded is True

        provider.unload_model()

        assert provider._model is None
        assert provider._tokenizer is None
        assert provider._model_loaded is False


class TestMLXProviderFactoryIntegration:
    """Tests for MLX provider factory integration."""

    def test_factory_creates_mlx_provider(self, mlx_config, mock_mlx_lm):
        """Factory should create MLXProvider for 'mlx' provider."""
        from pmd.core.config import Config
        from pmd.llm.factory import create_llm_provider
        from pmd.llm.mlx_provider import MLXProvider

        config = Config(llm_provider="mlx")
        provider = create_llm_provider(config)

        assert isinstance(provider, MLXProvider)

    def test_factory_case_insensitive(self, mlx_config, mock_mlx_lm):
        """Factory should handle case-insensitive provider name."""
        from pmd.core.config import Config
        from pmd.llm.factory import create_llm_provider
        from pmd.llm.mlx_provider import MLXProvider

        config = Config(llm_provider="MLX")
        provider = create_llm_provider(config)

        assert isinstance(provider, MLXProvider)

    def test_get_provider_name(self, mock_mlx_lm):
        """get_provider_name should return 'MLX (Local)'."""
        from pmd.core.config import Config
        from pmd.llm.factory import get_provider_name

        config = Config(llm_provider="mlx")
        name = get_provider_name(config)

        assert name == "MLX (Local)"
