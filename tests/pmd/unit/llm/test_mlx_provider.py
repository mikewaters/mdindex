"""Tests for MLX LLM provider."""

import os
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
    mock_embedding_tokenizer = MagicMock()

    # Mock embedding result with text_embeds attribute (primary for ModernBERT)
    mock_embedding_result = MagicMock()
    # text_embeds is the primary attribute for normalized embeddings
    mock_embedding_result.text_embeds = MagicMock()
    mock_embedding_result.text_embeds.tolist.return_value = [[0.1] * 768]
    # Set other attributes to None to simulate ModernBERT behavior
    mock_embedding_result.pooler_output = None
    mock_embedding_result.last_hidden_state = None

    # Create mock for sample_utils submodule
    mock_sample_utils = MagicMock()
    mock_sample_utils.make_sampler.return_value = MagicMock()

    with patch.dict("sys.modules", {
        "mlx_lm": MagicMock(),
        "mlx_lm.sample_utils": mock_sample_utils,
        "mlx_embeddings": MagicMock(),
    }):
        import sys
        mock_lm_module = sys.modules["mlx_lm"]
        mock_lm_module.load.return_value = (mock_model, mock_tokenizer)
        mock_lm_module.generate.return_value = "Generated response"

        mock_embed_module = sys.modules["mlx_embeddings"]
        # load() returns (model, tokenizer) tuple
        mock_embed_module.load.return_value = (mock_embedding_model, mock_embedding_tokenizer)
        # Uses generate() not embed()
        mock_embed_module.generate.return_value = mock_embedding_result

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
        assert len(result.embedding) == 768  # Size from mock

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

        # Check that generate was called with query prefix (from config)
        call_args = mock_embed_module.generate.call_args
        assert mlx_config.query_prefix in call_args[0][2]

    @pytest.mark.asyncio
    async def test_embed_formats_passage_text(self, mlx_config, mock_mlx_lm):
        """embed should add document prefix for documents."""
        from pmd.llm.mlx_provider import MLXProvider

        _, mock_embed_module, _, _, _ = mock_mlx_lm
        provider = MLXProvider(mlx_config)

        await provider.embed("document content", is_query=False)

        # Check that generate was called with document prefix (from config)
        call_args = mock_embed_module.generate.call_args
        assert mlx_config.document_prefix in call_args[0][2]

    @pytest.mark.asyncio
    async def test_embed_returns_none_on_error(self, mlx_config, mock_mlx_lm):
        """embed should return None on error."""
        from pmd.llm.mlx_provider import MLXProvider

        _, mock_embed_module, _, _, _ = mock_mlx_lm
        mock_embed_module.generate.side_effect = Exception("Model error")

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

        # Mock _get_hf_token to return None for simpler assertion
        with patch.object(provider, "_get_hf_token", return_value=None):
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


class TestMLXProviderHuggingFaceToken:
    """Tests for HuggingFace token support."""

    def test_get_hf_token_from_env(self, mlx_config, mock_mlx_lm):
        """Should get token from HF_TOKEN environment variable."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)

        with patch.dict("os.environ", {"HF_TOKEN": "test-token-123"}):
            token = provider._get_hf_token()

        assert token == "test-token-123"

    def test_get_hf_token_from_cached_login(self, mlx_config, mock_mlx_lm):
        """Should get token from huggingface-cli cached login."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)

        # Remove HF_TOKEN but keep other env vars
        env_without_token = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict("os.environ", env_without_token, clear=True):
            with patch("huggingface_hub.HfFolder.get_token", return_value="cached-token"):
                token = provider._get_hf_token()

        assert token == "cached-token"

    def test_get_hf_token_env_takes_priority(self, mlx_config, mock_mlx_lm):
        """HF_TOKEN env var should take priority over cached login."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)

        with patch.dict("os.environ", {"HF_TOKEN": "env-token"}):
            with patch("huggingface_hub.HfFolder.get_token", return_value="cached-token"):
                token = provider._get_hf_token()

        assert token == "env-token"

    def test_get_hf_token_returns_none_when_not_available(self, mlx_config, mock_mlx_lm):
        """Should return None when no token available."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)

        # Remove HF_TOKEN but keep other env vars, and mock HfFolder to raise exception
        env_without_token = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict("os.environ", env_without_token, clear=True):
            with patch("huggingface_hub.HfFolder.get_token", side_effect=Exception("No token")):
                token = provider._get_hf_token()

        assert token is None

    def test_model_load_passes_token(self, mlx_config, mock_mlx_lm):
        """Model load should pass token when available."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        provider = MLXProvider(mlx_config)

        with patch.object(provider, "_get_hf_token", return_value="my-token"):
            provider._ensure_model_loaded()

        # Check that load was called with tokenizer_config containing the token
        mock_lm_module.load.assert_called_once()
        call_kwargs = mock_lm_module.load.call_args
        assert call_kwargs[1].get("tokenizer_config", {}).get("token") == "my-token"

    def test_model_load_without_token(self, mlx_config, mock_mlx_lm):
        """Model load should work without token."""
        from pmd.llm.mlx_provider import MLXProvider

        mock_lm_module, _, _, _, _ = mock_mlx_lm
        provider = MLXProvider(mlx_config)

        with patch.object(provider, "_get_hf_token", return_value=None):
            provider._ensure_model_loaded()

        # Check that load was called with just the model name
        mock_lm_module.load.assert_called_once_with(mlx_config.model)

    def test_embedding_model_load_passes_token(self, mlx_config, mock_mlx_lm):
        """Embedding model load should pass token when available."""
        from pmd.llm.mlx_provider import MLXProvider

        _, mock_embed_module, _, _, _ = mock_mlx_lm
        provider = MLXProvider(mlx_config)

        with patch.object(provider, "_get_hf_token", return_value="embed-token"):
            provider._ensure_embedding_model_loaded()

        # Check that load was called with tokenizer_config containing token
        mock_embed_module.load.assert_called_once()
        call_kwargs = mock_embed_module.load.call_args
        assert call_kwargs[1].get("tokenizer_config", {}).get("token") == "embed-token"

    def test_embedding_model_load_without_token(self, mlx_config, mock_mlx_lm):
        """Embedding model load should work without token."""
        from pmd.llm.mlx_provider import MLXProvider

        _, mock_embed_module, _, _, _ = mock_mlx_lm
        provider = MLXProvider(mlx_config)

        with patch.object(provider, "_get_hf_token", return_value=None):
            provider._ensure_embedding_model_loaded()

        # Check that load was called with model and empty tokenizer_config
        mock_embed_module.load.assert_called_once()
        call_args = mock_embed_module.load.call_args
        assert call_args[0][0] == mlx_config.embedding_model
        assert call_args[1].get("tokenizer_config", {}) == {}


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
