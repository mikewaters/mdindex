"""Shared fixtures for LLM tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from pmd.core.config import LMStudioConfig, OpenRouterConfig, Config, ChunkConfig
from pmd.core.types import EmbeddingResult, RerankDocumentResult, RerankResult, RankedResult
from pmd.llm.base import LLMProvider


# Sentinel value to detect when None is explicitly passed
_UNSET = object()


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing dependent classes."""

    def __init__(
        self,
        embedding_result: EmbeddingResult | None = _UNSET,  # type: ignore
        generate_result: str | None = None,
        rerank_result: RerankResult | None = _UNSET,  # type: ignore
        is_available_result: bool = True,
        model_exists_result: bool = True,
    ):
        """Initialize mock provider with configurable responses.

        Use embedding_result=None to simulate LLM failure (returns None).
        Omit embedding_result to get default embedding response.
        """
        # Use sentinel to distinguish between "not provided" and "explicitly None"
        if embedding_result is _UNSET:
            self._embedding_result = EmbeddingResult(
                embedding=[0.1] * 768, model="mock-embed"
            )
        else:
            self._embedding_result = embedding_result

        self._generate_result = generate_result

        if rerank_result is _UNSET:
            self._rerank_result = RerankResult(results=[], model="mock-rerank")
        else:
            self._rerank_result = rerank_result

        self._is_available_result = is_available_result
        self._model_exists_result = model_exists_result

        # Track calls for assertions
        self.embed_calls: list[tuple] = []
        self.generate_calls: list[tuple] = []
        self.rerank_calls: list[tuple] = []

    async def embed(
        self,
        text: str,
        model: str | None = None,
        is_query: bool = False,
    ) -> EmbeddingResult | None:
        self.embed_calls.append((text, model, is_query))
        return self._embedding_result

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str | None:
        self.generate_calls.append((prompt, model, max_tokens, temperature))
        return self._generate_result

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        model: str | None = None,
    ) -> RerankResult:
        self.rerank_calls.append((query, documents, model))
        return self._rerank_result

    async def model_exists(self, model: str) -> bool:
        return self._model_exists_result

    async def is_available(self) -> bool:
        return self._is_available_result

    def get_default_embedding_model(self) -> str:
        return "mock-embed-model"

    def get_default_expansion_model(self) -> str:
        return "mock-expansion-model"

    def get_default_reranker_model(self) -> str:
        return "mock-reranker-model"

    async def close(self) -> None:
        """Close the provider (no-op for mock)."""
        pass


@pytest.fixture
def mock_llm_provider() -> MockLLMProvider:
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def lm_studio_config() -> LMStudioConfig:
    """Create LM Studio config for testing."""
    return LMStudioConfig(
        base_url="http://localhost:1234",
        embedding_model="nomic-embed-text",
        expansion_model="qwen2:0.5b",
        reranker_model="qwen2:0.5b",
        timeout=30.0,
    )


@pytest.fixture
def openrouter_config() -> OpenRouterConfig:
    """Create OpenRouter config for testing."""
    return OpenRouterConfig(
        api_key="test-api-key-12345",
        base_url="https://openrouter.io/api/v1",
        embedding_model="nomic-ai/nomic-embed-text",
        expansion_model="qwen/qwen-1.5-0.5b",
        reranker_model="qwen/qwen-1.5-0.5b",
        timeout=30.0,
    )


@pytest.fixture
def sample_config() -> Config:
    """Create a sample config for testing."""
    return Config(
        llm_provider="lm-studio",
        chunk=ChunkConfig(max_bytes=1000, min_chunk_size=50),
    )


@pytest.fixture
def sample_ranked_results() -> list[RankedResult]:
    """Create sample ranked results for testing."""
    return [
        RankedResult(
            file="doc1.md",
            display_path="doc1.md",
            title="Document 1",
            body="This is the first document about Python programming.",
            score=0.9,
            fts_score=0.85,
            vec_score=0.8,
        ),
        RankedResult(
            file="doc2.md",
            display_path="doc2.md",
            title="Document 2",
            body="This is the second document about JavaScript.",
            score=0.8,
            fts_score=0.75,
            vec_score=0.7,
        ),
        RankedResult(
            file="doc3.md",
            display_path="doc3.md",
            title="Document 3",
            body="This is the third document about Rust programming language.",
            score=0.7,
            fts_score=0.65,
            vec_score=0.6,
        ),
    ]


def make_embedding_response(embedding: list[float] | None = None, model: str = "test-model") -> dict:
    """Create a mock embedding API response."""
    if embedding is None:
        embedding = [0.1] * 768
    return {
        "data": [{"embedding": embedding}],
        "model": model,
    }


def make_chat_response(content: str, model: str = "test-model") -> dict:
    """Create a mock chat completion API response."""
    return {
        "choices": [{"message": {"content": content}}],
        "model": model,
    }


def make_models_response(models: list[str]) -> dict:
    """Create a mock models list API response."""
    return {
        "data": [{"id": m} for m in models],
    }


def make_rerank_document_result(
    filepath: str,
    relevant: bool,
    score: float | None = None,
) -> RerankDocumentResult:
    """Create a rerank document result for testing."""
    if score is None:
        score = 0.95 if relevant else 0.45
    return RerankDocumentResult(
        file=filepath,
        relevant=relevant,
        confidence=0.9 if relevant else 0.1,
        score=score,
        raw_token="yes" if relevant else "no",
    )
