"""Tests for EmbeddingGenerator."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from pmd.llm.embeddings import EmbeddingGenerator
from pmd.core.config import Config, ChunkConfig
from pmd.core.types import EmbeddingResult
from pmd.store.database import Database
from pmd.store.repositories.embeddings import EmbeddingRepository

from .conftest import MockLLMProvider


@pytest.fixture
def db(tmp_path: Path) -> Database:
    """Create test database."""
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    database.connect()
    yield database
    database.close()


@pytest.fixture
def embedding_repo(db: Database) -> EmbeddingRepository:
    """Create embedding repository for testing."""
    return EmbeddingRepository(db)


@pytest.fixture
def embedding_generator(
    mock_llm_provider: MockLLMProvider,
    embedding_repo: EmbeddingRepository,
    sample_config: Config,
) -> EmbeddingGenerator:
    """Create embedding generator for testing."""
    return EmbeddingGenerator(mock_llm_provider, embedding_repo, sample_config)


class TestEmbeddingGeneratorInit:
    """Tests for EmbeddingGenerator initialization."""

    def test_stores_llm_provider(self, embedding_generator, mock_llm_provider):
        """Should store LLM provider."""
        assert embedding_generator.llm == mock_llm_provider

    def test_stores_embedding_repo(self, embedding_generator, embedding_repo):
        """Should store embedding repository."""
        assert embedding_generator.embedding_repo == embedding_repo

    def test_stores_config(self, embedding_generator, sample_config):
        """Should store configuration."""
        assert embedding_generator.config == sample_config

    def test_gets_default_model_from_provider(self, embedding_generator):
        """Should get default embedding model from provider."""
        assert embedding_generator.model == "mock-embed-model"


class TestEmbedDocument:
    """Tests for EmbeddingGenerator.embed_document method."""

    @pytest.mark.asyncio
    async def test_embed_document_returns_chunk_count(self, embedding_generator):
        """embed_document should return number of chunks embedded."""
        content = "Short content for embedding"

        count = await embedding_generator.embed_document("hash123", content)

        assert count == 1

    @pytest.mark.asyncio
    async def test_embed_document_calls_llm(self, embedding_generator, mock_llm_provider):
        """embed_document should call LLM embed for each chunk."""
        content = "Content to embed"

        await embedding_generator.embed_document("hash123", content)

        assert len(mock_llm_provider.embed_calls) >= 1
        # Should be called as document (not query)
        assert mock_llm_provider.embed_calls[0][2] is False

    @pytest.mark.asyncio
    async def test_embed_document_stores_embeddings(self, embedding_generator, embedding_repo):
        """embed_document should store embeddings in repository."""
        content = "Content to embed"

        await embedding_generator.embed_document("hash123", content)

        assert embedding_repo.has_embeddings("hash123") is True

    @pytest.mark.asyncio
    async def test_embed_document_skips_existing(self, embedding_generator, mock_llm_provider):
        """embed_document should skip if embeddings exist."""
        content = "Content"

        # First call
        await embedding_generator.embed_document("hash123", content)
        call_count_1 = len(mock_llm_provider.embed_calls)

        # Second call should skip
        result = await embedding_generator.embed_document("hash123", content)

        assert result == 0
        assert len(mock_llm_provider.embed_calls) == call_count_1

    @pytest.mark.asyncio
    async def test_embed_document_force_regenerates(
        self, embedding_generator, mock_llm_provider
    ):
        """embed_document with force=True should regenerate."""
        content = "Content"

        await embedding_generator.embed_document("hash123", content)
        call_count_1 = len(mock_llm_provider.embed_calls)

        await embedding_generator.embed_document("hash123", content, force=True)

        assert len(mock_llm_provider.embed_calls) > call_count_1

    @pytest.mark.asyncio
    async def test_embed_document_handles_large_content(self, embedding_generator):
        """embed_document should chunk large content."""
        # Create content larger than max_bytes (1000 in sample_config)
        lines = ["This is line number " + str(i) + " with some content" for i in range(100)]
        content = "\n".join(lines)

        count = await embedding_generator.embed_document("hash123", content)

        # Should create multiple chunks
        assert count > 1

    @pytest.mark.asyncio
    async def test_embed_document_handles_llm_failure(self, embedding_repo, sample_config):
        """embed_document should handle LLM failure gracefully."""
        mock_provider = MockLLMProvider(embedding_result=None)
        generator = EmbeddingGenerator(mock_provider, embedding_repo, sample_config)

        count = await generator.embed_document("hash123", "content")

        assert count == 0


class TestEmbedQuery:
    """Tests for EmbeddingGenerator.embed_query method."""

    @pytest.mark.asyncio
    async def test_embed_query_returns_embedding(self, embedding_generator):
        """embed_query should return embedding vector."""
        result = await embedding_generator.embed_query("search query")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_query_calls_llm_as_query(
        self, embedding_generator, mock_llm_provider
    ):
        """embed_query should call LLM with is_query=True."""
        await embedding_generator.embed_query("query")

        assert len(mock_llm_provider.embed_calls) == 1
        assert mock_llm_provider.embed_calls[0][2] is True  # is_query

    @pytest.mark.asyncio
    async def test_embed_query_returns_none_on_failure(self, embedding_repo, sample_config):
        """embed_query should return None on LLM failure."""
        mock_provider = MockLLMProvider(embedding_result=None)
        generator = EmbeddingGenerator(mock_provider, embedding_repo, sample_config)

        result = await generator.embed_query("query")

        assert result is None


class TestGetEmbeddingsForContent:
    """Tests for EmbeddingGenerator.get_embeddings_for_content method."""

    @pytest.mark.asyncio
    async def test_get_embeddings_for_content(self, embedding_generator):
        """get_embeddings_for_content should return stored metadata."""
        await embedding_generator.embed_document("hash123", "content")

        results = embedding_generator.get_embeddings_for_content("hash123")

        assert len(results) >= 1
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2  # (seq, pos)

    def test_get_embeddings_for_nonexistent(self, embedding_generator):
        """get_embeddings_for_content should return empty for missing hash."""
        results = embedding_generator.get_embeddings_for_content("nonexistent")

        assert results == []


class TestClearEmbeddingsByModel:
    """Tests for EmbeddingGenerator.clear_embeddings_by_model method."""

    @pytest.mark.asyncio
    async def test_clear_embeddings_by_model(self, embedding_generator, embedding_repo):
        """clear_embeddings_by_model should remove embeddings."""
        await embedding_generator.embed_document("hash1", "content1")
        await embedding_generator.embed_document("hash2", "content2", force=True)

        count = await embedding_generator.clear_embeddings_by_model("mock-embed-model")

        assert count >= 2
        assert embedding_repo.has_embeddings("hash1") is False


class TestGetEmbeddingCount:
    """Tests for embedding count methods."""

    def test_get_embedding_count_empty(self, embedding_generator):
        """get_embedding_count should return 0 when empty."""
        count = embedding_generator.get_embedding_count()

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_embedding_count(self, embedding_generator):
        """get_embedding_count should return total count."""
        await embedding_generator.embed_document("hash1", "content1")
        await embedding_generator.embed_document("hash2", "content2", force=True)

        count = embedding_generator.get_embedding_count()

        assert count >= 2

    @pytest.mark.asyncio
    async def test_get_embedding_count_by_model(self, embedding_generator):
        """get_embedding_count_by_model should filter by model."""
        await embedding_generator.embed_document("hash1", "content1")

        count = embedding_generator.get_embedding_count_by_model("mock-embed-model")
        other_count = embedding_generator.get_embedding_count_by_model("other-model")

        assert count >= 1
        assert other_count == 0


class TestEmbedDocumentChunking:
    """Tests for document chunking during embedding."""

    @pytest.mark.asyncio
    async def test_stores_chunk_positions(self, embedding_generator, embedding_repo):
        """embed_document should store correct chunk positions."""
        # Large content that will be split
        lines = ["Line " + str(i) + " content" for i in range(50)]
        content = "\n".join(lines)

        await embedding_generator.embed_document("hash123", content)

        embeddings = embedding_repo.get_embeddings_for_content("hash123")
        positions = [e[1] for e in embeddings]  # pos is second element

        # First chunk should start at 0
        assert positions[0] == 0

        # Subsequent positions should increase
        if len(positions) > 1:
            for i in range(1, len(positions)):
                assert positions[i] > positions[i - 1]

    @pytest.mark.asyncio
    async def test_stores_sequence_numbers(self, embedding_generator, embedding_repo):
        """embed_document should store sequential chunk numbers."""
        lines = ["Line " + str(i) + " with content" for i in range(50)]
        content = "\n".join(lines)

        await embedding_generator.embed_document("hash123", content)

        embeddings = embedding_repo.get_embeddings_for_content("hash123")
        seqs = [e[0] for e in embeddings]

        # Should be sequential starting from 0
        expected_seqs = list(range(len(seqs)))
        assert seqs == expected_seqs
