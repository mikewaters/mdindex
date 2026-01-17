"""Integration tests for embedding generation using MLX.

These tests actually invoke mlx_embeddings to generate real embeddings,
downloading the model if needed. They require macOS with Apple Silicon.
"""

import sys
import pytest
from pathlib import Path

from pmd.core.config import MLXConfig
from pmd.core.types import EmbeddingResult
from pmd.store.database import Database
from pmd.store.repositories.collections import CollectionRepository
from pmd.store.repositories.documents import DocumentRepository
from pmd.store.repositories.embeddings import EmbeddingRepository


# Skip all tests in this module if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX embeddings require macOS with Apple Silicon"
)


@pytest.fixture(scope="module")
def mlx_config() -> MLXConfig:
    """Provide MLX configuration for embeddings.

    Uses nomic/modernbert-embed-base for high-quality embeddings.
    """
    return MLXConfig(
        embedding_model="mlx-community/nomicai-modernbert-embed-base-4bit",
        embedding_dimension=768,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        lazy_load=True,
    )


@pytest.fixture(scope="module")
def mlx_provider(mlx_config: MLXConfig):
    """Provide an MLX provider instance.

    Module-scoped to avoid reloading the model for each test.
    """
    from pmd.llm.mlx_provider import MLXProvider

    provider = MLXProvider(mlx_config)
    yield provider
    # Clean up - unload models to free memory
    provider.unload_all()


class TestMLXEmbeddingGeneration:
    """Tests for generating embeddings with MLX."""

    @pytest.mark.asyncio
    async def test_embed_simple_text(self, mlx_provider):
        """Should generate embeddings for simple text."""
        result = await mlx_provider.embed("Hello, world!")

        assert result is not None
        assert isinstance(result, EmbeddingResult)
        assert isinstance(result.embedding, list)
        assert len(result.embedding) > 0
        # ModernBERT-embed-base has 768 dimensions
        assert len(result.embedding) == 768

    @pytest.mark.asyncio
    async def test_embed_returns_floats(self, mlx_provider):
        """Embeddings should be lists of floats."""
        result = await mlx_provider.embed("Test embedding")

        assert result is not None
        assert all(isinstance(x, float) for x in result.embedding)

    @pytest.mark.asyncio
    async def test_embed_query_vs_passage(self, mlx_provider):
        """Query and passage embeddings should differ."""
        query_result = await mlx_provider.embed("machine learning", is_query=True)
        passage_result = await mlx_provider.embed("machine learning", is_query=False)

        assert query_result is not None
        assert passage_result is not None
        # E5 models use different prefixes for query vs passage
        # The embeddings should be different
        assert query_result.embedding != passage_result.embedding

    @pytest.mark.asyncio
    async def test_embed_similar_texts_have_similar_embeddings(self, mlx_provider):
        """Similar texts should have similar embeddings."""
        result1 = await mlx_provider.embed("The cat sat on the mat")
        result2 = await mlx_provider.embed("A cat was sitting on a mat")
        result3 = await mlx_provider.embed("Python is a programming language")

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None

        # Calculate cosine similarity
        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        sim_12 = cosine_similarity(result1.embedding, result2.embedding)
        sim_13 = cosine_similarity(result1.embedding, result3.embedding)

        # Similar sentences should have higher similarity than unrelated ones
        assert sim_12 > sim_13

    @pytest.mark.asyncio
    async def test_embed_empty_string(self, mlx_provider):
        """Should handle empty strings gracefully."""
        result = await mlx_provider.embed("")

        # Should either return a result or None, not crash
        if result is not None:
            assert isinstance(result.embedding, list)

    @pytest.mark.asyncio
    async def test_embed_long_text(self, mlx_provider):
        """Should handle longer texts."""
        long_text = "This is a test sentence. " * 100
        result = await mlx_provider.embed(long_text)

        assert result is not None
        assert len(result.embedding) == 768

    @pytest.mark.asyncio
    async def test_embed_unicode_text(self, mlx_provider):
        """Should handle unicode text properly."""
        result = await mlx_provider.embed("日本語のテスト 🎉 emoji test")

        assert result is not None
        assert len(result.embedding) == 768


class TestEmbeddingStorage:
    """Tests for storing and retrieving embeddings."""

    @pytest.fixture
    def integration_db(self, tmp_path: Path, mlx_config: MLXConfig) -> Database:
        """Provide a connected database for embedding tests."""
        db_path = tmp_path / "embedding_test.db"
        database = Database(db_path, embedding_dimension=mlx_config.embedding_dimension)
        database.connect()
        yield database
        database.close()

    @pytest.fixture
    def embedding_repo(self, integration_db: Database) -> EmbeddingRepository:
        """Provide an EmbeddingRepository instance."""
        return EmbeddingRepository(integration_db)

    @pytest.fixture
    def document_repo(self, integration_db: Database) -> DocumentRepository:
        """Provide a DocumentRepository instance."""
        return DocumentRepository(integration_db)

    @pytest.fixture
    def collection_repo(self, integration_db: Database) -> CollectionRepository:
        """Provide a CollectionRepository instance."""
        return CollectionRepository(integration_db)

    @pytest.mark.asyncio
    async def test_store_and_check_embedding(
        self,
        mlx_provider,
        embedding_repo: EmbeddingRepository,
        document_repo: DocumentRepository,
        collection_repo: CollectionRepository,
        tmp_path: Path,
    ):
        """Should store embeddings and verify they exist."""
        # Create a collection and document
        collection = collection_repo.create("test", str(tmp_path), "**/*.md")
        doc_result, _ = document_repo.add_or_update(
            collection.id,
            "test.md",
            "Test Document",
            "This is test content for embedding.",
        )

        # Generate embedding
        result = await mlx_provider.embed("This is test content for embedding.")
        assert result is not None

        # Store embedding
        embedding_repo.store_embedding(
            doc_result.hash,
            seq=0,
            pos=0,
            embedding=result.embedding,
            model=mlx_provider.get_default_embedding_model(),
        )

        # Verify it exists
        assert embedding_repo.has_embeddings(doc_result.hash)

    @pytest.mark.asyncio
    async def test_store_multiple_chunks(
        self,
        mlx_provider,
        embedding_repo: EmbeddingRepository,
        document_repo: DocumentRepository,
        collection_repo: CollectionRepository,
        tmp_path: Path,
    ):
        """Should store multiple embedding chunks for a document."""
        collection = collection_repo.create("chunks", str(tmp_path), "**/*.md")
        doc_result, _ = document_repo.add_or_update(
            collection.id,
            "chunked.md",
            "Chunked Document",
            "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        )

        # Generate and store embeddings for each chunk
        chunks = ["First paragraph.", "Second paragraph.", "Third paragraph."]
        for i, chunk in enumerate(chunks):
            result = await mlx_provider.embed(chunk)
            assert result is not None

            embedding_repo.store_embedding(
                doc_result.hash,
                seq=i,
                pos=i * 20,  # Approximate positions
                embedding=result.embedding,
                model=mlx_provider.get_default_embedding_model(),
            )

        # Verify all chunks are stored
        stored = embedding_repo.get_embeddings_for_content(doc_result.hash)
        assert len(stored) == 3
        assert [s[0] for s in stored] == [0, 1, 2]  # seq numbers

    @pytest.mark.asyncio
    async def test_count_embeddings(
        self,
        mlx_provider,
        embedding_repo: EmbeddingRepository,
        document_repo: DocumentRepository,
        collection_repo: CollectionRepository,
        tmp_path: Path,
    ):
        """Should correctly count embeddings."""
        collection = collection_repo.create("count", str(tmp_path), "**/*.md")

        initial_count = embedding_repo.count_embeddings()

        # Add some documents with embeddings
        for i in range(3):
            doc_result, _ = document_repo.add_or_update(
                collection.id,
                f"doc{i}.md",
                f"Document {i}",
                f"Content for document {i}",
            )

            result = await mlx_provider.embed(f"Content for document {i}")
            assert result is not None

            embedding_repo.store_embedding(
                doc_result.hash,
                seq=0,
                pos=0,
                embedding=result.embedding,
                model=mlx_provider.get_default_embedding_model(),
            )

        final_count = embedding_repo.count_embeddings()
        assert final_count == initial_count + 3

    @pytest.mark.asyncio
    async def test_delete_embeddings(
        self,
        mlx_provider,
        embedding_repo: EmbeddingRepository,
        document_repo: DocumentRepository,
        collection_repo: CollectionRepository,
        tmp_path: Path,
    ):
        """Should delete embeddings for a document."""
        collection = collection_repo.create("delete", str(tmp_path), "**/*.md")
        doc_result, _ = document_repo.add_or_update(
            collection.id,
            "deleteme.md",
            "Delete Me",
            "Content to be deleted.",
        )

        result = await mlx_provider.embed("Content to be deleted.")
        assert result is not None

        embedding_repo.store_embedding(
            doc_result.hash,
            seq=0,
            pos=0,
            embedding=result.embedding,
            model=mlx_provider.get_default_embedding_model(),
        )

        assert embedding_repo.has_embeddings(doc_result.hash)

        # Delete
        deleted_count = embedding_repo.delete_embeddings(doc_result.hash)

        assert deleted_count == 1
        assert not embedding_repo.has_embeddings(doc_result.hash)


class TestEmbeddingDimensions:
    """Tests for embedding dimension handling."""

    @pytest.mark.asyncio
    async def test_embedding_dimension_consistency(self, mlx_provider):
        """All embeddings from the same model should have the same dimension."""
        texts = [
            "Short",
            "A medium length sentence for testing",
            "A much longer piece of text that contains many words and should test "
            "whether the embedding dimension remains constant regardless of input length",
        ]

        dimensions = []
        for text in texts:
            result = await mlx_provider.embed(text)
            assert result is not None
            dimensions.append(len(result.embedding))

        # All dimensions should be the same
        assert len(set(dimensions)) == 1

    @pytest.mark.asyncio
    async def test_embedding_values_normalized(self, mlx_provider):
        """Embeddings should have reasonable value ranges."""
        result = await mlx_provider.embed("Test normalization")

        assert result is not None

        # Check that values are in a reasonable range (typically -1 to 1 or so)
        max_val = max(abs(x) for x in result.embedding)
        assert max_val < 100  # Sanity check - values shouldn't be huge
