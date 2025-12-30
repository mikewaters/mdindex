"""Tests for embedding storage and vector search."""

import pytest
from pathlib import Path

from pmd.store.database import Database
from pmd.store.embeddings import EmbeddingRepository, _serialize_embedding
from pmd.core.types import SearchResult, SearchSource

# Default embedding dimension (matches schema.py)
EMBEDDING_DIM = 768


def make_embedding(value: float = 0.1, dim: int = EMBEDDING_DIM) -> list[float]:
    """Create a test embedding with the required dimensions."""
    return [value] * dim


class TestSerializeEmbedding:
    """Tests for embedding serialization."""

    def test_serialize_empty_embedding(self):
        """Should serialize empty embedding."""
        result = _serialize_embedding([])
        assert result == b""

    def test_serialize_single_value(self):
        """Should serialize single float."""
        result = _serialize_embedding([1.0])
        assert len(result) == 4  # One float = 4 bytes

    def test_serialize_multiple_values(self):
        """Should serialize multiple floats."""
        result = _serialize_embedding([1.0, 2.0, 3.0])
        assert len(result) == 12  # Three floats = 12 bytes

    def test_serialize_preserves_values(self):
        """Serialization should be reversible."""
        import struct

        original = [1.5, -2.5, 3.14159]
        serialized = _serialize_embedding(original)
        deserialized = list(struct.unpack(f"{len(original)}f", serialized))

        for orig, deser in zip(original, deserialized):
            assert abs(orig - deser) < 1e-6


class TestEmbeddingStore:
    """Tests for storing embeddings."""

    def test_store_embedding_creates_metadata(self, embedding_repo: EmbeddingRepository, db: Database):
        """store_embedding should create metadata record."""
        embedding = make_embedding(0.1)
        embedding_repo.store_embedding("test_hash", 0, 0, embedding, "test-model")

        cursor = db.execute(
            "SELECT * FROM content_vectors WHERE hash = ?",
            ("test_hash",),
        )
        row = cursor.fetchone()

        assert row is not None
        assert row["hash"] == "test_hash"
        assert row["seq"] == 0
        assert row["pos"] == 0
        assert row["model"] == "test-model"

    def test_store_multiple_chunks(self, embedding_repo: EmbeddingRepository, db: Database):
        """store_embedding should handle multiple chunks."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model")
        embedding_repo.store_embedding("hash1", 1, 100, make_embedding(0.2), "model")
        embedding_repo.store_embedding("hash1", 2, 200, make_embedding(0.3), "model")

        cursor = db.execute(
            "SELECT COUNT(*) as count FROM content_vectors WHERE hash = ?",
            ("hash1",),
        )
        assert cursor.fetchone()["count"] == 3

    def test_store_embedding_updates_on_conflict(self, embedding_repo: EmbeddingRepository, db: Database):
        """store_embedding should update existing record."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model-v1")
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.2), "model-v2")

        cursor = db.execute(
            "SELECT model FROM content_vectors WHERE hash = ? AND seq = ?",
            ("hash1", 0),
        )
        assert cursor.fetchone()["model"] == "model-v2"


class TestEmbeddingGet:
    """Tests for retrieving embedding metadata."""

    def test_get_embeddings_for_content(self, embedding_repo: EmbeddingRepository):
        """get_embeddings_for_content should return all chunks."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model")
        embedding_repo.store_embedding("hash1", 1, 100, make_embedding(0.2), "model")

        results = embedding_repo.get_embeddings_for_content("hash1")

        assert len(results) == 2
        assert results[0] == (0, 0, "model")
        assert results[1] == (1, 100, "model")

    def test_get_embeddings_ordered_by_seq(self, embedding_repo: EmbeddingRepository):
        """get_embeddings_for_content should return ordered by seq."""
        embedding_repo.store_embedding("hash1", 2, 200, make_embedding(0.3), "model")
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model")
        embedding_repo.store_embedding("hash1", 1, 100, make_embedding(0.2), "model")

        results = embedding_repo.get_embeddings_for_content("hash1")
        seqs = [r[0] for r in results]

        assert seqs == [0, 1, 2]

    def test_get_embeddings_nonexistent_hash(self, embedding_repo: EmbeddingRepository):
        """get_embeddings_for_content should return empty for nonexistent."""
        results = embedding_repo.get_embeddings_for_content("nonexistent")

        assert results == []


class TestEmbeddingHasEmbeddings:
    """Tests for checking if embeddings exist."""

    def test_has_embeddings_true(self, embedding_repo: EmbeddingRepository):
        """has_embeddings should return True when embeddings exist."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model")

        assert embedding_repo.has_embeddings("hash1") is True

    def test_has_embeddings_false(self, embedding_repo: EmbeddingRepository):
        """has_embeddings should return False when no embeddings."""
        assert embedding_repo.has_embeddings("nonexistent") is False

    def test_has_embeddings_with_model_filter(self, embedding_repo: EmbeddingRepository):
        """has_embeddings should filter by model."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model-a")

        assert embedding_repo.has_embeddings("hash1", "model-a") is True
        assert embedding_repo.has_embeddings("hash1", "model-b") is False


class TestEmbeddingDelete:
    """Tests for deleting embeddings."""

    def test_delete_embeddings(self, embedding_repo: EmbeddingRepository):
        """delete_embeddings should remove all chunks for hash."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model")
        embedding_repo.store_embedding("hash1", 1, 100, make_embedding(0.2), "model")

        count = embedding_repo.delete_embeddings("hash1")

        assert count == 2
        assert embedding_repo.has_embeddings("hash1") is False

    def test_delete_embeddings_nonexistent(self, embedding_repo: EmbeddingRepository):
        """delete_embeddings should return 0 for nonexistent."""
        count = embedding_repo.delete_embeddings("nonexistent")

        assert count == 0

    def test_delete_embeddings_only_target(self, embedding_repo: EmbeddingRepository):
        """delete_embeddings should only delete target hash."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model")
        embedding_repo.store_embedding("hash2", 0, 0, make_embedding(0.2), "model")

        embedding_repo.delete_embeddings("hash1")

        assert embedding_repo.has_embeddings("hash1") is False
        assert embedding_repo.has_embeddings("hash2") is True


class TestEmbeddingClearByModel:
    """Tests for clearing embeddings by model."""

    def test_clear_by_model(self, embedding_repo: EmbeddingRepository):
        """clear_embeddings_by_model should remove all for model."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model-a")
        embedding_repo.store_embedding("hash2", 0, 0, make_embedding(0.2), "model-a")
        embedding_repo.store_embedding("hash3", 0, 0, make_embedding(0.3), "model-b")

        count = embedding_repo.clear_embeddings_by_model("model-a")

        assert count == 2
        assert embedding_repo.has_embeddings("hash1") is False
        assert embedding_repo.has_embeddings("hash2") is False
        assert embedding_repo.has_embeddings("hash3") is True

    def test_clear_by_model_nonexistent(self, embedding_repo: EmbeddingRepository):
        """clear_embeddings_by_model should return 0 for nonexistent model."""
        count = embedding_repo.clear_embeddings_by_model("nonexistent-model")

        assert count == 0


class TestEmbeddingCount:
    """Tests for counting embeddings."""

    def test_count_embeddings_empty(self, embedding_repo: EmbeddingRepository):
        """count_embeddings should return 0 when empty."""
        count = embedding_repo.count_embeddings()

        assert count == 0

    def test_count_embeddings_all(self, embedding_repo: EmbeddingRepository):
        """count_embeddings should count all embeddings."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model-a")
        embedding_repo.store_embedding("hash1", 1, 100, make_embedding(0.2), "model-a")
        embedding_repo.store_embedding("hash2", 0, 0, make_embedding(0.3), "model-b")

        count = embedding_repo.count_embeddings()

        assert count == 3

    def test_count_embeddings_by_model(self, embedding_repo: EmbeddingRepository):
        """count_embeddings should filter by model."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model-a")
        embedding_repo.store_embedding("hash2", 0, 0, make_embedding(0.2), "model-a")
        embedding_repo.store_embedding("hash3", 0, 0, make_embedding(0.3), "model-b")

        assert embedding_repo.count_embeddings("model-a") == 2
        assert embedding_repo.count_embeddings("model-b") == 1


class TestEmbeddingVectorSearch:
    """Tests for vector similarity search."""

    def test_search_vectors_without_vec_extension(self, embedding_repo: EmbeddingRepository, db: Database):
        """search_vectors should return empty if vec not available."""
        if db.vec_available:
            pytest.skip("sqlite-vec is available, skipping no-vec test")

        query_embedding = [0.1, 0.2, 0.3]
        results = embedding_repo.search_vectors(query_embedding, limit=5)

        assert results == []

    def test_search_vectors_empty_query(self, embedding_repo: EmbeddingRepository):
        """search_vectors should return empty for empty query embedding."""
        results = embedding_repo.search_vectors([], limit=5)

        assert results == []

    @pytest.mark.skipif(
        True,  # Skip by default since sqlite-vec may not be installed
        reason="sqlite-vec extension required for vector search tests"
    )
    def test_search_vectors_returns_results(
        self,
        embedding_repo: EmbeddingRepository,
        document_repo,
        sample_collection,
        db: Database,
    ):
        """search_vectors should return matching documents."""
        if not db.vec_available:
            pytest.skip("sqlite-vec not available")

        # Create document
        content = "Test document for vector search"
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            content,
        )

        # Store embedding (384 dimensions for e5-small)
        embedding = [0.1] * 384
        embedding_repo.store_embedding(doc.hash, 0, 0, embedding, "nomic-embed-text")

        # Search with similar embedding
        query = [0.1] * 384
        results = embedding_repo.search_vectors(query, limit=5)

        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].source == SearchSource.VECTOR


class TestEmbeddingIntegration:
    """Integration tests for embedding workflow."""

    def test_full_embedding_workflow(self, embedding_repo: EmbeddingRepository):
        """Test complete embedding lifecycle."""
        # Store
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model")
        embedding_repo.store_embedding("hash1", 1, 50, make_embedding(0.2), "model")

        # Verify stored
        assert embedding_repo.has_embeddings("hash1") is True
        assert embedding_repo.count_embeddings() == 2

        # Get metadata
        chunks = embedding_repo.get_embeddings_for_content("hash1")
        assert len(chunks) == 2

        # Delete
        deleted = embedding_repo.delete_embeddings("hash1")
        assert deleted == 2
        assert embedding_repo.has_embeddings("hash1") is False

    def test_multiple_models_same_content(self, embedding_repo: EmbeddingRepository):
        """Same content can have embeddings from different models."""
        embedding_repo.store_embedding("hash1", 0, 0, make_embedding(0.1), "model-a")
        embedding_repo.store_embedding("hash1", 1, 0, make_embedding(0.2), "model-b")

        assert embedding_repo.has_embeddings("hash1", "model-a") is True
        assert embedding_repo.has_embeddings("hash1", "model-b") is True
        assert embedding_repo.count_embeddings() == 2
