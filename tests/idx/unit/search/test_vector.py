"""Tests for idx.search.vector module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idx.search.vector import create_vector_node, VectorIndexer, VectorSearch
from idx.search.models import SearchResult
from idx.transform.chunker import Chunk


class TestCreateVectorNode:
    """Tests for create_vector_node function."""

    def test_creates_node_with_stable_id(self) -> None:
        """Node ID is stable based on hash and seq."""
        chunk = Chunk(seq=0, pos=0, text="Test content", size=12)
        node = create_vector_node(
            chunk=chunk,
            content_hash="abc123def456",
            path="notes/test.md",
            dataset_name="my-vault",
        )

        assert node.id_ == "abc123def456:0"

    def test_creates_node_with_metadata(self) -> None:
        """Node contains required metadata."""
        chunk = Chunk(seq=1, pos=100, text="Test content", size=12)
        node = create_vector_node(
            chunk=chunk,
            content_hash="abc123",
            path="notes/test.md",
            dataset_name="my-vault",
        )

        assert node.metadata["hash"] == "abc123"
        assert node.metadata["seq"] == 1
        assert node.metadata["pos"] == 100
        assert node.metadata["path"] == "notes/test.md"
        assert node.metadata["dataset_name"] == "my-vault"
        assert node.metadata["chunk_size"] == 12

    def test_includes_additional_metadata(self) -> None:
        """Additional metadata is included."""
        chunk = Chunk(seq=0, pos=0, text="Test", size=4)
        node = create_vector_node(
            chunk=chunk,
            content_hash="abc",
            path="test.md",
            dataset_name="vault",
            additional_metadata={"tags": ["important", "work"]},
        )

        assert node.metadata["tags"] == ["important", "work"]

    def test_node_contains_chunk_text(self) -> None:
        """Node text matches chunk text."""
        chunk = Chunk(seq=0, pos=0, text="Hello, World!", size=13)
        node = create_vector_node(
            chunk=chunk,
            content_hash="abc",
            path="test.md",
            dataset_name="vault",
        )

        assert node.text == "Hello, World!"


class TestVectorIndexer:
    """Tests for VectorIndexer class."""

    @pytest.fixture
    def mock_settings(self, tmp_path: Path):
        """Mock settings for testing."""
        mock = MagicMock()
        mock.vector_store_path = tmp_path / "vectors"
        mock.embedding_model = "BAAI/bge-small-en-v1.5"
        mock.performance.chunk_max_bytes = 1000
        mock.performance.chunk_min_bytes = 100
        return mock

    def test_create_nodes_for_document(self, mock_settings: MagicMock) -> None:
        """Creates correct number of nodes for document."""
        with patch("idx.search.vector.get_settings", return_value=mock_settings):
            with patch("idx.search.vector.HuggingFaceEmbedding"):
                with patch("idx.search.vector.LlamaSettings"):
                    indexer = VectorIndexer(persist_dir=mock_settings.vector_store_path)

                    # Short document should create one node
                    nodes = indexer.create_nodes_for_document(
                        body="Short document content.",
                        content_hash="hash123",
                        path="test.md",
                        dataset_name="vault",
                    )

                    assert len(nodes) >= 1
                    assert nodes[0].metadata["path"] == "test.md"
                    assert nodes[0].metadata["dataset_name"] == "vault"

    def test_creates_nodes_with_stable_ids(self, mock_settings: MagicMock) -> None:
        """Node IDs are stable across calls."""
        with patch("idx.search.vector.get_settings", return_value=mock_settings):
            with patch("idx.search.vector.HuggingFaceEmbedding"):
                with patch("idx.search.vector.LlamaSettings"):
                    indexer = VectorIndexer(persist_dir=mock_settings.vector_store_path)

                    nodes1 = indexer.create_nodes_for_document(
                        body="Test content",
                        content_hash="hash123",
                        path="test.md",
                        dataset_name="vault",
                    )

                    nodes2 = indexer.create_nodes_for_document(
                        body="Test content",
                        content_hash="hash123",
                        path="test.md",
                        dataset_name="vault",
                    )

                    assert [n.id_ for n in nodes1] == [n.id_ for n in nodes2]


class TestVectorSearchDedupe:
    """Tests for VectorSearch deduplication logic."""

    @pytest.fixture
    def vector_search(self, tmp_path: Path):
        """Create VectorSearch with mock index."""
        with patch("idx.search.vector.get_settings") as mock_settings:
            mock_settings.return_value.vector_store_path = tmp_path / "vectors"
            mock_settings.return_value.embedding_model = "test-model"
            with patch("idx.search.vector.HuggingFaceEmbedding"):
                with patch("idx.search.vector.LlamaSettings"):
                    with patch.object(VectorSearch, "_load_index"):
                        search = VectorSearch(persist_dir=tmp_path / "vectors")
                        search._index = None  # No actual index
                        return search

    def test_dedupe_by_path_keeps_best_score(self, vector_search: VectorSearch) -> None:
        """Deduplication keeps the highest-scoring chunk per document."""
        results = [
            SearchResult(
                path="doc1.md",
                dataset_name="vault",
                score=0.9,
                chunk_seq=0,
            ),
            SearchResult(
                path="doc1.md",
                dataset_name="vault",
                score=0.7,
                chunk_seq=1,
            ),
            SearchResult(
                path="doc2.md",
                dataset_name="vault",
                score=0.8,
                chunk_seq=0,
            ),
        ]

        deduped = vector_search._dedupe_by_path(results)

        assert len(deduped) == 2
        doc1_result = next(r for r in deduped if r.path == "doc1.md")
        assert doc1_result.score == 0.9  # Best score kept

    def test_dedupe_considers_dataset_name(self, vector_search: VectorSearch) -> None:
        """Same path in different datasets are kept separate."""
        results = [
            SearchResult(
                path="readme.md",
                dataset_name="vault1",
                score=0.9,
            ),
            SearchResult(
                path="readme.md",
                dataset_name="vault2",
                score=0.8,
            ),
        ]

        deduped = vector_search._dedupe_by_path(results)

        assert len(deduped) == 2  # Different datasets, both kept

    def test_normalize_scores(self, vector_search: VectorSearch) -> None:
        """Scores are normalized to 0-1 range."""
        results = [
            SearchResult(path="a.md", dataset_name="vault", score=10.0),
            SearchResult(path="b.md", dataset_name="vault", score=5.0),
            SearchResult(path="c.md", dataset_name="vault", score=2.5),
        ]

        normalized = vector_search._normalize_scores(results)

        assert normalized[0].score == 1.0  # Max score -> 1.0
        assert normalized[1].score == 0.5  # Half of max -> 0.5
        assert normalized[2].score == 0.25  # Quarter of max -> 0.25

    def test_normalize_empty_results(self, vector_search: VectorSearch) -> None:
        """Empty results don't cause errors."""
        normalized = vector_search._normalize_scores([])
        assert normalized == []

    def test_search_returns_empty_when_no_index(self, vector_search: VectorSearch) -> None:
        """Search returns empty results when no index exists."""
        results = vector_search.search("test query")

        assert results.results == []
        assert results.query == "test query"
        assert results.mode == "vector"
        assert results.total_candidates == 0
