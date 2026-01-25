"""Tests for idx.search.hybrid module.

Tests for HybridSearch class using LlamaIndex QueryFusionRetriever.
"""

from unittest.mock import MagicMock, patch

import pytest

from idx.search.hybrid import HybridSearch
from idx.search.models import SearchResult


class TestHybridSearch:
    """Tests for HybridSearch class."""

    @pytest.fixture
    def mock_vector_manager(self) -> MagicMock:
        """Create mock VectorStoreManager."""
        manager = MagicMock()
        mock_index = MagicMock()
        manager.load_or_create.return_value = mock_index
        return manager

    @pytest.fixture
    def mock_fts_retriever(self) -> MagicMock:
        """Create mock FTSChunkRetriever."""
        retriever = MagicMock()
        retriever.similarity_top_k = 20
        retriever.dataset_name = None
        return retriever

    @pytest.fixture
    def hybrid_search(
        self, mock_vector_manager: MagicMock, mock_fts_retriever: MagicMock
    ) -> HybridSearch:
        """Create HybridSearch with mocked dependencies."""
        return HybridSearch(
            vector_manager=mock_vector_manager,
            fts_retriever=mock_fts_retriever,
        )

    def test_init_default(self) -> None:
        """Test HybridSearch initializes with default dependencies."""
        with patch("idx.search.hybrid.VectorStoreManager") as mock_vm_cls, \
             patch("idx.search.hybrid.FTSChunkRetriever") as mock_fts_cls:
            mock_vm_cls.return_value = MagicMock()
            mock_fts_cls.return_value = MagicMock()

            HybridSearch()

            mock_vm_cls.assert_called_once()
            mock_fts_cls.assert_called_once()

    def test_init_custom_dependencies(
        self, mock_vector_manager: MagicMock, mock_fts_retriever: MagicMock
    ) -> None:
        """Test HybridSearch accepts custom dependencies."""
        search = HybridSearch(
            vector_manager=mock_vector_manager,
            fts_retriever=mock_fts_retriever,
        )

        assert search._vector_manager is mock_vector_manager
        assert search._fts_retriever is mock_fts_retriever

    def test_search_creates_fusion_retriever(
        self, hybrid_search: HybridSearch
    ) -> None:
        """Test search creates QueryFusionRetriever with correct parameters."""
        mock_node = MagicMock()
        mock_node.metadata = {"source_doc_id": "vault:doc1.md"}
        mock_node.text = "Sample text"

        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.5

        with patch("llama_index.core.retrievers.QueryFusionRetriever") as mock_fusion_cls:
            mock_fusion = MagicMock()
            mock_fusion.retrieve.return_value = [mock_nws]
            mock_fusion_cls.return_value = mock_fusion

            hybrid_search.search("test query", top_k=10)

            # Verify QueryFusionRetriever was created with correct settings
            mock_fusion_cls.assert_called_once()
            call_kwargs = mock_fusion_cls.call_args.kwargs
            assert call_kwargs["mode"] == "reciprocal_rerank"
            assert call_kwargs["num_queries"] == 1
            assert call_kwargs["use_async"] is False
            assert call_kwargs["similarity_top_k"] == 10

    def test_search_returns_search_results(
        self, hybrid_search: HybridSearch
    ) -> None:
        """Test search returns list of SearchResult objects."""
        mock_node = MagicMock()
        mock_node.metadata = {"source_doc_id": "vault:doc1.md"}
        mock_node.text = "Sample text"

        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.5

        with patch("llama_index.core.retrievers.QueryFusionRetriever") as mock_fusion_cls:
            mock_fusion = MagicMock()
            mock_fusion.retrieve.return_value = [mock_nws]
            mock_fusion_cls.return_value = mock_fusion

            results = hybrid_search.search("test query")

            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].path == "doc1.md"
            assert results[0].dataset_name == "vault"
            assert results[0].scores["rrf"] == 0.5

    def test_search_with_custom_k_values(
        self, hybrid_search: HybridSearch, mock_fts_retriever: MagicMock
    ) -> None:
        """Test search configures retrievers with custom k values."""
        mock_node = MagicMock()
        mock_node.metadata = {"source_doc_id": "vault:doc1.md"}
        mock_node.text = "Sample text"

        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.5

        with patch("llama_index.core.retrievers.QueryFusionRetriever") as mock_fusion_cls:
            mock_fusion = MagicMock()
            mock_fusion.retrieve.return_value = [mock_nws]
            mock_fusion_cls.return_value = mock_fusion

            hybrid_search.search(
                "test query",
                top_k=5,
                k_lex=30,
                k_dense=40,
            )

            # Verify FTS retriever was configured
            assert mock_fts_retriever.similarity_top_k == 30

    def test_search_with_default_k_values(
        self, hybrid_search: HybridSearch, mock_fts_retriever: MagicMock
    ) -> None:
        """Test search uses 2x top_k as default for k_lex and k_dense."""
        mock_node = MagicMock()
        mock_node.metadata = {"source_doc_id": "vault:doc1.md"}
        mock_node.text = "Sample text"

        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.5

        with patch("llama_index.core.retrievers.QueryFusionRetriever") as mock_fusion_cls:
            mock_fusion = MagicMock()
            mock_fusion.retrieve.return_value = [mock_nws]
            mock_fusion_cls.return_value = mock_fusion

            hybrid_search.search("test query", top_k=10)

            # Default k_lex should be 2 * top_k = 20
            assert mock_fts_retriever.similarity_top_k == 20

    def test_search_with_dataset_filter(
        self, hybrid_search: HybridSearch, mock_fts_retriever: MagicMock
    ) -> None:
        """Test search passes dataset filter to both retrievers."""
        mock_node = MagicMock()
        mock_node.metadata = {"source_doc_id": "my-vault:doc1.md"}
        mock_node.text = "Sample text"

        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.5

        with patch("llama_index.core.retrievers.QueryFusionRetriever") as mock_fusion_cls:
            mock_fusion = MagicMock()
            mock_fusion.retrieve.return_value = [mock_nws]
            mock_fusion_cls.return_value = mock_fusion

            hybrid_search.search("test query", dataset_name="my-vault")

            # Verify FTS retriever was configured with dataset
            assert mock_fts_retriever.dataset_name == "my-vault"

    def test_search_empty_results(self, hybrid_search: HybridSearch) -> None:
        """Test search handles empty results gracefully."""
        with patch("llama_index.core.retrievers.QueryFusionRetriever") as mock_fusion_cls:
            mock_fusion = MagicMock()
            mock_fusion.retrieve.return_value = []
            mock_fusion_cls.return_value = mock_fusion

            results = hybrid_search.search("test query")

            assert results == []

    def test_convert_node_extracts_chunk_metadata(
        self, hybrid_search: HybridSearch
    ) -> None:
        """Test node conversion extracts chunk_seq and chunk_pos."""
        mock_node = MagicMock()
        mock_node.metadata = {
            "source_doc_id": "vault:notes/doc.md",
            "chunk_seq": 2,
            "chunk_pos": 1024,
        }
        mock_node.text = "Chunk content"

        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.75

        result = hybrid_search._convert_node_to_result(mock_nws)

        assert result.chunk_seq == 2
        assert result.chunk_pos == 1024
        assert result.chunk_text == "Chunk content"

    def test_convert_node_fallback_dataset(
        self, hybrid_search: HybridSearch
    ) -> None:
        """Test node conversion uses fallback dataset_name."""
        mock_node = MagicMock()
        mock_node.metadata = {"relative_path": "doc.md"}
        mock_node.text = "Content"

        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.5

        result = hybrid_search._convert_node_to_result(
            mock_nws, dataset_name="fallback-vault"
        )

        assert result.dataset_name == "fallback-vault"
        assert result.path == "doc.md"

    def test_convert_node_excludes_internal_metadata(
        self, hybrid_search: HybridSearch
    ) -> None:
        """Test node conversion excludes internal metadata keys."""
        mock_node = MagicMock()
        mock_node.metadata = {
            "source_doc_id": "vault:doc.md",
            "chunk_seq": 1,
            "chunk_pos": 0,
            "doc_id": 42,
            "custom_field": "value",
        }
        mock_node.text = "Content"

        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.5

        result = hybrid_search._convert_node_to_result(mock_nws)

        # Internal keys should be excluded from metadata
        assert "source_doc_id" not in result.metadata
        assert "chunk_seq" not in result.metadata
        assert "chunk_pos" not in result.metadata
        assert "doc_id" not in result.metadata

        # Custom fields should be preserved
        assert result.metadata.get("custom_field") == "value"
