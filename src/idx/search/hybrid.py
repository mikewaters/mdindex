"""idx.search.hybrid - Hybrid search combining FTS5 and vector retrieval.

Provides hybrid search using LlamaIndex QueryFusionRetriever with Reciprocal
Rank Fusion (RRF) to combine lexical (FTS5) and dense (vector) retrieval.

Example usage:
    from idx.search.hybrid import HybridSearch
    from idx.store.database import get_session
    from idx.store.session_context import use_session

    with get_session() as session:
        with use_session(session):
            hybrid = HybridSearch()
            results = hybrid.search("machine learning concepts", top_k=10)

            # With dataset filtering
            results = hybrid.search(
                "project notes",
                top_k=5,
                dataset_name="obsidian"
            )
"""

from typing import TYPE_CHECKING

from idx.core.logging import get_logger
from idx.search.fts_chunk import FTSChunkRetriever
from idx.search.models import SearchResult
from idx.store.vector import VectorStoreManager

if TYPE_CHECKING:
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.schema import NodeWithScore

__all__ = ["HybridSearch"]

logger = get_logger(__name__)


class HybridSearch:
    """Hybrid search combining FTS5 and vector retrieval using RRF fusion.

    Uses LlamaIndex's QueryFusionRetriever with reciprocal_rerank mode to
    combine results from FTSChunkRetriever (lexical) and VectorIndexRetriever
    (dense semantic). The fusion applies Reciprocal Rank Fusion (RRF) scoring.

    The search is performed at chunk level, returning chunk-level results
    with RRF scores. Both retrievers must use matching node IDs for proper
    deduplication during fusion.

    Attributes:
        _vector_manager: VectorStoreManager for lazy vector index loading.
        _fts_retriever: FTSChunkRetriever instance for lexical search.
        _fusion_retriever: Cached QueryFusionRetriever instance.

    Example:
        hybrid = HybridSearch()

        # Basic hybrid search
        results = hybrid.search("python async patterns")

        # Search with custom parameters
        results = hybrid.search(
            "meeting notes",
            top_k=20,
            k_lex=30,
            k_dense=30,
            dataset_name="obsidian"
        )
    """

    def __init__(
        self,
        vector_manager: VectorStoreManager | None = None,
        fts_retriever: FTSChunkRetriever | None = None,
    ) -> None:
        """Initialize the HybridSearch.

        Args:
            vector_manager: VectorStoreManager for index access. If None,
                creates a new VectorStoreManager with default settings.
            fts_retriever: FTSChunkRetriever for FTS5 searches. If None,
                creates a new FTSChunkRetriever with default settings.
        """
        self._vector_manager = vector_manager or VectorStoreManager()
        self._fts_retriever = fts_retriever or FTSChunkRetriever()
        self._fusion_retriever: "QueryFusionRetriever | None" = None

    def _create_fusion_retriever(
        self,
        k_lex: int,
        k_dense: int,
        k_fused: int,
        dataset_name: str | None = None,
    ) -> "QueryFusionRetriever":
        """Create a QueryFusionRetriever with the specified parameters.

        Creates FTS and vector retrievers with appropriate settings and
        combines them using QueryFusionRetriever with RRF mode.

        Args:
            k_lex: Number of results to retrieve from FTS5.
            k_dense: Number of results to retrieve from vector search.
            k_fused: Number of fused results to return.
            dataset_name: Optional dataset name filter.

        Returns:
            Configured QueryFusionRetriever instance.
        """
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.vector_stores import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )

        # Configure FTS retriever
        self._fts_retriever.similarity_top_k = k_lex
        self._fts_retriever.dataset_name = dataset_name

        # Load vector index and create retriever with filters
        index = self._vector_manager.load_or_create()

        # Build metadata filters if dataset_name is specified
        filters = None
        if dataset_name:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="source_doc_id",
                        operator=FilterOperator.CONTAINS,
                        value=f"{dataset_name}:",
                    )
                ]
            )
            logger.debug(f"Filtering hybrid search by dataset: {dataset_name}")

        vector_retriever = index.as_retriever(
            similarity_top_k=k_dense,
            filters=filters,
        )

        # Create fusion retriever with RRF mode
        fusion_retriever = QueryFusionRetriever(
            retrievers=[self._fts_retriever, vector_retriever],
            similarity_top_k=k_fused,
            num_queries=1,  # No query expansion
            mode="reciprocal_rerank",  # RRF fusion
            use_async=False,
        )

        logger.debug(
            f"Created QueryFusionRetriever: k_lex={k_lex}, k_dense={k_dense}, "
            f"k_fused={k_fused}, dataset={dataset_name}"
        )

        return fusion_retriever

    def _convert_node_to_result(
        self,
        node_with_score: "NodeWithScore",
        dataset_name: str | None = None,
    ) -> SearchResult:
        """Convert a LlamaIndex NodeWithScore to SearchResult.

        Extracts metadata from the node and constructs a SearchResult
        with the RRF score stored in scores["rrf"].

        Args:
            node_with_score: LlamaIndex NodeWithScore from fusion retriever.
            dataset_name: Fallback dataset name if not in metadata.

        Returns:
            SearchResult with RRF score and chunk metadata.
        """
        node = node_with_score.node
        score = node_with_score.score or 0.0

        # Extract source_doc_id and parse dataset_name and path
        source_doc_id = node.metadata.get("source_doc_id", "")
        if ":" in source_doc_id:
            ds_name, path = source_doc_id.split(":", 1)
        else:
            # Fallback: use provided dataset_name or empty
            ds_name = dataset_name or ""
            path = node.metadata.get("relative_path", "")

        # Extract chunk metadata
        chunk_seq = node.metadata.get("chunk_seq")
        chunk_pos = node.metadata.get("chunk_pos")

        # Build metadata dict (exclude internal keys)
        metadata = {
            k: v
            for k, v in node.metadata.items()
            if k not in ("source_doc_id", "chunk_seq", "chunk_pos", "doc_id")
        }

        return SearchResult(
            path=path,
            dataset_name=ds_name,
            score=score,
            chunk_text=node.text,
            chunk_seq=chunk_seq,
            chunk_pos=chunk_pos,
            metadata=metadata,
            scores={"rrf": score},
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        k_lex: int | None = None,
        k_dense: int | None = None,
        dataset_name: str | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid search combining FTS5 and vector retrieval.

        Uses QueryFusionRetriever with RRF to fuse results from both
        retrievers. The RRF formula assigns scores based on rank position:
        score = sum(1 / (k + rank)) across retrievers.

        Args:
            query: The search query string.
            top_k: Maximum number of fused results to return. Defaults to 10.
            k_lex: Number of candidates from FTS5 retriever. If None, defaults
                to 2 * top_k to ensure sufficient candidates for fusion.
            k_dense: Number of candidates from vector retriever. If None,
                defaults to 2 * top_k to ensure sufficient candidates.
            dataset_name: Optional dataset name filter. When provided, both
                retrievers filter results to that dataset.

        Returns:
            List of SearchResult objects ordered by RRF score (highest first).
            Each result includes:
            - path: Document path within the dataset
            - dataset_name: Source dataset name
            - score: RRF fusion score
            - chunk_text: The matched chunk text
            - chunk_seq: Chunk sequence number (if available)
            - chunk_pos: Byte position in document (if available)
            - metadata: Document metadata
            - scores: Dict with "rrf" key containing the fusion score
        """
        from llama_index.core.schema import QueryBundle

        # Default k values to 2x top_k for sufficient fusion candidates
        k_lex_actual = k_lex if k_lex is not None else top_k * 2
        k_dense_actual = k_dense if k_dense is not None else top_k * 2

        # Create fusion retriever with current parameters
        fusion_retriever = self._create_fusion_retriever(
            k_lex=k_lex_actual,
            k_dense=k_dense_actual,
            k_fused=top_k,
            dataset_name=dataset_name,
        )

        # Execute hybrid search
        query_bundle = QueryBundle(query_str=query)
        nodes_with_scores = fusion_retriever.retrieve(query_bundle)

        logger.debug(
            f"Hybrid search '{query[:50]}...' returned {len(nodes_with_scores)} results"
        )

        # Convert to SearchResult objects
        results = [
            self._convert_node_to_result(nws, dataset_name)
            for nws in nodes_with_scores
        ]

        return results
