"""idx.search.vector - Vector similarity search implementation.

Provides vector search using LlamaIndex VectorIndexRetriever with lazy index loading.
Supports optional dataset filtering via metadata filters.

Example usage:
    from idx.search.vector import VectorSearch
    from idx.store.vector import VectorStoreManager

    vector_search = VectorSearch()
    results = vector_search.search("machine learning concepts", top_k=10)

    # With dataset filtering
    results = vector_search.search(
        "project notes",
        top_k=5,
        dataset_name="obsidian"
    )
"""

from typing import TYPE_CHECKING

from idx.core.logging import get_logger
from idx.search.models import SearchResult
from idx.store.vector import VectorStoreManager

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex

__all__ = ["VectorSearch"]

logger = get_logger(__name__)


class VectorSearch:
    """Vector similarity search using LlamaIndex VectorIndexRetriever.

    Thin wrapper around LlamaIndex's vector retrieval with lazy index loading.
    Converts LlamaIndex NodeWithScore results to internal SearchResult format.

    The index is loaded lazily on first search via VectorStoreManager.
    Supports filtering by dataset_name through source_doc_id prefix matching.

    Attributes:
        _vector_manager: VectorStoreManager instance for index access.
        _index: Cached VectorStoreIndex, loaded lazily.

    Example:
        vector_search = VectorSearch()

        # Basic search
        results = vector_search.search("python tutorials")

        # Search within a specific dataset
        results = vector_search.search(
            "meeting notes",
            top_k=20,
            dataset_name="obsidian"
        )
    """

    def __init__(
        self,
        vector_manager: VectorStoreManager | None = None,
    ) -> None:
        """Initialize the VectorSearch.

        Args:
            vector_manager: VectorStoreManager instance for index access.
                If None, creates a new VectorStoreManager with default settings.
        """
        self._vector_manager = vector_manager or VectorStoreManager()
        self._index: "VectorStoreIndex | None" = None

    def _ensure_index(self) -> "VectorStoreIndex":
        """Ensure the vector index is loaded.

        Lazy-loads the index on first access via VectorStoreManager.

        Returns:
            VectorStoreIndex ready for retrieval.
        """
        if self._index is None:
            logger.debug("Lazy-loading vector index")
            self._index = self._vector_manager.load_or_create()
            logger.info("Vector index loaded for search")
        return self._index

    def search(
        self,
        query: str,
        top_k: int = 10,
        dataset_name: str | None = None,
    ) -> list[SearchResult]:
        """Search vector store for similar documents.

        Performs semantic similarity search using the embedded query.
        Results include the similarity score in scores["vector"].

        Args:
            query: The search query string.
            top_k: Maximum number of results to return. Defaults to 10.
            dataset_name: Optional dataset name to filter results.
                When provided, only returns results from that dataset.
                Filters by matching source_doc_id prefix.

        Returns:
            List of SearchResult objects ordered by similarity (highest first).
            Each result includes:
            - path: Document path within the dataset
            - dataset_name: Source dataset name
            - score: Vector similarity score
            - chunk_text: The matched chunk text
            - chunk_seq: Chunk sequence number (if available)
            - chunk_pos: Byte position in document (if available)
            - metadata: Document metadata
            - scores: Dict with "vector" key containing similarity score
        """
        from llama_index.core.schema import QueryBundle
        from llama_index.core.vector_stores import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )

        index = self._ensure_index()

        # Build metadata filters if dataset_name is specified
        filters = None
        if dataset_name:
            # Filter by source_doc_id prefix (format: {dataset_name}:{path})
            # Using CONTAINS operator since LlamaIndex doesn't have STARTS_WITH
            # Note: This works because source_doc_id starts with dataset_name:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="source_doc_id",
                        operator=FilterOperator.CONTAINS,
                        value=f"{dataset_name}:",
                    )
                ]
            )
            logger.debug(f"Filtering vector search by dataset: {dataset_name}")

        # Create retriever with appropriate settings
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=filters,
        )

        # Execute search
        query_bundle = QueryBundle(query_str=query)
        nodes_with_scores = retriever.retrieve(query_bundle)

        logger.debug(
            f"Vector search '{query[:50]}...' returned {len(nodes_with_scores)} results"
        )

        # Convert to SearchResult objects
        results = []
        for node_with_score in nodes_with_scores:
            node = node_with_score.node
            score = node_with_score.score or 0.0

            # Extract source_doc_id and parse dataset_name and path
            source_doc_id = node.metadata.get("source_doc_id", "")
            if ":" in source_doc_id:
                ds_name, path = source_doc_id.split(":", 1)
            else:
                # Fallback: try to get from metadata or use defaults
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

            results.append(
                SearchResult(
                    path=path,
                    dataset_name=ds_name,
                    score=score,
                    chunk_text=node.text,
                    chunk_seq=chunk_seq,
                    chunk_pos=chunk_pos,
                    metadata=metadata,
                    scores={"vector": score},
                )
            )

        return results
