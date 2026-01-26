"""idx.search.vector - Vector similarity search implementation.

Provides vector search using direct SimpleVectorStore queries with text
lookup from SQLite. Supports optional dataset filtering via metadata.

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
    from llama_index.core.vector_stores import SimpleVectorStore

__all__ = ["VectorSearch"]

logger = get_logger(__name__)


class VectorSearch:
    """Vector similarity search using direct SimpleVectorStore queries.

    Queries the vector store directly and looks up chunk text from SQLite.
    This approach works regardless of LlamaIndex's docstore state.

    Attributes:
        _vector_manager: VectorStoreManager instance for vector store access.
        _vector_store: Cached SimpleVectorStore, loaded lazily.
        _embed_model: Embedding model for query vectorization.

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
            vector_manager: VectorStoreManager instance for vector store access.
                If None, creates a new VectorStoreManager with default settings.
        """
        self._vector_manager = vector_manager or VectorStoreManager()
        self._vector_store: "SimpleVectorStore | None" = None
        self._embed_model = None

    def _ensure_vector_store(self) -> "SimpleVectorStore":
        """Ensure the vector store is loaded.

        Returns:
            SimpleVectorStore ready for queries.
        """
        if self._vector_store is None:
            logger.debug("Lazy-loading vector store")
            self._vector_store = self._vector_manager.get_vector_store()
            logger.info("Vector store loaded for search")
        return self._vector_store

    def _ensure_embed_model(self):
        """Ensure the embedding model is loaded."""
        if self._embed_model is None:
            self._embed_model = self._vector_manager._get_embed_model()
        return self._embed_model

    def _lookup_chunk_text(self, chunk_ids: list[str]) -> dict[str, str]:
        """Look up chunk text from SQLite FTS table.

        Args:
            chunk_ids: List of chunk IDs to look up.

        Returns:
            Dict mapping chunk_id to text content.
        """
        if not chunk_ids:
            return {}

        from idx.store.database import get_engine
        from sqlalchemy import text

        engine = get_engine()
        # c0 = chunk_id, c1 = text
        # Use named parameters for SQLAlchemy
        placeholders = ",".join([f":id{i}" for i in range(len(chunk_ids))])
        params = {f"id{i}": cid for i, cid in enumerate(chunk_ids)}
        query = text(f"SELECT c0, c1 FROM chunks_fts_content WHERE c0 IN ({placeholders})")

        with engine.connect() as conn:
            result = conn.execute(query, params)
            return {row[0]: row[1] for row in result}

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
        from llama_index.core.vector_stores import VectorStoreQuery

        vector_store = self._ensure_vector_store()
        embed_model = self._ensure_embed_model()

        # Generate query embedding
        query_embedding = embed_model.get_query_embedding(query)

        # Query vector store directly
        # Request more results than needed if filtering, to account for filtered-out items
        fetch_k = top_k * 3 if dataset_name else top_k
        vs_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=fetch_k,
        )
        result = vector_store.query(vs_query)

        if not result.ids:
            logger.debug(f"Vector search '{query[:50]}...' returned 0 results")
            return []

        # Look up chunk text from SQLite
        chunk_texts = self._lookup_chunk_text(result.ids)

        # Build results from vector store metadata and SQLite text
        results = []
        for i, node_id in enumerate(result.ids):
            score = result.similarities[i] if result.similarities else 0.0
            metadata = vector_store.data.metadata_dict.get(node_id, {})

            # Extract source_doc_id and parse dataset_name and path
            source_doc_id = metadata.get("source_doc_id", "")
            if ":" in source_doc_id:
                ds_name, path = source_doc_id.split(":", 1)
            else:
                ds_name = ""
                path = metadata.get("relative_path", "")

            # Apply dataset filter if specified
            if dataset_name and ds_name != dataset_name:
                continue

            # Get chunk text from SQLite lookup
            chunk_text = chunk_texts.get(node_id, "")

            # Extract chunk metadata
            chunk_seq = metadata.get("chunk_seq")
            chunk_pos = metadata.get("chunk_pos")

            # Build metadata dict (exclude internal keys)
            result_metadata = {
                k: v
                for k, v in metadata.items()
                if k not in ("source_doc_id", "chunk_seq", "chunk_pos", "doc_id",
                           "_node_type", "document_id", "ref_doc_id")
            }

            results.append(
                SearchResult(
                    path=path,
                    dataset_name=ds_name,
                    score=score,
                    chunk_text=chunk_text,
                    chunk_seq=chunk_seq,
                    chunk_pos=chunk_pos,
                    metadata=result_metadata,
                    scores={"vector": score},
                )
            )

            # Stop if we have enough results
            if len(results) >= top_k:
                break

        logger.debug(
            f"Vector search '{query[:50]}...' returned {len(results)} results"
        )

        return results
