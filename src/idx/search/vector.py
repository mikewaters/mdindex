"""Vector search implementation using LlamaIndex.

Provides VectorSearch class for semantic similarity search using embeddings.
Uses LlamaIndex's VectorStoreIndex with SimpleVectorStore for persistence.

Example usage:
    from idx.search.vector import VectorSearch

    search = VectorSearch()
    results = search.search("how to implement authentication", limit=10)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_index.core import (
    Settings as LlamaSettings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from idx.core.logging import get_logger
from idx.core.settings import get_settings
from idx.search.models import SearchResult, SearchResults
from idx.transform.chunker import Chunk, LineChunker

__all__ = [
    "VectorSearch",
    "VectorIndexer",
    "create_vector_node",
]

logger = get_logger(__name__)


def create_vector_node(
    chunk: Chunk,
    content_hash: str,
    path: str,
    dataset_name: str,
    *,
    additional_metadata: dict[str, Any] | None = None,
) -> TextNode:
    """Create a TextNode for vector indexing with stable ID.

    Args:
        chunk: Chunk object with seq, pos, text, and size.
        content_hash: SHA256 hash of the document content.
        path: Document path within the dataset.
        dataset_name: Name of the source dataset.
        additional_metadata: Optional additional metadata to include.

    Returns:
        TextNode with stable ID and metadata for vector indexing.
    """
    # Stable node ID: hash:seq
    node_id = f"{content_hash}:{chunk.seq}"

    # Build metadata
    metadata = {
        "hash": content_hash,
        "seq": chunk.seq,
        "pos": chunk.pos,
        "path": path,
        "dataset_name": dataset_name,
        "chunk_size": chunk.size,
    }

    if additional_metadata:
        metadata.update(additional_metadata)

    return TextNode(
        id_=node_id,
        text=chunk.text,
        metadata=metadata,
    )


class VectorIndexer:
    """Manages vector index persistence and updates.

    Handles creating, loading, and updating the vector index using
    LlamaIndex's VectorStoreIndex with SimpleVectorStore backend.

    Example:
        indexer = VectorIndexer()

        # Add nodes for a document
        nodes = indexer.create_nodes_for_document(
            body="Document content...",
            content_hash="abc123",
            path="notes/test.md",
            dataset_name="my-vault",
        )
        indexer.add_nodes(nodes)
        indexer.persist()
    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize the vector indexer.

        Args:
            persist_dir: Directory for vector store persistence.
                Defaults to settings.vector_store_path.
            embedding_model: Name/path of the embedding model.
                Defaults to settings.embedding_model.
        """
        settings = get_settings()
        self._persist_dir = persist_dir or settings.vector_store_path
        self._embedding_model = embedding_model or settings.embedding_model
        self._chunker = LineChunker(
            max_bytes=settings.performance.chunk_max_bytes,
            min_chunk_size=settings.performance.chunk_min_bytes,
        )

        # Initialize embedding model
        self._embed_model = HuggingFaceEmbedding(
            model_name=self._embedding_model,
        )
        LlamaSettings.embed_model = self._embed_model

        # Load or create index
        self._index: VectorStoreIndex | None = None
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Load existing index or create a new one."""
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        docstore_path = self._persist_dir / "docstore.json"
        if docstore_path.exists():
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self._persist_dir)
                )
                self._index = load_index_from_storage(storage_context)
                logger.debug(f"Loaded existing vector index from {self._persist_dir}")
            except Exception as e:
                logger.warning(f"Failed to load index, creating new: {e}")
                self._index = VectorStoreIndex([])
        else:
            self._index = VectorStoreIndex([])
            logger.debug("Created new vector index")

    def create_nodes_for_document(
        self,
        body: str,
        content_hash: str,
        path: str,
        dataset_name: str,
        *,
        additional_metadata: dict[str, Any] | None = None,
    ) -> list[TextNode]:
        """Create vector nodes for a document.

        Chunks the document body and creates TextNodes with stable IDs
        and metadata for vector indexing.

        Args:
            body: The normalized document body text.
            content_hash: SHA256 hash of the document content.
            path: Document path within the dataset.
            dataset_name: Name of the source dataset.
            additional_metadata: Optional additional metadata.

        Returns:
            List of TextNodes ready for indexing.
        """
        chunks = self._chunker.chunk(body)
        nodes = []

        for chunk in chunks:
            node = create_vector_node(
                chunk=chunk,
                content_hash=content_hash,
                path=path,
                dataset_name=dataset_name,
                additional_metadata=additional_metadata,
            )
            nodes.append(node)

        logger.debug(f"Created {len(nodes)} nodes for {path}")
        return nodes

    def add_nodes(self, nodes: list[TextNode]) -> None:
        """Add nodes to the vector index.

        Args:
            nodes: List of TextNodes to add.
        """
        if not nodes:
            return

        if self._index is None:
            self._index = VectorStoreIndex([])

        self._index.insert_nodes(nodes)
        logger.debug(f"Added {len(nodes)} nodes to index")

    def delete_nodes_for_document(self, content_hash: str) -> int:
        """Delete all nodes for a document by content hash.

        Removes all nodes with IDs starting with the given hash.

        Args:
            content_hash: SHA256 hash of the document content.

        Returns:
            Number of nodes deleted.
        """
        if self._index is None:
            return 0

        # Get all node IDs matching the hash
        docstore = self._index.docstore
        ids_to_delete = [
            node_id
            for node_id in docstore.docs.keys()
            if node_id.startswith(f"{content_hash}:")
        ]

        if not ids_to_delete:
            return 0

        # Delete nodes
        for node_id in ids_to_delete:
            self._index.delete_ref_doc(node_id, delete_from_docstore=True)

        logger.debug(f"Deleted {len(ids_to_delete)} nodes for hash {content_hash[:8]}...")
        return len(ids_to_delete)

    def persist(self) -> None:
        """Persist the vector index to disk."""
        if self._index is None:
            return

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index.storage_context.persist(persist_dir=str(self._persist_dir))
        logger.debug(f"Persisted vector index to {self._persist_dir}")

    def count(self) -> int:
        """Count nodes in the index.

        Returns:
            Number of nodes in the index.
        """
        if self._index is None:
            return 0
        return len(self._index.docstore.docs)


class VectorSearch:
    """Semantic similarity search using vector embeddings.

    Provides search functionality with dataset filtering and
    document-level deduplication.

    Example:
        search = VectorSearch()
        results = search.search(
            "how to implement authentication",
            dataset_name="my-vault",
            limit=10,
        )
    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize the vector search.

        Args:
            persist_dir: Directory for vector store persistence.
                Defaults to settings.vector_store_path.
            embedding_model: Name/path of the embedding model.
                Defaults to settings.embedding_model.
        """
        settings = get_settings()
        self._persist_dir = persist_dir or settings.vector_store_path
        self._embedding_model = embedding_model or settings.embedding_model

        # Initialize embedding model
        self._embed_model = HuggingFaceEmbedding(
            model_name=self._embedding_model,
        )
        LlamaSettings.embed_model = self._embed_model

        # Load index
        self._index: VectorStoreIndex | None = None
        self._load_index()

    def _load_index(self) -> None:
        """Load the vector index from storage."""
        docstore_path = self._persist_dir / "docstore.json"
        if docstore_path.exists():
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self._persist_dir)
                )
                self._index = load_index_from_storage(storage_context)
                logger.debug(f"Loaded vector index from {self._persist_dir}")
            except Exception as e:
                logger.warning(f"Failed to load vector index: {e}")
                self._index = None
        else:
            logger.debug("No vector index found")
            self._index = None

    def search(
        self,
        query: str,
        *,
        dataset_name: str | None = None,
        limit: int = 10,
        dedupe_by_path: bool = True,
    ) -> SearchResults:
        """Search for similar documents.

        Args:
            query: The search query.
            dataset_name: Filter by dataset name. None for global search.
            limit: Maximum number of results.
            dedupe_by_path: If True, keep only the best chunk per document.

        Returns:
            SearchResults with ranked results.
        """
        import time

        start_time = time.time()

        if self._index is None:
            return SearchResults(
                results=[],
                query=query,
                mode="vector",
                total_candidates=0,
                timing_ms=0.0,
            )

        # Query the index - get more candidates for filtering
        retriever = self._index.as_retriever(
            similarity_top_k=limit * 3 if dedupe_by_path else limit
        )
        nodes = retriever.retrieve(query)

        # Filter by dataset if specified
        if dataset_name:
            nodes = [
                n for n in nodes
                if n.metadata.get("dataset_name") == dataset_name
            ]

        total_candidates = len(nodes)

        # Convert to SearchResults
        results = []
        for node in nodes:
            result = SearchResult(
                path=node.metadata.get("path", ""),
                dataset_name=node.metadata.get("dataset_name", ""),
                score=node.score if node.score is not None else 0.0,
                chunk_text=node.text,
                chunk_seq=node.metadata.get("seq"),
                chunk_pos=node.metadata.get("pos"),
                metadata={},
                scores={"vector": node.score if node.score is not None else 0.0},
            )
            results.append(result)

        # Deduplicate by path (keep best chunk per document)
        if dedupe_by_path:
            results = self._dedupe_by_path(results)

        # Normalize scores to 0-1
        results = self._normalize_scores(results)

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:limit]

        timing_ms = (time.time() - start_time) * 1000

        return SearchResults(
            results=results,
            query=query,
            mode="vector",
            total_candidates=total_candidates,
            timing_ms=timing_ms,
        )

    def search_with_scores(
        self,
        query: str,
        *,
        dataset_name: str | None = None,
        limit: int = 100,
    ) -> list[tuple[str, str, float]]:
        """Search and return (path, dataset_name, score) tuples.

        Used for hybrid search fusion.

        Args:
            query: The search query.
            dataset_name: Filter by dataset name.
            limit: Maximum number of results.

        Returns:
            List of (path, dataset_name, score) tuples.
        """
        results = self.search(
            query,
            dataset_name=dataset_name,
            limit=limit,
            dedupe_by_path=True,
        )
        return [
            (r.path, r.dataset_name, r.score)
            for r in results.results
        ]

    def _dedupe_by_path(self, results: list[SearchResult]) -> list[SearchResult]:
        """Deduplicate results by path, keeping the best chunk per document.

        Args:
            results: List of SearchResult objects.

        Returns:
            Deduplicated list with best chunk per document.
        """
        best_by_path: dict[tuple[str, str], SearchResult] = {}

        for result in results:
            key = (result.dataset_name, result.path)
            if key not in best_by_path or result.score > best_by_path[key].score:
                best_by_path[key] = result

        return list(best_by_path.values())

    def _normalize_scores(self, results: list[SearchResult]) -> list[SearchResult]:
        """Normalize scores to 0-1 range using max normalization.

        Args:
            results: List of SearchResult objects.

        Returns:
            Results with normalized scores.
        """
        if not results:
            return results

        max_score = max(r.score for r in results)
        if max_score <= 0:
            return results

        for result in results:
            result.score = result.score / max_score
            result.scores["vector"] = result.score

        return results
