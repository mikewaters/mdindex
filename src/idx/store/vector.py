"""idx.store.vector - Vector store management using LlamaIndex.

Provides vector storage and retrieval capabilities using LlamaIndex's
SimpleVectorStore with StorageContext for persistence. Embeddings are
generated using HuggingFace models.

Example usage:
    from idx.store.vector import VectorStoreManager

    manager = VectorStoreManager()
    index = manager.load_or_create()
    manager.insert_nodes([node1, node2])
    manager.persist()
    retriever = manager.get_retriever(similarity_top_k=10)
"""

from pathlib import Path
from typing import TYPE_CHECKING

from idx.core.logging import get_logger
from idx.core.settings import get_settings

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.schema import TextNode

__all__ = ["VectorStoreManager"]

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages vector storage using LlamaIndex SimpleVectorStore.

    Provides lazy initialization of the vector index and embedding model.
    Uses StorageContext for persistence to disk.

    The manager supports:
    - Loading an existing index or creating a new one
    - Inserting nodes with automatic embedding generation
    - Deleting nodes by ID or by reference document ID
    - Getting a retriever for similarity search

    Attributes:
        persist_dir: Directory path for persisting the vector store.
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        """Initialize the VectorStoreManager.

        Args:
            persist_dir: Directory for persisting the vector store.
                If None, uses settings.vector_store_path.
        """
        settings = get_settings()
        self._persist_dir = persist_dir or settings.vector_store_path
        self._embed_settings = settings.embedding

        # Lazy-initialized components
        self._index: "VectorStoreIndex | None" = None
        self._embed_model: "BaseEmbedding | None" = None

        logger.debug(
            f"VectorStoreManager initialized with persist_dir={self._persist_dir}"
        )

    @property
    def persist_dir(self) -> Path:
        """Get the persistence directory path."""
        return self._persist_dir

    def _get_embed_model(self) -> "BaseEmbedding":
        """Get or create the embedding model (lazy initialization).

        Returns the configured embedding model based on settings.embedding.backend:
        - "mlx": MLXEmbedding for Apple Silicon
        - "huggingface": HuggingFaceEmbedding for general use

        Returns:
            BaseEmbedding instance configured from settings.
        """
        if self._embed_model is None:
            if self._embed_settings.backend == "mlx":
                from idx.embedding.mlx import MLXEmbedding

                logger.debug(f"Loading MLX embedding model: {self._embed_settings.model_name}")
                self._embed_model = MLXEmbedding(
                    model_name=self._embed_settings.model_name,
                    embed_batch_size=self._embed_settings.batch_size,
                )
                logger.info(f"MLX embedding model loaded: {self._embed_settings.model_name}")
            else:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                logger.debug(f"Loading HuggingFace embedding model: {self._embed_settings.model_name}")
                self._embed_model = HuggingFaceEmbedding(
                    model_name=self._embed_settings.model_name,
                    embed_batch_size=self._embed_settings.batch_size,
                )
                logger.info(f"HuggingFace embedding model loaded: {self._embed_settings.model_name}")

        return self._embed_model

    def _create_new_index(self) -> "VectorStoreIndex":
        """Create a new empty vector store index.

        Returns:
            New VectorStoreIndex with SimpleVectorStore backend.
        """
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.core.vector_stores import SimpleVectorStore

        logger.debug("Creating new vector store index")

        # Create empty vector store and storage context
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create empty index
        index = VectorStoreIndex.from_documents(
            documents=[],
            storage_context=storage_context,
            embed_model=self._get_embed_model(),
            show_progress=False,
        )

        logger.info("New vector store index created")
        return index

    def _load_existing_index(self) -> "VectorStoreIndex":
        """Load an existing index from disk.

        Returns:
            VectorStoreIndex loaded from persist_dir.

        Raises:
            FileNotFoundError: If the persist_dir doesn't exist.
        """
        from llama_index.core import StorageContext, load_index_from_storage

        logger.debug(f"Loading vector store index from {self._persist_dir}")

        storage_context = StorageContext.from_defaults(
            persist_dir=str(self._persist_dir)
        )
        index = load_index_from_storage(
            storage_context,
            embed_model=self._get_embed_model(),
        )

        logger.info(f"Vector store index loaded from {self._persist_dir}")
        return index

    def load_or_create(self) -> "VectorStoreIndex":
        """Load an existing index or create a new one.

        If the persist directory exists and contains a valid index,
        loads it. Otherwise, creates a new empty index.

        Returns:
            VectorStoreIndex ready for use.
        """
        if self._index is not None:
            return self._index

        # Check if persist directory exists and has index files
        persist_path = self._persist_dir
        docstore_path = persist_path / "docstore.json"

        if persist_path.exists() and docstore_path.exists():
            try:
                self._index = self._load_existing_index()
            except Exception as e:
                logger.warning(
                    f"Failed to load existing index, creating new one: {e}"
                )
                self._index = self._create_new_index()
        else:
            # Ensure directory exists for future persistence
            persist_path.mkdir(parents=True, exist_ok=True)
            self._index = self._create_new_index()

        return self._index

    def persist(self) -> None:
        """Save the index to disk.

        Persists the vector store, docstore, and index metadata
        to the configured persist_dir.

        Raises:
            RuntimeError: If the index hasn't been loaded or created yet.
        """
        if self._index is None:
            raise RuntimeError(
                "No index to persist. Call load_or_create() first."
            )

        # Ensure directory exists
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._index.storage_context.persist(persist_dir=str(self._persist_dir))
        logger.info(f"Vector store persisted to {self._persist_dir}")

    def insert_nodes(self, nodes: list["TextNode"]) -> None:
        """Add nodes to the index with automatic embedding generation.

        Nodes will have their embeddings computed if not already present.
        Existing nodes with the same ID will be updated.

        Args:
            nodes: List of TextNode objects to insert.

        Raises:
            RuntimeError: If the index hasn't been loaded or created yet.
        """
        if not nodes:
            logger.debug("No nodes to insert")
            return

        if self._index is None:
            raise RuntimeError(
                "No index loaded. Call load_or_create() first."
            )

        logger.debug(f"Inserting {len(nodes)} nodes into vector store")
        self._index.insert_nodes(nodes)
        logger.info(f"Inserted {len(nodes)} nodes into vector store")

    def delete_nodes(self, node_ids: list[str]) -> None:
        """Delete specific nodes from the index by their IDs.

        Args:
            node_ids: List of node IDs to delete.

        Raises:
            RuntimeError: If the index hasn't been loaded or created yet.
        """
        if not node_ids:
            logger.debug("No node IDs to delete")
            return

        if self._index is None:
            raise RuntimeError(
                "No index loaded. Call load_or_create() first."
            )

        logger.debug(f"Deleting {len(node_ids)} nodes from vector store")
        self._index.delete_nodes(node_ids)
        logger.info(f"Deleted {len(node_ids)} nodes from vector store")

    def delete_ref_doc(self, ref_doc_id: str) -> None:
        """Delete all nodes associated with a reference document.

        This removes all chunks/nodes that were derived from the
        specified source document.

        Args:
            ref_doc_id: Reference document ID (source_doc_id).

        Raises:
            RuntimeError: If the index hasn't been loaded or created yet.
        """
        if self._index is None:
            raise RuntimeError(
                "No index loaded. Call load_or_create() first."
            )

        logger.debug(f"Deleting nodes for ref_doc_id: {ref_doc_id}")
        self._index.delete_ref_doc(ref_doc_id)
        logger.info(f"Deleted nodes for ref_doc_id: {ref_doc_id}")

    def get_retriever(
        self,
        similarity_top_k: int = 10,
    ) -> "VectorIndexRetriever":
        """Get a retriever for similarity search.

        Args:
            similarity_top_k: Number of most similar nodes to retrieve.

        Returns:
            VectorIndexRetriever configured for the index.

        Raises:
            RuntimeError: If the index hasn't been loaded or created yet.
        """
        if self._index is None:
            raise RuntimeError(
                "No index loaded. Call load_or_create() first."
            )

        logger.debug(f"Creating retriever with similarity_top_k={similarity_top_k}")
        return self._index.as_retriever(similarity_top_k=similarity_top_k)

    def clear(self) -> None:
        """Clear the in-memory index cache.

        This forces the index to be reloaded on next access.
        Does not delete persisted data.
        """
        self._index = None
        logger.debug("Vector store index cache cleared")

    @property
    def is_loaded(self) -> bool:
        """Check if an index is currently loaded in memory."""
        return self._index is not None
