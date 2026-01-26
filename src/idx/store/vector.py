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
    from llama_index.core.vector_stores import SimpleVectorStore

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
        self._vector_store: "SimpleVectorStore | None" = None

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

        # stores_text must be set after construction (not accepted in __init__)
        vector_store = SimpleVectorStore()
        vector_store.stores_text = True
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

    def _create_index_from_vector_store(self) -> "VectorStoreIndex":
        """Create an index from a persisted vector store file.

        Used when only the vector store was persisted (not the full index).
        This happens when persist_vector_store() was called instead of persist().

        Returns:
            VectorStoreIndex created from the loaded vector store.
        """
        from llama_index.core import StorageContext, VectorStoreIndex

        logger.debug(f"Creating index from vector store at {self._persist_dir}")

        # Load the vector store
        vector_store = self.get_vector_store()

        # Create storage context with the loaded vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from the vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self._get_embed_model(),
        )

        logger.info(f"Index created from vector store at {self._persist_dir}")
        return index

    def load_or_create(self) -> "VectorStoreIndex":
        """Load an existing index or create a new one.

        If the persist directory exists and contains a valid index,
        loads it. Otherwise, creates a new empty index.

        Handles two persistence scenarios:
        1. Full index (docstore.json exists): Load via StorageContext
        2. Vector store only (default__vector_store.json exists): Create index from vector store

        Returns:
            VectorStoreIndex ready for use.
        """
        if self._index is not None:
            return self._index

        persist_path = self._persist_dir
        vector_store_path = persist_path / "default__vector_store.json"
        docstore_path = persist_path / "docstore.json"

        if persist_path.exists() and docstore_path.exists():
            # Full index exists (from StorageContext.persist)
            try:
                self._index = self._load_existing_index()
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self._index = self._create_new_index()
        elif persist_path.exists() and vector_store_path.exists():
            # Vector store only (from persist_vector_store)
            # Create index from the persisted vector store
            try:
                self._index = self._create_index_from_vector_store()
            except Exception as e:
                logger.warning(f"Failed to create index from vector store: {e}")
                self._index = self._create_new_index()
        else:
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
        self._vector_store = None
        logger.debug("Vector store index cache cleared")

    @property
    def is_loaded(self) -> bool:
        """Check if an index is currently loaded in memory."""
        return self._index is not None

    def get_vector_store(self) -> "SimpleVectorStore":
        """Get or create the raw SimpleVectorStore for pipeline integration.

        Returns the underlying SimpleVectorStore instance, creating it if needed.
        This is used to pass the vector store to IngestionPipeline's vector_store
        parameter for native integration.

        If an index already exists (from load_or_create), returns its vector store.
        Otherwise creates a new empty SimpleVectorStore.

        Returns:
            SimpleVectorStore instance for pipeline use.
        """
        from llama_index.core.vector_stores import SimpleVectorStore

        if self._vector_store is not None:
            return self._vector_store

        # If we have an index, get its vector store
        if self._index is not None:
            self._vector_store = self._index.storage_context.vector_store
            return self._vector_store

        # Check if we can load from disk
        persist_path = self._persist_dir
        vector_store_path = persist_path / "default__vector_store.json"

        if vector_store_path.exists():
            logger.debug(f"Loading vector store from {vector_store_path}")
            self._vector_store = SimpleVectorStore.from_persist_path(
                str(vector_store_path)
            )
            # stores_text isn't persisted, must set after load for from_vector_store() support
            self._vector_store.stores_text = True
            logger.info(f"Vector store loaded from {vector_store_path}")
        else:
            logger.debug("Creating new empty vector store")
            # stores_text must be set after construction (not accepted in __init__)
            self._vector_store = SimpleVectorStore()
            self._vector_store.stores_text = True

        return self._vector_store

    def delete_by_dataset(self, dataset_name: str) -> int:
        """Delete all vectors associated with a dataset.

        Removes all nodes whose ref_doc_id starts with '{dataset_name}:'.
        This is used by force=True to clear dataset vectors before re-indexing.

        Args:
            dataset_name: Name of the dataset to clear vectors for.

        Returns:
            Number of nodes deleted.
        """
        vector_store = self.get_vector_store()
        prefix = f"{dataset_name}:"
        deleted_count = 0

        # Get all node IDs that match the dataset prefix
        # SimpleVectorStore stores data in embedding_dict keyed by node_id
        # We need to find nodes by examining the docstore or metadata
        if self._index is None:
            # Try to load the index to access the docstore
            try:
                self.load_or_create()
            except Exception as e:
                logger.warning(f"Could not load index for deletion: {e}")
                return 0

        if self._index is not None:
            # Get ref_doc_info which maps ref_doc_id to node_ids
            ref_doc_info = self._index.ref_doc_info
            node_ids_to_delete = []

            for ref_doc_id, info in ref_doc_info.items():
                if ref_doc_id.startswith(prefix):
                    node_ids_to_delete.extend(info.node_ids)

            if node_ids_to_delete:
                logger.debug(
                    f"Deleting {len(node_ids_to_delete)} nodes for dataset '{dataset_name}'"
                )
                self._index.delete_nodes(node_ids_to_delete)
                deleted_count = len(node_ids_to_delete)
                logger.info(
                    f"Deleted {deleted_count} vectors for dataset '{dataset_name}'"
                )

        return deleted_count

    def persist_vector_store(self, persist_dir: Path | None = None) -> None:
        """Persist just the vector store to disk.

        Used for persisting after pipeline.run() when using native
        vector_store integration.

        Args:
            persist_dir: Directory to persist to. Defaults to self._persist_dir.
        """
        if self._vector_store is None:
            logger.debug("No vector store to persist")
            return

        target_dir = persist_dir or self._persist_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        vector_store_path = target_dir / "default__vector_store.json"
        self._vector_store.persist(str(vector_store_path))
        logger.info(f"Vector store persisted to {vector_store_path}")
