"""idx.transform.embedding - Embedding generation pipeline transform.

Provides a LlamaIndex TransformComponent that generates embeddings for nodes
and inserts them into the vector store. Designed for use in ingestion pipelines.

Example usage:
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from idx.transform.embedding import EmbeddingTransform
    from idx.store.vector import VectorStoreManager

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vector_manager = VectorStoreManager()
    vector_manager.load_or_create()

    embedding_transform = EmbeddingTransform(
        embed_model=embed_model,
        vector_store_manager=vector_manager,
        batch_size=32,
    )
    pipeline = IngestionPipeline(transformations=[embedding_transform])
    nodes = pipeline.run(nodes=nodes)
"""

from typing import TYPE_CHECKING, Any

from llama_index.core.schema import BaseNode, TransformComponent

from idx.core.logging import get_logger

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding

    from idx.store.vector import VectorStoreManager

__all__ = ["EmbeddingTransform"]

logger = get_logger(__name__)


class EmbeddingTransform(TransformComponent):
    """LlamaIndex TransformComponent that generates embeddings and stores nodes.

    This transform handles embedding generation and vector store insertion:
    1. Extracts text content from each node
    2. Generates embeddings in batches using the configured embedding model
    3. Attaches embeddings to nodes (node.embedding = embedding)
    4. Inserts nodes into the vector store via VectorStoreManager

    The transform returns nodes unchanged (with embeddings attached) so they
    can be used by potential downstream transforms.

    Attributes:
        embed_model: The embedding model to use for generating embeddings.
        vector_store_manager: Manager for vector store operations.
        batch_size: Number of texts to embed in each batch.
    """

    # Private attributes (not Pydantic fields)
    _embed_model: "BaseEmbedding | None" = None
    _vector_store_manager: "VectorStoreManager | None" = None
    _batch_size: int = 32

    def __init__(
        self,
        embed_model: "BaseEmbedding",
        vector_store_manager: "VectorStoreManager",
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        """Initialize the embedding transform.

        Args:
            embed_model: The embedding model to use (e.g., HuggingFaceEmbedding).
            vector_store_manager: VectorStoreManager instance for storing nodes.
                The manager should have load_or_create() called before use.
            batch_size: Number of texts to embed in each batch. Defaults to 32.
            **kwargs: Additional arguments passed to TransformComponent.
        """
        super().__init__(**kwargs)
        self._embed_model = embed_model
        self._vector_store_manager = vector_store_manager
        self._batch_size = batch_size

    @property
    def embed_model(self) -> "BaseEmbedding":
        """Get the embedding model."""
        if self._embed_model is None:
            raise RuntimeError("Embedding model not set")
        return self._embed_model

    @property
    def vector_store_manager(self) -> "VectorStoreManager":
        """Get the vector store manager."""
        if self._vector_store_manager is None:
            raise RuntimeError("VectorStoreManager not set")
        return self._vector_store_manager

    @property
    def batch_size(self) -> int:
        """Get the batch size for embedding generation."""
        return self._batch_size

    def __call__(
        self,
        nodes: list[BaseNode],
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Generate embeddings for nodes and insert into vector store.

        Processes nodes in batches:
        1. Extracts text content from each node
        2. Generates embeddings using embed_model.get_text_embedding_batch()
        3. Attaches embeddings to nodes (node.embedding = embedding)
        4. Inserts all nodes into the vector store

        Args:
            nodes: List of nodes to process.
            **kwargs: Additional arguments (unused).

        Returns:
            The same nodes with embeddings attached, for downstream transforms.
        """
        if not nodes:
            logger.debug("No nodes to embed")
            return nodes

        logger.debug(f"Generating embeddings for {len(nodes)} nodes")

        # Extract text content from all nodes
        texts = [node.get_content() for node in nodes]

        # Generate embeddings in batches
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch_texts = texts[i : i + self._batch_size]
            batch_embeddings = self.embed_model.get_text_embedding_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                f"Embedded batch {i // self._batch_size + 1} "
                f"({len(batch_texts)} texts)"
            )

        # Attach embeddings to nodes
        for node, embedding in zip(nodes, all_embeddings):
            node.embedding = embedding

        logger.info(f"Generated embeddings for {len(nodes)} nodes")

        # Insert nodes into vector store
        self.vector_store_manager.insert_nodes(nodes)

        return nodes
