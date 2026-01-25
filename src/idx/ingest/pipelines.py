"""Ingest pipeline for indexing documents.

Provides the IngestPipeline class for ingesting documents from various
sources into the idx system, persisting them to the database and
updating derived indexes (FTS, vector).

Uses LlamaIndex's IngestionPipeline for document transformations with
PersistenceTransform handling database persistence and FTS indexing
as the final pipeline step. Uses ambient session via contextvars.

Pipeline flow:
1. TextNormalizerTransform (normalize whitespace, BOM, etc.)
2. PersistenceTransform (upsert to documents table + documents_fts)
3. MarkdownNodeParser (split into chunks)
4. ChunkPersistenceTransform (upsert chunks to chunks_fts)
5. [Optional] Embedding computation (via HuggingFaceEmbedding)
6. [Optional] Vector store insertion (via VectorStoreManager)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.ingestion.pipeline import DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode

from idx.core.logging import get_logger
from idx.ingest.schemas import IngestDirectoryConfig, IngestObsidianConfig, IngestResult
from idx.ingest.directory import DirectorySource
from idx.ingest.obsidian import ObsidianVaultSource
from idx.ingest.sources import create_source
from idx.store.database import get_session
from idx.store.fts import create_fts_table
from idx.store.fts_chunk import create_chunks_fts_table
from idx.store.dataset import DatasetService, normalize_dataset_name
from idx.store.session_context import use_session
from idx.ingest.cache import load_pipeline, persist_pipeline

from idx.transform.llama import (
    PersistenceTransform,
    TextNormalizerTransform,
    ChunkPersistenceTransform,
)

if TYPE_CHECKING:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from idx.store.vector import VectorStoreManager

__all__ = [
    "IngestPipeline",
]

logger = get_logger(__name__)


# TODO: make `config` generic to support multiple ingestion types
def _get_source_instance(config):
    match config:
        case IngestDirectoryConfig():
            return DirectorySource(
                config.source_path,
                patterns=config.patterns,
                encoding=config.encoding,
            )
        case IngestObsidianConfig():
            return ObsidianVaultSource(config.source_path)
        case _:
            raise TypeError(f"Unsupported config type: {type(config)}")


class IngestPipeline:
    """Pipeline for ingesting documents from sources.

    Pipeline flow:
    1. TextNormalizerTransform - normalize whitespace, BOM, etc.
    2. PersistenceTransform - upsert to documents table + documents_fts
    3. MarkdownNodeParser - split into chunks (TextNodes)
    4. ChunkPersistenceTransform - upsert chunks to chunks_fts
    5. [Optional] Embedding computation - via HuggingFaceEmbedding
    6. [Optional] Vector store insertion - via VectorStoreManager

    Uses LlamaIndex's IngestionPipeline for document transformations
    with PersistenceTransform and ChunkPersistenceTransform handling
    database persistence and FTS indexing within the pipeline.

    Vector indexing is optional and controlled by the config flag
    `enable_vector_indexing`. When enabled:
    - Embeddings are computed using HuggingFaceEmbedding
    - Vectors are inserted into SimpleVectorStore via VectorStoreManager
    - Both FTS and vector stores are updated atomically

    Note: Stale document handling has been moved to idx.store.cleanup.
    Use cleanup_stale_documents() for maintenance operations.

    Example:
        config = IngestDirectoryConfig(
            directory=Path("/path/to/docs"),
            dataset_name="my-docs",
            patterns=["**/*.md"],
            enable_vector_indexing=True,
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)
        print(f"Processed {result.total_processed} documents")
        print(f"Chunks: {result.chunks_created}, Vectors: {result.vectors_inserted}")
    """

    def __init__(self) -> None:
        """Initialize the IngestPipeline.

        Lazy-initializes embedding model and vector store manager
        when vector indexing is enabled.
        """
        self._embed_model: "HuggingFaceEmbedding | None" = None
        self._vector_store_manager: "VectorStoreManager | None" = None

    def _get_embed_model(self) -> "HuggingFaceEmbedding":
        """Get or create the embedding model (lazy initialization).

        Returns:
            HuggingFaceEmbedding instance configured from settings.
        """
        if self._embed_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from idx.core.settings import get_settings

            settings = get_settings()
            logger.debug(f"Loading embedding model: {settings.embedding_model}")
            self._embed_model = HuggingFaceEmbedding(
                model_name=settings.embedding_model,
                embed_batch_size=settings.performance.embedding_batch_size,
            )
            logger.info(f"Embedding model loaded: {settings.embedding_model}")

        return self._embed_model

    def _get_vector_store_manager(self) -> "VectorStoreManager":
        """Get or create the VectorStoreManager (lazy initialization).

        Returns:
            VectorStoreManager instance.
        """
        if self._vector_store_manager is None:
            from idx.store.vector import VectorStoreManager

            self._vector_store_manager = VectorStoreManager()

        return self._vector_store_manager

    def _compute_embeddings(
        self,
        nodes: list[TextNode],
    ) -> list[TextNode]:
        """Compute embeddings for nodes using batch processing.

        Args:
            nodes: List of TextNodes to embed.

        Returns:
            The same nodes with embeddings set.
        """
        if not nodes:
            return nodes

        embed_model = self._get_embed_model()

        # Extract texts for batch embedding
        texts = [node.get_content() for node in nodes]

        logger.debug(f"Computing embeddings for {len(texts)} chunks")
        embeddings = embed_model.get_text_embedding_batch(texts)

        # Assign embeddings to nodes
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        logger.info(f"Computed embeddings for {len(nodes)} chunks")
        return nodes

    def _insert_vectors(
        self,
        nodes: list[TextNode],
    ) -> int:
        """Insert nodes into the vector store.

        Args:
            nodes: List of TextNodes with embeddings.

        Returns:
            Number of vectors inserted.
        """
        if not nodes:
            return 0

        manager = self._get_vector_store_manager()

        # Ensure index is loaded or created
        manager.load_or_create()

        logger.debug(f"Inserting {len(nodes)} vectors into store")
        manager.insert_nodes(nodes)

        return len(nodes)

    def ingest(self, config: IngestDirectoryConfig | IngestObsidianConfig) -> IngestResult:
        """Ingest documents from a local directory.

        Creates or retrieves the dataset, enumerates matching files,
        and runs them through the LlamaIndex transformation pipeline
        with PersistenceTransform and ChunkPersistenceTransform handling
        persistence and FTS indexing.

        When `enable_vector_indexing` is True, also computes embeddings
        and inserts vectors into the vector store. Both FTS and vector
        stores are updated atomically - on failure, both are rolled back.

        Notes:
        - This implementation leans on LLamaIndex node caching, and
        so we cannot know if a document was updated/skipped until after
        running the pipeline.

        - LlamaIndex can technically split Documents into TextNodes,
        but effectively this pipeline treats each Document as a single node,
        and so the terms are used interchangeably.

        Args:
            config: Ingestion configuration, either directory or Obsidian.

        Returns:
            IngestResult with statistics about the operation.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """

        started_at = datetime.now(tz=timezone.utc)
        normalized_name = normalize_dataset_name(config.dataset_name)

        logger.info(
            f"Starting directory ingestion: {config.source_path} -> {normalized_name}"
        )

        source = _get_source_instance(config)

        # Track results
        result = IngestResult(
            dataset_id=0,  # Will be set after dataset creation
            dataset_name=normalized_name,
            started_at=started_at,
        )

        # Track nodes for vector indexing (collected during pipeline run)
        nodes_for_vectors: list[TextNode] = []

        with get_session() as session:
            with use_session(session):

                # Ensure FTS tables exist
                #TODO: move to migration?
                engine = session.get_bind()
                if engine is not None:
                    create_fts_table(engine)  # type: ignore
                    create_chunks_fts_table(engine)  # type: ignore

                # Create or get dataset
                dataset_id = DatasetService.create_or_update(
                    session,
                    config.dataset_name,
                    source_type=source.type_name,
                    source_path=str(source.path),
                )
                result.dataset_id = dataset_id

                # Create persistence transform (uses ambient session)
                persist = PersistenceTransform(
                    dataset_id=dataset_id,
                    force=config.force,
                )
                split = MarkdownNodeParser(
                    include_metadata=True,
                    include_prev_next_rel=True,
                    header_path_separator=" / ",
                )
                # Create chunk persistence transform (uses ambient session)
                chunk_persist = ChunkPersistenceTransform(
                    dataset_name=normalized_name,
                )

                # Build pipeline with transforms + persistence + chunk persistence
                pipeline = IngestionPipeline(
                    transformations=[
                        TextNormalizerTransform(),
                        persist,
                        split,
                        chunk_persist,
                    ],
                    docstore=SimpleDocumentStore(),
                    docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
                )

                # Load persisted pipeline if available
                if not config.force:
                    pipeline = load_pipeline(normalized_name, pipeline)

                # Run pipeline - persistence happens inside using ambient session
                logger.info(f"Running {len(source.documents)} documents through pipeline")
                nodes = pipeline.run(documents=source.documents)

                # Update the cache
                persist_pipeline(normalized_name, pipeline)

                # Copy stats from persistence transform
                result.documents_created = persist.stats.created
                result.documents_updated = persist.stats.updated
                result.documents_skipped = persist.stats.skipped
                result.documents_failed = persist.stats.failed
                result.errors = list(persist.stats.errors)

                # Copy stats from chunk persistence transform
                result.chunks_created = chunk_persist.stats.created

                # this only works because the reader doesn't split documents
                result.documents_read = len(source.documents)

                # Collect nodes for vector indexing if enabled
                if config.enable_vector_indexing and nodes:
                    # Filter to only TextNodes (chunks from MarkdownNodeParser)
                    nodes_for_vectors = [n for n in nodes if isinstance(n, TextNode)]

                # Vector indexing happens inside the session context for atomic rollback
                if config.enable_vector_indexing and nodes_for_vectors:
                    try:
                        # Compute embeddings
                        self._compute_embeddings(nodes_for_vectors)

                        # Insert into vector store
                        vectors_inserted = self._insert_vectors(nodes_for_vectors)
                        result.vectors_inserted = vectors_inserted

                        # Persist vector store after successful insertion
                        manager = self._get_vector_store_manager()
                        manager.persist()

                        logger.info(f"Vector indexing complete: {vectors_inserted} vectors inserted")

                    except Exception as e:
                        error_msg = f"Vector indexing failed: {e}"
                        logger.error(error_msg)
                        result.errors.append(error_msg)
                        # Re-raise to trigger session rollback
                        raise

        result.completed_at = datetime.now(tz=timezone.utc)

        logger.info(
            f"Ingestion complete: "
            f"created={result.documents_created}, "
            f"updated={result.documents_updated}, "
            f"skipped={result.documents_skipped}, "
            f"failed={result.documents_failed}, "
            f"chunks={result.chunks_created}, "
            f"vectors={result.vectors_inserted}"
        )

        return result

