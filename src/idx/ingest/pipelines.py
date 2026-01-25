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
5. SizeAwareChunkSplitter (split oversized nodes for embedding)
6. embed_model (generate embeddings via native vector_store integration)

Vector store integration uses LlamaIndex's native vector_store parameter
with DocstoreStrategy.UPSERTS for proper upsert semantics on document changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.ingestion.pipeline import DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import MarkdownNodeParser

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
from idx.ingest.cache import load_pipeline, persist_pipeline, clear_cache

from idx.transform.llama import (
    PersistenceTransform,
    TextNormalizerTransform,
    ChunkPersistenceTransform,
)
from idx.transform.splitter import SizeAwareChunkSplitter

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
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
    5. SizeAwareChunkSplitter - split oversized nodes for embedding
    6. embed_model - generate embeddings via native vector_store integration

    Uses LlamaIndex's IngestionPipeline with native vector_store parameter
    for vector storage. This enables DocstoreStrategy.UPSERTS which properly
    handles document updates by re-embedding changed content.

    Vector indexing is always performed using the configured embedding
    backend (MLX or HuggingFace).

    Note: Stale document handling has been moved to idx.store.cleanup.
    Use cleanup_stale_documents() for maintenance operations.

    Example:
        config = IngestDirectoryConfig(
            directory=Path("/path/to/docs"),
            dataset_name="my-docs",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)
        print(f"Processed {result.total_processed} documents")
        print(f"Chunks: {result.chunks_created}, Vectors: {result.vectors_inserted}")
    """

    def __init__(self) -> None:
        """Initialize the IngestPipeline.

        Lazy-initializes embedding model and vector store manager.
        """
        self._embed_model: "BaseEmbedding | None" = None
        self._vector_store_manager: "VectorStoreManager | None" = None

    def _get_embed_model(self) -> "BaseEmbedding":
        """Get or create the embedding model (lazy initialization).

        Returns the configured embedding model based on settings.embedding.backend:
        - "mlx": MLXEmbedding for Apple Silicon
        - "huggingface": HuggingFaceEmbedding for general use

        Returns:
            BaseEmbedding instance configured from settings.
        """
        if self._embed_model is None:
            from idx.core.settings import get_settings

            settings = get_settings()
            embed_settings = settings.embedding

            if embed_settings.backend == "mlx":
                from idx.embedding.mlx import MLXEmbedding

                logger.debug(f"Loading MLX embedding model: {embed_settings.model_name}")
                self._embed_model = MLXEmbedding(
                    model_name=embed_settings.model_name,
                    embed_batch_size=embed_settings.batch_size,
                )
                logger.info(f"MLX embedding model loaded: {embed_settings.model_name}")
            else:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                logger.debug(f"Loading HuggingFace embedding model: {embed_settings.model_name}")
                self._embed_model = HuggingFaceEmbedding(
                    model_name=embed_settings.model_name,
                    embed_batch_size=embed_settings.batch_size,
                )
                logger.info(f"HuggingFace embedding model loaded: {embed_settings.model_name}")

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

    def ingest(self, config: IngestDirectoryConfig | IngestObsidianConfig) -> IngestResult:
        """Ingest documents from a local directory.

        Creates or retrieves the dataset, enumerates matching files,
        and runs them through the LlamaIndex transformation pipeline
        with PersistenceTransform and ChunkPersistenceTransform handling
        persistence and FTS indexing.

        Vector indexing uses LlamaIndex's native vector_store integration
        with DocstoreStrategy.UPSERTS for proper handling of document updates.

        Notes:
        - This implementation uses LlamaIndex's docstore caching. Documents
        with unchanged content hashes are skipped. Changed documents are
        re-embedded automatically via the UPSERTS strategy.

        - LlamaIndex can technically split Documents into TextNodes,
        but effectively this pipeline treats each Document as a single node,
        and so the terms are used interchangeably.

        - When force=True, the pipeline cache and dataset vectors are cleared
        before running, ensuring a complete re-index.

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

                # Create size-aware splitter for oversized chunks
                size_splitter = SizeAwareChunkSplitter(
                    max_chars=2000,
                    fallback_chunk_size=512,
                    fallback_chunk_overlap=50,
                )

                # Get vector store manager and embed model
                vector_manager = self._get_vector_store_manager()
                embed_model = self._get_embed_model()

                # Handle force=True: clear cache and dataset vectors
                if config.force:
                    logger.info(f"Force mode: clearing cache for dataset '{normalized_name}'")
                    clear_cache(normalized_name)
                    deleted = vector_manager.delete_by_dataset(normalized_name)
                    if deleted > 0:
                        logger.info(f"Cleared {deleted} vectors for dataset '{normalized_name}'")

                # Get the vector store for native pipeline integration
                vector_store = vector_manager.get_vector_store()

                # Build pipeline with native vector_store integration
                # Using UPSERTS strategy for proper handling of document updates
                pipeline = IngestionPipeline(
                    transformations=[
                        TextNormalizerTransform(),
                        persist,
                        split,
                        chunk_persist,
                        size_splitter,
                        embed_model,
                    ],
                    docstore=SimpleDocumentStore(),
                    docstore_strategy=DocstoreStrategy.UPSERTS,
                    vector_store=vector_store,
                )

                # Load persisted pipeline docstore if available
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

                # Vectors are inserted by native vector_store integration during pipeline run
                # Count is equal to the number of nodes returned by the pipeline
                result.vectors_inserted = len(nodes) if nodes else 0

                # Persist vector store after successful pipeline run
                vector_manager.persist_vector_store()

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

