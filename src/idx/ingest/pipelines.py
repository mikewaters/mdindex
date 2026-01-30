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
from dataclasses import dataclass, field

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

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.extractors import KeywordExtractor, TitleExtractor
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.storage.docstore.duckdb import DuckDBDocumentStore
from llama_index.vector_stores.duckdb import DuckDBVectorStore

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from idx.store.vector import VectorStoreManager

__all__ = [
    "IngestPipeline",
]

logger = get_logger(__name__)


# # TODO: make `config` generic to support multiple ingestion types
# def _get_source_instance(config):
#     match config:
#         case IngestDirectoryConfig():
#             return DirectorySource(
#                 config.source_path,
#                 patterns=config.patterns,
#                 encoding=config.encoding,
#             )
#         case IngestObsidianConfig():
#             return ObsidianVaultSource(config.source_path)
#         case _:
#             raise TypeError(f"Unsupported config type: {type(config)}")

# -----------------------------------------------------------------------------
# Bronze Layer Components
# -----------------------------------------------------------------------------


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of document content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_meta_hash(metadata: dict[str, Any]) -> str:
    """Compute SHA256 hash of metadata dict."""
    # Sort keys for deterministic hashing
    import json

    normalized = json.dumps(metadata, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class BronzeMetadataTransform(TransformComponent):
    """
    Transform that adds bronze-layer versioning metadata to documents.

    This transform:
    1. Computes content_hash from document text
    2. Adds bronze_meta_version from config
    3. Will compute bronze_meta_hash after other extractors run
    """
    meta_version: str = "1.0.0"
    #def __init__(self, meta_version: str = "1.0.0"):
    #    self.meta_version = meta_version

    def __call__(self, nodes: list[BaseNode], **kwargs: Any) -> list[BaseNode]:
        for node in nodes:
            # Add content hash
            node.metadata["content_hash"] = compute_content_hash(node.get_content())

            # Add version
            node.metadata["bronze_meta_version"] = self.meta_version

        return nodes


class BronzeMetaHashTransform(TransformComponent):
    """
    Final transform that computes bronze_meta_hash after all extraction.

    Must run AFTER all other metadata extractors.
    """

    def __call__(self, nodes: list[BaseNode], **kwargs: Any) -> list[BaseNode]:
        for node in nodes:
            # Extract only the "extracted" metadata (exclude hashes and version)
            extracted_meta = {
                k: v
                for k, v in node.metadata.items()
                if k not in ("content_hash", "bronze_meta_version", "bronze_meta_hash")
            }
            node.metadata["bronze_meta_hash"] = compute_meta_hash(extracted_meta)

        return nodes

# -----------------------------------------------------------------------------
# Downstream Layer Components
# -----------------------------------------------------------------------------


class CompositeHashTransform(TransformComponent):
    """
    Transform that creates a composite hash for downstream change detection.

    The composite hash combines content_hash and bronze_meta_hash so that
    downstream re-processes when either changes.
    """
    include_metadata: bool = True
    #def __init__(self, include_metadata: bool = True):
    #    self.include_metadata = include_metadata

    def __call__(self, nodes: list[BaseNode], **kwargs: Any) -> list[BaseNode]:
        for node in nodes:
            content_hash = node.metadata.get("content_hash", "")
            meta_hash = node.metadata.get("bronze_meta_hash", "")

            if self.include_metadata:
                composite = f"{content_hash}:{meta_hash}"
            else:
                composite = content_hash

            # Store composite hash - this participates in LlamaIndex's dedup
            node.metadata["downstream_hash"] = hashlib.sha256(
                composite.encode()
            ).hexdigest()

        return nodes


class NonFlatVectorDBStripMetadataTransform(TransformComponent):
    """
    For vector databases whicih store originating metadata alongside embeddings,
    if they do not support objects (and you are using them in metadata),
    this will strip those keys to avoid a runtimeerror.
    DuckDB, I am looking at you.
    Qdrant supports this, AHEM.
    But really, this goes against LLamaindex best-practices, metadata should be flat.
    I should instead store raw metadata elsewhere, or process it correctly.
    Or I could prefix the individual nodes, but then obsidian
    might have nested metadata of its own.
    """
    keys_to_strip: list[str] = field(default_factory=lambda: [])

    def __call__(self, nodes: list[BaseNode], **kwargs: Any) -> list[BaseNode]:
        for node in nodes:
            for key in self.keys_to_strip:
                node.metadata.pop(key, None)
        return nodes

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

    # def __init__(self) -> None:
    #     """Initialize the IngestPipeline.

    #     Lazy-initializes embedding model and vector store manager.
    #     """
    #     self._embed_model: "BaseEmbedding | None" = None
    #     self._vector_store_manager: "VectorStoreManager | None" = None

    def _get_embed_model(self) -> "BaseEmbedding":
        """Get or create the embedding model (lazy initialization).

        Returns the configured embedding model based on settings.embedding.backend:
        - "mlx": MLXEmbedding for Apple Silicon
        - "huggingface": HuggingFaceEmbedding for general use

        Returns:
            BaseEmbedding instance configured from settings.
        """
        #if self._embed_model is None:
        from idx.core.settings import get_settings

        settings = get_settings()
        embed_settings = settings.embedding

        if embed_settings.backend == "mlx":
            from idx.embedding.mlx import MLXEmbedding

            logger.debug(f"Loading MLX embedding model: {embed_settings.model_name}")
            return MLXEmbedding(
                model_name=embed_settings.model_name,
                embed_batch_size=embed_settings.batch_size,
            )
            logger.info(f"MLX embedding model loaded: {embed_settings.model_name}")
        else:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            logger.debug(f"Loading HuggingFace embedding model: {embed_settings.model_name}")
            return HuggingFaceEmbedding(
                model_name=embed_settings.model_name,
                embed_batch_size=embed_settings.batch_size,
            )
            logger.info(f"HuggingFace embedding model loaded: {embed_settings.model_name}")

        #return self._embed_model

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

        #source = _get_source_instance(config)
        source = create_source(config)

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
                from idx.store.vector import VectorStoreManager
                vector_manager = VectorStoreManager()
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
                #if not config.force:
                #    pipeline = load_pipeline(normalized_name, pipeline)

                # Run pipeline - persistence happens inside using ambient session
                logger.info(f"Running {len(source.documents)} documents through pipeline")
                nodes = pipeline.run(documents=source.documents)

                # Update the cache
                #persist_pipeline(normalized_name, pipeline)

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

    def ingest_test(self, config: IngestObsidianConfig) -> IngestResult:
        """Ingest documents from a local directory.

        """

        started_at = datetime.now(tz=timezone.utc)
        normalized_name = normalize_dataset_name(config.dataset_name)

        logger.info(
            f"Starting directory ingestion: {config.source_path} -> {normalized_name}"
        )

        #source = _get_source_instance(config)
        source = create_source(config)

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
                #vector_manager = self._get_vector_store_manager()
                embed_model = self._get_embed_model()
                Settings.embed_model = embed_model

                # Handle force=True: clear cache and dataset vectors
                # if config.force:
                #     logger.info(f"Force mode: clearing cache for dataset '{normalized_name}'")
                #     clear_cache(normalized_name)
                #     deleted = vector_manager.delete_by_dataset(normalized_name)
                #     if deleted > 0:
                #         logger.info(f"Cleared {deleted} vectors for dataset '{normalized_name}'")

                # Get the vector store for native pipeline integration
                from idx.store.vector import VectorStoreManager
                vector_manager = VectorStoreManager()
                vector_store = vector_manager.get_vector_store()
                # Create DuckDB vector store for downstream
                vector_store = DuckDBVectorStore(
                    database_name="downstream.duckdb",
                    persist_dir="./downstream_persist",
                    flat_metadata=False
                )
                # Build pipeline with native vector_store integration
                # Using UPSERTS strategy for proper handling of document updates
                pipeline = IngestionPipeline(
                    transformations=[
                        TextNormalizerTransform(),
                        BronzeMetadataTransform(meta_version="1.0.0"),
                        BronzeMetaHashTransform(),
                        # Add composite hash for change detection
                        CompositeHashTransform(include_metadata=True),
                        persist,
                        split,
                        chunk_persist,
                        size_splitter,  # get rid of this, add a real embedding splitter
                        StripMetadataTransform(keys_to_strip=["frontmatter", "wikilinks", "backlinks"]),
                        embed_model,
                        
                    ],
                    docstore=SimpleDocumentStore(),
                    docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
                    vector_store=vector_store,
                )

                # Load persisted pipeline docstore if available
                #if not config.force:
                #    pipeline = load_pipeline(normalized_name, pipeline)

                # Run pipeline - persistence happens inside using ambient session
                logger.info(f"Running {len(source.documents)} documents through pipeline")
                for doc in source.documents:
                    print(f"Document: {doc.metadata}")
                    print(doc.excluded_embed_metadata_keys)
                nodes = pipeline.run(documents=source.documents)

                # Update the cache
                #persist_pipeline(normalized_name, pipeline)

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

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m idx.ingest.two_stage <source_dir>")
        sys.exit(1)

    config = IngestObsidianConfig(source_path=Path(sys.argv[1]))

    pipeline = IngestPipeline() 
    result = pipeline.ingest_test(
        config
    )