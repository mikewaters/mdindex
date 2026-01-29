"""
Two-stage ingestion pipeline demonstration.

This module demonstrates a two-tier ingestion architecture using LlamaIndex:
1. Bronze stage: Load documents, extract metadata, store in DuckDB docstore
2. Downstream stage: Read from bronze, chunk, embed, store in vector store

Key design goals:
- Bronze layer serves as canonical source with metadata versioning
- Downstream can be rebuilt independently from bronze
- Change detection based on content_hash + bronze_meta_hash
"""

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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class BronzeConfig:
    """Configuration for the bronze (preparatory) pipeline."""

    # DuckDB database path for bronze docstore
    db_path: str = "bronze.db"

    # Metadata version - bump this to force re-extraction
    meta_version: str = "1.0.0"

    # Whether to use LLM-based extractors (requires API key)
    use_llm_extractors: bool = False


@dataclass
class DownstreamConfig:
    """Configuration for the downstream (vector) pipeline."""

    # DuckDB database path for vector store
    db_path: str = "downstream.duckdb"

    # Persist directory for downstream state
    persist_dir: str = "./downstream_persist"

    # Chunking parameters
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Embedding model
    embed_model_name: str = "BAAI/bge-small-en-v1.5"

    # Change detection: "content_only" or "content_and_metadata"
    change_detection: str = "content_and_metadata"


@dataclass
class IngestResult:
    """Result of an ingestion run."""

    total_documents: int = 0
    documents_processed: int = 0
    documents_skipped: int = 0
    nodes_created: int = 0
    errors: list[str] = field(default_factory=list)


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
    meta_version: str
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


def create_bronze_pipeline(config: BronzeConfig, dataset_id: int) -> IngestionPipeline:
    """
    Create the bronze (preparatory) ingestion pipeline.

    This pipeline:
    - Does NOT chunk documents (keeps them whole)
    - Extracts metadata (title, keywords if LLM enabled)
    - Stores to DuckDB docstore with versioning metadata
    """
    # Create DuckDB docstore for bronze layer
    #docstore = DuckDBDocumentStore.from_local(config.db_path)
    from idx.store.docstore import SQLiteDocumentStore
    docstore = SQLiteDocumentStore(dataset_id=dataset_id) #database_path=Path(config.db_path))

    # Build transformations - metadata extraction only, no chunking
    transformations: list[TransformComponent] = [
        # First: add content hash and version
        BronzeMetadataTransform(meta_version=config.meta_version),
    ]

    # Optional LLM-based extractors
    if config.use_llm_extractors:
        transformations.extend(
            [
                TitleExtractor(nodes=1),
                KeywordExtractor(keywords=5),
            ]
        )

    # Final: compute meta hash after all extraction
    transformations.append(BronzeMetaHashTransform())

    return IngestionPipeline(
        transformations=transformations,
        docstore=docstore,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )


# -----------------------------------------------------------------------------
# Downstream Layer Components
# -----------------------------------------------------------------------------


class CompositeHashTransform(TransformComponent):
    """
    Transform that creates a composite hash for downstream change detection.

    The composite hash combines content_hash and bronze_meta_hash so that
    downstream re-processes when either changes.
    """

    def __init__(self, include_metadata: bool = True):
        self.include_metadata = include_metadata

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


def create_downstream_pipeline(
    config: DownstreamConfig,
    bronze_docstore: DuckDBDocumentStore,
) -> IngestionPipeline:
    """
    Create the downstream (vector) ingestion pipeline.

    This pipeline:
    - Reads documents from bronze docstore
    - Adds composite hash for change detection
    - Chunks documents
    - Embeds chunks
    - Stores in DuckDB vector store
    """
    # Create embedding model
    embed_model = HuggingFaceEmbedding(model_name=config.embed_model_name)
    Settings.embed_model = embed_model

    # Create DuckDB vector store for downstream
    vector_store = DuckDBVectorStore(
        database_name=config.db_path,
        persist_dir=config.persist_dir,
    )

    # Downstream docstore for tracking processed chunks
    downstream_docstore = SimpleDocumentStore()

    # Determine if we use metadata in change detection
    include_metadata = config.change_detection == "content_and_metadata"

    transformations: list[TransformComponent] = [
        # Add composite hash for change detection
        CompositeHashTransform(include_metadata=include_metadata),
        # Chunk documents
        SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        ),
        # Embed (handled by pipeline with embed_model in Settings)
        embed_model,
    ]

    return IngestionPipeline(
        transformations=transformations,
        docstore=downstream_docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
from idx.ingest.schemas import IngestObsidianConfig
from idx.ingest.sources import create_source
def ingest_to_bronze(
    source_config: IngestObsidianConfig,
    dest_config: BronzeConfig | None = None,
) -> IngestResult:
    """
    Run the bronze (preparatory) pipeline.

    Loads documents from source_dir, extracts metadata, and stores
    in DuckDB docstore with versioning information.

    Args:
        source_config: Configuration for the source to ingest
        dest_config: Configuration for the bronze pipeline

    Returns:
        IngestResult with processing statistics
    """
    dest_config = dest_config or BronzeConfig()
    result = IngestResult()
    from idx.store.dataset import DatasetService
    from idx.store.database import get_session
    from idx.store.session_context import use_session
    
    source = create_source(source_config)
    with get_session() as session:
        with use_session(session):
            dataset_id = DatasetService.create_or_update(
                session,
                config.dataset_name,
                source_type=source.type_name,
                source_path=str(source.path),
            )
            result.dataset_id = dataset_id

    

    
    documents = source.documents
    result.total_documents = len(documents)

    if not documents:
        return result

    # Create and run bronze pipeline
    pipeline = create_bronze_pipeline(dest_config, result.dataset_id)

    try:
        nodes = pipeline.run(documents=documents, show_progress=True)
        result.documents_processed = len(nodes)
        result.nodes_created = len(nodes)
    except Exception as e:
        result.errors.append(str(e))

    return result


def build_vector_index(
    bronze_db_path: str | Path,
    config: DownstreamConfig | None = None,
) -> tuple[VectorStoreIndex | None, IngestResult]:
    """
    Run the downstream pipeline from bronze to vector store.

    Reads documents from bronze DuckDB docstore, chunks, embeds,
    and stores in DuckDB vector store.

    Args:
        bronze_db_path: Path to bronze DuckDB database
        config: Downstream pipeline configuration

    Returns:
        Tuple of (VectorStoreIndex, IngestResult)
    """
    config = config or DownstreamConfig()
    result = IngestResult()

    # Load bronze docstore
    bronze_docstore = DuckDBDocumentStore.from_local(str(bronze_db_path))

    # Get all documents from bronze
    doc_ids = list(bronze_docstore.docs.keys())
    result.total_documents = len(doc_ids)

    if not doc_ids:
        return None, result

    # Convert stored nodes back to Documents for downstream processing
    documents = []
    for doc_id in doc_ids:
        node = bronze_docstore.get_document(doc_id)
        if node:
            # Create Document from stored node
            doc = Document(
                doc_id=doc_id,
                text=node.get_content(),
                metadata=node.metadata.copy(),
            )
            documents.append(doc)

    # Create and run downstream pipeline
    pipeline = create_downstream_pipeline(config, bronze_docstore)

    try:
        nodes = pipeline.run(documents=documents, show_progress=True)
        result.documents_processed = len(documents)
        result.nodes_created = len(nodes)

        # Build index from vector store
        vector_store = DuckDBVectorStore(
            database_name=config.db_path,
            persist_dir=config.persist_dir,
        )
        index = VectorStoreIndex.from_vector_store(vector_store)

        return index, result

    except Exception as e:
        result.errors.append(str(e))
        return None, result


def reprocess_source(
    source_dir: str | Path,
    bronze_config: BronzeConfig | None = None,
    downstream_config: DownstreamConfig | None = None,
    file_patterns: list[str] | None = None,
) -> tuple[IngestResult, IngestResult]:
    """
    Convenience function: run both bronze and downstream pipelines.

    Args:
        source_dir: Directory containing source documents
        bronze_config: Bronze pipeline configuration
        downstream_config: Downstream pipeline configuration
        file_patterns: Glob patterns for files

    Returns:
        Tuple of (bronze_result, downstream_result)
    """
    bronze_config = bronze_config or BronzeConfig()
    downstream_config = downstream_config or DownstreamConfig()

    # Run bronze pipeline
    bronze_result = ingest_to_bronze(
        source_dir=source_dir,
        config=bronze_config,
        file_patterns=file_patterns,
    )

    # Run downstream pipeline
    _, downstream_result = build_vector_index(
        bronze_db_path=bronze_config.db_path,
        config=downstream_config,
    )

    return bronze_result, downstream_result


# -----------------------------------------------------------------------------
# Demo / CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m idx.ingest.two_stage <source_dir>")
        sys.exit(1)

    config = IngestObsidianConfig(source_path=Path(sys.argv[1]))

    print(f"Running two-stage ingestion on: {config.source_path}")

    bronze_result = ingest_to_bronze(config)

    print("\n--- Bronze Pipeline Results ---")
    print(f"  Total documents: {bronze_result.total_documents}")
    print(f"  Processed: {bronze_result.documents_processed}")
    print(f"  Errors: {bronze_result.errors}")

    # print("\n--- Downstream Pipeline Results ---")
    # print(f"  Total from bronze: {downstream_result.total_documents}")
    # print(f"  Nodes created: {downstream_result.nodes_created}")
    # print(f"  Errors: {downstream_result.errors}")
