# Multiple Ingestion Pipeline Design

**Date:** 2026-01-27
**Status:** Implemented (Demo)
**Spec Reference:** `features/multiple-ingest.md`
**Implementation:** `src/idx/ingest/two_stage.py`

---

## Summary

This document describes the design and implementation of a standalone two-stage ingestion demo using pure LlamaIndex abstractions. The module demonstrates the bronze/downstream architecture from the spec without integrating with existing `idx` infrastructure.

---

## Architecture

```
Source Directory
       │
       ▼
┌──────────────────────────────────────────────────┐
│           BRONZE PIPELINE                         │
│  ┌────────────────────────────────────────────┐  │
│  │ SimpleDirectoryReader                       │  │
│  │     ↓                                       │  │
│  │ BronzeMetadataTransform (content_hash, ver) │  │
│  │     ↓                                       │  │
│  │ [Optional: TitleExtractor, KeywordExtractor]│  │
│  │     ↓                                       │  │
│  │ BronzeMetaHashTransform (meta_hash)         │  │
│  └────────────────────────────────────────────┘  │
│                      ↓                            │
│            DuckDBDocumentStore                    │
│            (bronze.duckdb)                        │
└──────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│         DOWNSTREAM PIPELINE                       │
│  ┌────────────────────────────────────────────┐  │
│  │ Read from DuckDBDocumentStore               │  │
│  │     ↓                                       │  │
│  │ CompositeHashTransform (change detection)   │  │
│  │     ↓                                       │  │
│  │ SentenceSplitter (chunking)                 │  │
│  │     ↓                                       │  │
│  │ HuggingFaceEmbedding                        │  │
│  └────────────────────────────────────────────┘  │
│                      ↓                            │
│            DuckDBVectorStore                      │
│            (downstream.duckdb)                    │
└──────────────────────────────────────────────────┘
```

---

## Key Components

### Bronze Layer

| Component | Purpose |
|-----------|---------|
| `DuckDBDocumentStore` | Canonical document storage with metadata |
| `BronzeMetadataTransform` | Adds `content_hash` and `bronze_meta_version` |
| `BronzeMetaHashTransform` | Computes `bronze_meta_hash` after extraction |
| `DocstoreStrategy.UPSERTS` | Skip unchanged documents on re-run |

### Downstream Layer

| Component | Purpose |
|-----------|---------|
| `CompositeHashTransform` | Creates hash from content + metadata for change detection |
| `SentenceSplitter` | Chunks documents |
| `HuggingFaceEmbedding` | Generates embeddings |
| `DuckDBVectorStore` | Persistent vector storage |
| `SimpleDocumentStore` | Tracks processed chunks for dedup |

---

## Data Contract

### Bronze Document Metadata

Each document in bronze contains:

```python
{
    "content_hash": str,        # SHA256 of document text
    "bronze_meta_version": str, # e.g., "1.0.0"
    "bronze_meta_hash": str,    # SHA256 of extracted metadata
    # ... plus any extracted fields (title, keywords, etc.)
}
```

### Change Detection Rules

| content_hash | bronze_meta_hash | Downstream Action |
|--------------|------------------|-------------------|
| Changed | - | Reprocess |
| Same | Changed | Reprocess (if `change_detection="content_and_metadata"`) |
| Same | Same | Skip |

---

## Public API

### `ingest_to_bronze(source_dir, config, file_patterns) -> IngestResult`

Runs the bronze pipeline: loads documents, extracts metadata, stores to DuckDB.

### `build_vector_index(bronze_db_path, config) -> (VectorStoreIndex, IngestResult)`

Runs downstream: reads from bronze, chunks, embeds, builds vector index.

### `reprocess_source(source_dir, bronze_config, downstream_config) -> (IngestResult, IngestResult)`

Convenience wrapper running both pipelines.

---

## Configuration

### BronzeConfig

```python
@dataclass
class BronzeConfig:
    db_path: str = "bronze.duckdb"      # DuckDB database path
    meta_version: str = "1.0.0"         # Bump to force re-extraction
    use_llm_extractors: bool = False    # Enable TitleExtractor, etc.
```

### DownstreamConfig

```python
@dataclass
class DownstreamConfig:
    db_path: str = "downstream.duckdb"
    persist_dir: str = "./downstream_persist"
    chunk_size: int = 512
    chunk_overlap: int = 50
    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    change_detection: str = "content_and_metadata"  # or "content_only"
```

---

## Usage

```python
from idx.ingest.two_stage import (
    ingest_to_bronze,
    build_vector_index,
    BronzeConfig,
    DownstreamConfig,
)

# Stage 1: Ingest to bronze
bronze_result = ingest_to_bronze(
    source_dir="./docs",
    config=BronzeConfig(meta_version="1.0.0"),
)

# Stage 2: Build vector index from bronze
index, downstream_result = build_vector_index(
    bronze_db_path="bronze.duckdb",
    config=DownstreamConfig(chunk_size=512),
)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is this about?")
```

### CLI

```bash
python -m idx.ingest.two_stage ./my_docs/
```

---

## Spec Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| DuckDB-backed bronze docstore | ✅ | `DuckDBDocumentStore` |
| Metadata extraction only in prep | ✅ | No chunking/embedding in bronze |
| `content_hash` + `bronze_meta_hash` versioning | ✅ | Custom transforms |
| Downstream reads from bronze | ✅ | Loads docs from DuckDB |
| Independent persistence | ✅ | Separate DuckDB files |
| `DocstoreStrategy.UPSERTS` | ✅ | Both pipelines |
| Pluggable reader interface | ✅ | `SimpleDirectoryReader` |
| Change detection rules | ✅ | `CompositeHashTransform` |

---

## Limitations / Future Work

1. **No LLM extractors by default** - Requires API key; disabled for demo simplicity
2. **No deletion handling** - Would need `UPSERTS_AND_DELETE` strategy
3. **Single embedding model** - Could support multiple downstream pipelines with different models
4. **No incremental bronze reads** - Downstream reads all bronze docs; could optimize
5. **Not integrated with existing idx** - Intentional for demo isolation
