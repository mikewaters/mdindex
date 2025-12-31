# Command-line Usage

This document provides comprehensive documentation for the PMD (Python Markdown Search) command-line interface.

---

## Table of Contents

1. [Global Options](#1-global-options)
2. [Configuration](#2-configuration)
3. [Collection Commands](#3-collection-commands)
4. [Search Commands](#4-search-commands)
5. [Indexing Commands](#5-indexing-commands)
6. [Status Command](#6-status-command)
7. [Use Cases](#7-use-cases)
8. [Search Examples](#8-search-examples)

---

## 1. Global Options

All PMD commands support the following global options:

```
pmd [OPTIONS] COMMAND [ARGS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-v, --version` | flag | — | Show version number |
| `-L, --log-level` | choice | `WARNING` | Set logging level: DEBUG, INFO, WARNING, ERROR |
| `-c, --config` | path | `$PMD_CONFIG` | Path to TOML configuration file |
| `--phoenix-tracing` | flag | — | Enable OpenTelemetry tracing to Arize Phoenix |
| `--phoenix-endpoint` | url | — | Phoenix OTLP endpoint (overrides config) |

**Examples:**

```bash
# Run with debug logging
pmd -L DEBUG search "query"

# Use custom config file
pmd -c ./my-config.toml collection list

# Enable tracing
pmd --phoenix-tracing search "query"
```

---

## 2. Configuration

### 2.1 Configuration File Format (TOML)

PMD uses TOML configuration files. Create a `pmd.toml` file:

```toml
# Main configuration
db_path = "~/.cache/pmd/index.db"
llm_provider = "mlx"  # Options: mlx, lm-studio, openrouter

# MLX local model configuration (Apple Silicon)
[mlx]
model = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
embedding_model = "mlx-community/nomicai-modernbert-embed-base-4bit"
embedding_dimension = 768
query_prefix = "search_query: "
document_prefix = "search_document: "
max_tokens = 256
temperature = 0.7
lazy_load = true

# LM Studio configuration (optional)
[lm_studio]
base_url = "http://localhost:1234"
embedding_model = "nomic-embed-text"
expansion_model = "qwen2:0.5b"
reranker_model = "qwen2:0.5b"
timeout = 120.0

# OpenRouter API configuration (optional)
[openrouter]
api_key = ""  # Set via OPENROUTER_API_KEY env var
base_url = "https://openrouter.io/api/v1"
embedding_model = "nomic-ai/nomic-embed-text"
expansion_model = "qwen/qwen-1.5-0.5b"
reranker_model = "qwen/qwen-1.5-0.5b"
timeout = 120.0

# Search pipeline configuration
[search]
default_limit = 5
fts_weight = 1.0
vec_weight = 1.0
rrf_k = 60
top_rank_bonus = 0.05
expansion_weight = 0.5
rerank_candidates = 30

# Document chunking configuration
[chunk]
max_bytes = 6144    # ~2000 tokens per chunk
min_chunk_size = 100

# Phoenix/OpenTelemetry tracing (optional)
[tracing]
enabled = false
phoenix_endpoint = "http://localhost:6006/v1/traces"
service_name = "pmd"
service_version = "1.0.0"
sample_rate = 1.0
batch_export = true
```

### 2.2 Configuration Options Reference

#### Main Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `db_path` | Path | `~/.cache/pmd/index.db` | SQLite database file path |
| `llm_provider` | str | `"mlx"` | LLM provider: mlx, lm-studio, openrouter |

#### MLX Configuration (`[mlx]`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | `"mlx-community/Qwen2.5-1.5B-Instruct-4bit"` | Text generation model |
| `embedding_model` | str | `"mlx-community/nomicai-modernbert-embed-base-4bit"` | Embedding model |
| `embedding_dimension` | int | `768` | Embedding vector dimension |
| `query_prefix` | str | `"search_query: "` | Prefix for query embeddings |
| `document_prefix` | str | `"search_document: "` | Prefix for document embeddings |
| `max_tokens` | int | `256` | Maximum tokens to generate |
| `temperature` | float | `0.7` | Sampling temperature |
| `lazy_load` | bool | `true` | Lazy-load models on first use |

#### Search Configuration (`[search]`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_limit` | int | `5` | Default maximum results |
| `fts_weight` | float | `1.0` | Full-text search weight in hybrid search |
| `vec_weight` | float | `1.0` | Vector search weight in hybrid search |
| `rrf_k` | int | `60` | Reciprocal Rank Fusion k parameter |
| `top_rank_bonus` | float | `0.05` | Bonus for top-ranked results |
| `expansion_weight` | float | `0.5` | Query expansion results weight |
| `rerank_candidates` | int | `30` | Candidates to rerank |

#### Chunk Configuration (`[chunk]`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_bytes` | int | `6144` | Maximum chunk size in bytes |
| `min_chunk_size` | int | `100` | Minimum chunk size |

#### Tracing Configuration (`[tracing]`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable Phoenix tracing |
| `phoenix_endpoint` | str | `"http://localhost:6006/v1/traces"` | Phoenix OTLP endpoint |
| `service_name` | str | `"pmd"` | Service name for traces |
| `sample_rate` | float | `1.0` | Tracing sample rate (0.0-1.0) |
| `batch_export` | bool | `true` | Use batch export for traces |

### 2.3 Environment Variables

Environment variables override configuration file values:

| Variable | Overrides | Example |
|----------|-----------|---------|
| `PMD_CONFIG` | Config file path | `PMD_CONFIG=./pmd.toml` |
| `LLM_PROVIDER` | `llm_provider` | `LLM_PROVIDER=openrouter` |
| `INDEX_PATH` | `db_path` | `INDEX_PATH=~/.cache/pmd/index.db` |
| `LM_STUDIO_URL` | `lm_studio.base_url` | `LM_STUDIO_URL=http://localhost:1234` |
| `OPENROUTER_API_KEY` | `openrouter.api_key` | `OPENROUTER_API_KEY=sk_...` |
| `MLX_MODEL` | `mlx.model` | `MLX_MODEL=qwen2...` |
| `MLX_EMBEDDING_MODEL` | `mlx.embedding_model` | `MLX_EMBEDDING_MODEL=nomic...` |
| `PHOENIX_TRACING` | `tracing.enabled` | `PHOENIX_TRACING=1` |
| `PHOENIX_ENDPOINT` | `tracing.phoenix_endpoint` | `PHOENIX_ENDPOINT=http://...` |
| `PHOENIX_SAMPLE_RATE` | `tracing.sample_rate` | `PHOENIX_SAMPLE_RATE=0.5` |

### 2.4 Configuration Precedence

1. **CLI flags** (highest priority)
2. **Environment variables**
3. **Configuration file**
4. **Built-in defaults** (lowest priority)

---

## 3. Collection Commands

Collections define document sources for indexing and searching.

### 3.1 `pmd collection add`

Add a collection from a source.

```
pmd collection add NAME PATH [OPTIONS]
```

**Arguments:**
- `NAME` — Collection name (unique identifier)
- `PATH` — Directory path, URL, or entity URI

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-g, --glob` | pattern | `**/*.md` | File glob pattern (filesystem sources) |
| `-s, --source` | choice | `filesystem` | Source type: filesystem, http, entity |
| `--sitemap` | url | — | Sitemap URL (HTTP sources) |
| `--auth-type` | choice | `none` | Auth type: none, bearer, basic, api_key |
| `--auth-token` | str | — | Auth token (or `$ENV:VAR_NAME` reference) |
| `--username` | str | — | Username (basic auth) |

**Examples:**

```bash
# Add local markdown files
pmd collection add docs ./docs --glob "**/*.md"

# Add with different pattern
pmd collection add notes ~/notes --glob "*.txt"

# Add HTTP source with sitemap
pmd collection add api-docs https://api.example.com \
  --source http \
  --sitemap https://api.example.com/sitemap.xml

# Add with bearer token auth
pmd collection add private https://docs.example.com \
  --source http \
  --auth-type bearer \
  --auth-token "$ENV:BEARER_TOKEN"
```

### 3.2 `pmd collection list`

List all configured collections.

```
pmd collection list
```

**Output:**
```
Collections (3):
  📁 docs
    Path: /home/user/project/docs
    Pattern: **/*.md
    Updated: 2025-12-30T10:30:00

  📁 notes
    Path: /home/user/notes
    Pattern: *.txt
    Updated: 2025-12-30T09:15:00
```

### 3.3 `pmd collection show`

Show detailed information about a collection.

```
pmd collection show NAME
```

**Example:**
```bash
pmd collection show docs
```

### 3.4 `pmd collection remove`

Remove a collection and its associated documents.

```
pmd collection remove NAME
```

**Example:**
```bash
pmd collection remove old-docs
```

### 3.5 `pmd collection rename`

Rename a collection.

```
pmd collection rename OLD_NAME NEW_NAME
```

**Example:**
```bash
pmd collection rename docs documentation
```

---

## 4. Search Commands

PMD provides three search modes with increasing sophistication.

### 4.1 `pmd search` (Full-Text Search)

BM25 full-text search using SQLite FTS5. No LLM required.

```
pmd search QUERY [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-l, --limit` | int | `5` | Maximum results to return |
| `-c, --collection` | str | — | Limit to specific collection |
| `-s, --score` | float | `0.0` | Minimum score threshold (0.0-1.0) |

**Example:**
```bash
pmd search "authentication patterns" --limit 10
```

### 4.2 `pmd vsearch` (Vector Search)

Semantic vector search using embeddings. Requires LLM provider for embedding generation.

```
pmd vsearch QUERY [OPTIONS]
```

**Options:** Same as `pmd search`

**Example:**
```bash
pmd vsearch "how to validate tokens" --collection api-docs
```

### 4.3 `pmd query` (Hybrid Search)

Hybrid search combining FTS + vector search with optional LLM reranking. Most comprehensive but requires LLM.

```
pmd query QUERY [OPTIONS]
```

**Options:** Same as `pmd search`

**Example:**
```bash
pmd query "async error handling" -c myproject --score 0.5 --limit 10
```

### 4.4 Search Output Format

```
Search Results (5)
======================================================================
1. Authentication Guide [0.850]
   File: docs/auth/guide.md
   FTS Score: 0.720
   Vector Score: 0.910
   Rerank: 0.950 (Yes) conf=0.95
   Blend: 40% RRF + 60% rerank

2. Token Validation [0.780]
   File: docs/auth/tokens.md
   ...
```

---

## 5. Indexing Commands

### 5.1 `pmd index`

Index all documents in a collection (generates FTS index).

```
pmd index COLLECTION [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-f, --force` | flag | Force reindex of all documents |
| `--embed` | flag | Generate embeddings after indexing |

**Examples:**
```bash
# Index a collection
pmd index docs

# Force reindex with embeddings
pmd index docs --force --embed
```

### 5.2 `pmd embed`

Generate embeddings for documents in a collection.

```
pmd embed COLLECTION [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-f, --force` | flag | Force re-embedding of all documents |

**Example:**
```bash
pmd embed docs --force
```

### 5.3 `pmd update-all`

Update all collections (index and optionally embed).

```
pmd update-all [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--embed` | flag | Generate embeddings after indexing |

**Example:**
```bash
pmd update-all --embed
```

### 5.4 `pmd cleanup`

Clean up orphaned data and cache.

```
pmd cleanup
```

---

## 6. Status Command

### 6.1 `pmd status`

Show index status and synchronization information.

```
pmd status [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--check-sync` | flag | — | Check FTS and vector index synchronization |
| `--collection` | str | — | Limit sync check to specific collection |
| `--sync-limit` | int | `20` | Max sample paths per sync category |

**Examples:**
```bash
# Basic status
pmd status

# Check synchronization
pmd status --check-sync

# Check specific collection
pmd status --check-sync --collection docs
```

**Output:**
```
PMD Index Status
==================================================
Collections: 3
Documents: 150
Embedded: 120
Index Size: 2048000 bytes

Collections:
  • docs (/path/to/docs)
  • api (/path/to/api)
  • blog (/path/to/blog)
```

---

## 7. Use Cases

### 7.1 Indexing a New Corpus

**Step 1: Add a collection**
```bash
pmd collection add my-docs ./documents --glob "**/*.md"
```

**Step 2: Index the collection**
```bash
pmd index my-docs
```

**Step 3: Generate embeddings for vector search**
```bash
pmd embed my-docs
```

Or combine indexing and embedding:
```bash
pmd index my-docs --embed
```

**Step 4: Verify**
```bash
pmd status
```

### 7.2 Searching a Corpus

**Full-text search (fast, keyword-based):**
```bash
pmd search "error handling" --limit 10
```

**Vector search (semantic, requires embeddings):**
```bash
pmd vsearch "how to handle errors gracefully"
```

**Hybrid search (best results, combines both):**
```bash
pmd query "async error handling patterns" --score 0.5
```

**Search within a specific collection:**
```bash
pmd query "authentication" --collection api-docs
```

### 7.3 Maintaining the Index

**Update all collections after changes:**
```bash
pmd update-all --embed
```

**Check synchronization status:**
```bash
pmd status --check-sync
```

**Clean up orphaned data:**
```bash
pmd cleanup
```

---

## 8. Search Examples

The following examples are derived from the PMD integration test suite and demonstrate real search patterns.

### 8.1 Vector Search for ML Topics

**Query:** Find documents about clustering algorithms

```bash
pmd vsearch "unsupervised machine learning clustering partitioning data points"
```

**Result:** Finds "K-means clustering.md"

This demonstrates how semantic vector search matches concepts even when the query uses different terminology than the document.

### 8.2 Hybrid Search with Technical Keywords

**Query:** Find graph database documentation

```bash
pmd query "weaviate neo4j graph database"
```

**Result:** Finds "Graph Databases.md"

Hybrid search combines exact keyword matching (FTS finds "weaviate", "neo4j") with semantic understanding for comprehensive results.

### 8.3 Conceptual Search (Synonym Matching)

**Query:** Describe the concept without using technical terms

```bash
pmd vsearch "partitioning data into groups using centroids and distance"
```

**Result:** Finds "K-means clustering.md" (without using "K-means" in query)

Vector search understands that "centroids and distance" relates to K-means clustering, demonstrating semantic matching beyond keywords.

### 8.4 Full-Text Search with Collection Filtering

**Query:** Search for common terms within a specific collection

```bash
pmd search "authentication" --collection api-docs --limit 10
```

**Result:** Returns documents from the api-docs collection ranked by BM25 relevance

FTS is fast and effective for keyword-based searches when you know the terminology.

### 8.5 Exploratory Search Across Domains

**Query:** Find documents about relationships and connections

```bash
pmd vsearch "storing relationships between entities connected data"
```

**Result:** Finds "Graph Databases.md"

This shows how embeddings capture meaning, matching documents about graph databases when searching for "relationships between entities" without using database terminology.

### 8.6 Multi-Topic Search with Score Filtering

**Query:** Search across multiple topics with minimum relevance threshold

```bash
pmd query "machine learning clustering algorithm unsupervised graph database" \
  --limit 10 \
  --score 0.5
```

**Result:** Returns only documents with relevance score above 0.5

Use `--score` to filter out low-relevance matches when queries span multiple topics.

