# PMD - Python Markdown Search

A hybrid search engine for markdown documents with full-text search, vector semantic search, and LLM-powered relevance ranking.

## Features

- **BM25 Full-Text Search** via SQLite FTS5
- **Vector Semantic Search** via sqlite-vec embeddings
- **LLM Re-ranking** via LM Studio (default) or OpenRouter
- **Hybrid Search Pipeline** with Reciprocal Rank Fusion (RRF)
- **Query Expansion** for improved recall
- **Content-Addressable Storage** for efficient deduplication
- **MCP Server Integration** for AI agent access
- **Multiple Output Formats**: JSON, CSV, XML, Markdown, plain text

## Installation

```bash
# Clone the repository
git clone https://github.com/yourrepo/pmd.git
cd pmd

# Install in development mode
pip install -e ".[dev]"

# Or install just the runtime dependencies
pip install -e .
```

### Requirements

- Python 3.11+
- SQLite with FTS5 support (included in most Python distributions)
- LM Studio (recommended) or OpenRouter API key for LLM features

## Quick Start

### 1. Create a Collection

A collection is a directory of markdown files you want to search.

```bash
# Add a collection named "notes" pointing to your markdown directory
pmd collection add notes /path/to/your/markdown/files

# Optionally specify a custom glob pattern (default: **/*.md)
pmd collection add notes /path/to/files -g "**/*.markdown"
```

### 2. Index Documents

```bash
# Index a specific collection
pmd index notes

# Or update all collections
pmd update-all
```

### 3. Search

```bash
# Basic full-text search (FTS5/BM25)
pmd search "your search query"

# Hybrid search with LLM query expansion and reranking
pmd query "your search query"

# Vector semantic search (requires embeddings)
pmd vsearch "your search query"
```

### 4. Manage Collections

```bash
# List all collections
pmd collection list

# Remove a collection
pmd collection remove notes

# Rename a collection
pmd collection rename notes my-notes
```

### 5. Check Status

```bash
pmd status
```

## CLI Reference

### Search Commands

| Command | Description |
|---------|-------------|
| `pmd search <query>` | BM25 keyword search via FTS5 |
| `pmd vsearch <query>` | Vector semantic search using embeddings |
| `pmd query <query>` | Hybrid search with expansion and reranking |

**Common Options:**
- `-l, --limit N` - Maximum results (default: 5)
- `-c, --collection NAME` - Limit to specific collection
- `-s, --score N` - Minimum score threshold (default: 0.0)

### Collection Commands

| Command | Description |
|---------|-------------|
| `pmd collection add <name> <path>` | Add a new collection |
| `pmd collection list` | List all collections |
| `pmd collection remove <name>` | Remove a collection |
| `pmd collection rename <old> <new>` | Rename a collection |

**Options for `add`:**
- `-g, --glob PATTERN` - File glob pattern (default: `**/*.md`)

### Index Commands

| Command | Description |
|---------|-------------|
| `pmd index <collection>` | Index documents in a collection |
| `pmd update-all` | Update all collections |
| `pmd embed <collection>` | Generate embeddings (requires LLM) |
| `pmd cleanup` | Remove orphaned data |

**Options for `index` and `embed`:**
- `-f, --force` - Force reindex/re-embed all documents

### Other Commands

| Command | Description |
|---------|-------------|
| `pmd status` | Show index status |
| `pmd -v, --version` | Show version |
| `pmd -h, --help` | Show help |

## Python API

### Basic Usage

```python
from pmd.core.config import Config
from pmd.store.database import Database
from pmd.store.collections import CollectionRepository
from pmd.store.documents import DocumentRepository
from pmd.store.search import FTS5SearchRepository

# Initialize
config = Config.from_env()
db = Database(config.db_path)
db.connect()

# Create repositories
collections = CollectionRepository(db)
documents = DocumentRepository(db)
search = FTS5SearchRepository(db)

# Create a collection
collection = collections.create("notes", "/path/to/files", "**/*.md")

# Add a document
doc, is_new = documents.add_or_update(
    collection_id=collection.id,
    path="example.md",
    title="Example Document",
    content="# Example\n\nThis is example content.",
)

# Index for search
search.index_document(doc.filepath, doc.filepath, doc.body)

# Search
results = search.search_fts("example", limit=5)
for result in results:
    print(f"{result.title}: {result.score}")

db.close()
```

### Hybrid Search with LLM

```python
import asyncio
from pmd.core.config import Config
from pmd.store.database import Database
from pmd.store.search import FTS5SearchRepository
from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig
from pmd.llm import create_llm_provider, QueryExpander, DocumentReranker

async def hybrid_search(query: str):
    config = Config.from_env()
    db = Database(config.db_path)
    db.connect()

    # Initialize LLM provider
    llm = create_llm_provider(config)

    # Create search components
    search_repo = FTS5SearchRepository(db)
    expander = QueryExpander(llm)
    reranker = DocumentReranker(llm)

    # Configure pipeline
    pipeline_config = SearchPipelineConfig(
        enable_query_expansion=True,
        enable_reranking=True,
        rerank_candidates=30,
    )

    pipeline = HybridSearchPipeline(
        search_repo,
        pipeline_config,
        query_expander=expander,
        reranker=reranker,
    )

    # Execute search
    results = await pipeline.search(query, limit=5)

    for result in results:
        print(f"{result.title}")
        print(f"  Score: {result.score:.3f}")
        print(f"  FTS: {result.fts_score}, Rerank: {result.rerank_score}")

    await llm.close()
    db.close()

# Run
asyncio.run(hybrid_search("your query"))
```

### Embedding Generation

```python
import asyncio
from pmd.core.config import Config
from pmd.store.database import Database
from pmd.store.embeddings import EmbeddingRepository
from pmd.llm import create_llm_provider
from pmd.llm.embeddings import EmbeddingGenerator

async def generate_embeddings(content: str, hash_value: str):
    config = Config.from_env()
    db = Database(config.db_path)
    db.connect()

    llm = create_llm_provider(config)
    embedding_repo = EmbeddingRepository(db)
    generator = EmbeddingGenerator(llm, embedding_repo, config)

    # Generate embeddings for document
    chunks_embedded = await generator.embed_document(hash_value, content)
    print(f"Embedded {chunks_embedded} chunks")

    # Generate query embedding
    query_embedding = await generator.embed_query("search query")
    print(f"Query embedding dimensions: {len(query_embedding)}")

    await llm.close()
    db.close()

asyncio.run(generate_embeddings("Document content...", "abc123hash"))
```

### MCP Server Integration

```python
import asyncio
from pmd.core.config import Config
from pmd.mcp.server import PMDMCPServer

async def run_mcp_server():
    config = Config.from_env()
    server = PMDMCPServer(config)

    await server.initialize()

    # Search
    results = await server.search("python programming", limit=5)
    print(results)

    # Get document
    doc = await server.get_document("notes", "example.md")
    print(doc)

    # List collections
    collections = await server.list_collections()
    print(collections)

    # Get status
    status = await server.get_status()
    print(status)

    await server.shutdown()

asyncio.run(run_mcp_server())
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `lm-studio` | LLM provider: `lm-studio` or `openrouter` |
| `LM_STUDIO_URL` | `http://localhost:1234` | LM Studio API endpoint |
| `OPENROUTER_API_KEY` | (none) | OpenRouter API key |
| `OPENROUTER_URL` | `https://openrouter.io/api/v1` | OpenRouter endpoint |
| `INDEX_PATH` | `~/.cache/pmd/index.db` | Database file path |
| `XDG_CACHE_HOME` | `~/.cache` | Cache directory base |

### LLM Provider Configuration

**LM Studio (Default)**
```bash
export LLM_PROVIDER=lm-studio
export LM_STUDIO_URL=http://localhost:1234
```

**OpenRouter**
```bash
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-v1-...
```

### Default Models

| Provider | Embedding | Expansion | Reranking |
|----------|-----------|-----------|-----------|
| LM Studio | `nomic-embed-text` | `qwen2:0.5b` | `qwen2:0.5b` |
| OpenRouter | `nomic-ai/nomic-embed-text` | `qwen/qwen-1.5-0.5b` | `qwen/qwen-1.5-0.5b` |

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=pmd
```

### Type Checking

```bash
mypy src/pmd
```

### Linting

```bash
ruff check src/pmd
ruff format src/pmd
```

## License

MIT License - See LICENSE file for details.

## Related Projects

- **QMD**: Original TypeScript implementation
- **sqlite-vec**: Vector similarity search for SQLite
- **LM Studio**: Local LLM inference
- **OpenRouter**: Multi-model API gateway
