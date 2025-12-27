# PMD Python Implementation Requirements Specification

## 1. Overview

PMD (Python Markdown Search) is an on-device hybrid search engine for markdown documents. This specification defines the Python re-implementation requirements, including the class model, project structure, and dependencies.

### Core Capabilities
- **BM25 Full-Text Search** via SQLite FTS5
- **Vector Semantic Search** via sqlite-vec or faiss
- **LLM Re-ranking** via Ollama
- **Hybrid Search Pipeline** with Reciprocal Rank Fusion (RRF)
- **MCP Server Integration** for AI agent access
- **Multiple Output Formats**: JSON, CSV, XML, Markdown, plain text

---

## 2. Project Directory Layout

```
pmd/
├── pyproject.toml              # Project metadata, dependencies, build config
├── README.md                   # Documentation
├── LICENSE                     # License file
├── .gitignore
├── .env.example                # Environment variable template
│
├── src/
│   └── pmd/
│       ├── __init__.py         # Package init, version export
│       ├── __main__.py         # Entry point: python -m pmd
│       ├── py.typed            # PEP 561 marker
│       │
│       ├── cli/                # Command-line interface
│       │   ├── __init__.py
│       │   ├── main.py         # CLI entry point, argument parsing
│       │   ├── commands/       # Command implementations
│       │   │   ├── __init__.py
│       │   │   ├── search.py       # search, vsearch, query commands
│       │   │   ├── collection.py   # collection add/remove/rename/list
│       │   │   ├── document.py     # get, multi-get, ls commands
│       │   │   ├── context.py      # context add/list/rm commands
│       │   │   ├── index.py        # update-all, embed, cleanup
│       │   │   └── status.py       # status command
│       │   └── output.py       # CLI-specific colored output, progress bars
│       │
│       ├── core/               # Core business logic
│       │   ├── __init__.py
│       │   ├── config.py       # Configuration management
│       │   ├── exceptions.py   # Custom exception hierarchy
│       │   └── types.py        # Type definitions (TypedDict, dataclasses)
│       │
│       ├── store/              # Data access layer
│       │   ├── __init__.py
│       │   ├── database.py     # SQLite connection management
│       │   ├── schema.py       # Schema definitions, migrations
│       │   ├── collections.py  # Collection CRUD operations
│       │   ├── documents.py    # Document storage, retrieval
│       │   ├── search.py       # FTS5 and vector search
│       │   ├── embeddings.py   # Vector storage and retrieval
│       │   ├── contexts.py     # Path context management
│       │   ├── cache.py        # Ollama response caching
│       │   └── virtual_paths.py # qmd:// URI handling
│       │
│       ├── llm/                # LLM abstraction layer
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract LLM interface
│       │   ├── ollama.py       # Ollama implementation
│       │   ├── embeddings.py   # Embedding generation
│       │   ├── reranker.py     # Document reranking
│       │   └── query_expansion.py # Query expansion
│       │
│       ├── search/             # Search orchestration
│       │   ├── __init__.py
│       │   ├── pipeline.py     # Hybrid search pipeline
│       │   ├── fusion.py       # Reciprocal Rank Fusion
│       │   ├── chunking.py     # Document chunking for embeddings
│       │   └── scoring.py      # Score normalization, blending
│       │
│       ├── formatters/         # Output formatting
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract formatter interface
│       │   ├── json.py         # JSON formatter
│       │   ├── csv.py          # CSV formatter
│       │   ├── xml.py          # XML formatter
│       │   ├── markdown.py     # Markdown formatter
│       │   ├── files.py        # Plain files list formatter
│       │   └── snippet.py      # Snippet extraction
│       │
│       ├── mcp/                # MCP server
│       │   ├── __init__.py
│       │   ├── server.py       # MCP server setup
│       │   ├── resources.py    # qmd:// resource handlers
│       │   ├── tools.py        # MCP tool implementations
│       │   └── prompts.py      # MCP prompt definitions
│       │
│       └── utils/              # Shared utilities
│           ├── __init__.py
│           ├── hashing.py      # SHA256 content hashing
│           ├── text.py         # Text processing utilities
│           ├── files.py        # File I/O, glob patterns
│           └── levenshtein.py  # Fuzzy matching
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── fixtures/               # Test data files
│   │   └── markdown/           # Sample markdown files
│   │
│   ├── unit/                   # Unit tests
│   │   ├── test_store.py
│   │   ├── test_search.py
│   │   ├── test_formatters.py
│   │   ├── test_llm.py
│   │   └── test_utils.py
│   │
│   ├── integration/            # Integration tests
│   │   ├── test_pipeline.py
│   │   ├── test_cli.py
│   │   └── test_mcp.py
│   │
│   └── e2e/                    # End-to-end tests
│       └── test_full_workflow.py
│
└── scripts/                    # Development scripts
    ├── setup_dev.sh            # Development environment setup
    └── run_tests.sh            # Test runner script
```

---

## 3. Class Model

### 3.1 Core Types (`core/types.py`)

```python
from dataclasses import dataclass, field
from typing import TypedDict, Literal, Optional
from enum import Enum

class SearchSource(Enum):
    FTS = "fts"
    VECTOR = "vec"
    HYBRID = "hybrid"

class OutputFormat(Enum):
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "md"
    FILES = "files"
    CLI = "cli"

@dataclass(frozen=True)
class VirtualPath:
    """Represents a pmd:// URI path."""
    collection_name: str
    path: str

    def __str__(self) -> str:
        return f"pmd://{self.collection_name}/{self.path}"

@dataclass
class Collection:
    """Represents an indexed collection (directory)."""
    id: int
    name: str
    pwd: str  # Base directory path
    glob_pattern: str
    created_at: str
    updated_at: str

@dataclass
class DocumentResult:
    """Represents a retrieved document."""
    filepath: str
    display_path: str
    title: str
    context: Optional[str]
    hash: str
    collection_id: int
    modified_at: str
    body_length: int
    body: Optional[str] = None

@dataclass
class SearchResult(DocumentResult):
    """Extends DocumentResult with search-specific fields."""
    score: float = 0.0
    source: SearchSource = SearchSource.FTS
    chunk_pos: Optional[int] = None
    snippet: Optional[str] = None

@dataclass
class RankedResult:
    """Result after RRF fusion and reranking."""
    file: str
    display_path: str
    title: str
    body: str
    score: float
    fts_score: Optional[float] = None
    vec_score: Optional[float] = None
    rerank_score: Optional[float] = None

@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embedding: list[float]
    model: str

@dataclass
class RerankDocumentResult:
    """Single document reranking result."""
    file: str
    relevant: bool
    confidence: float
    score: float
    raw_token: str
    logprob: Optional[float]

@dataclass
class RerankResult:
    """Complete reranking result."""
    results: list[RerankDocumentResult]
    model: str

@dataclass
class PathContext:
    """Context description for a path prefix."""
    id: int
    collection_id: int
    path_prefix: str
    context: str
    created_at: str

@dataclass
class Chunk:
    """Document chunk for embedding."""
    text: str
    pos: int  # Character position in original document

@dataclass
class SnippetResult:
    """Extracted snippet with match position."""
    text: str
    match_start: int
    match_end: int

class DocumentNotFound(TypedDict):
    """Error result when document not found."""
    error: str
    suggestions: list[str]

@dataclass
class IndexStatus:
    """Status information for the index."""
    collections: list[Collection]
    total_documents: int
    embedded_documents: int
    index_size_bytes: int
    cache_entries: int
    ollama_available: bool
    models_available: dict[str, bool]
```

### 3.2 Configuration (`core/config.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class OllamaConfig:
    """Ollama service configuration."""
    base_url: str = "http://localhost:11434"
    embedding_model: str = "embeddinggemma"
    expansion_model: str = "qwen3:0.6b"
    reranker_model: str = "ExpedientFalcon/Qwen3-Reranker-0.6B-GGUF:Q8_0"
    timeout: float = 120.0

@dataclass
class SearchConfig:
    """Search pipeline configuration."""
    default_limit: int = 5
    fts_weight: float = 1.0
    vec_weight: float = 1.0
    rrf_k: int = 60
    top_rank_bonus: float = 0.05
    expansion_weight: float = 0.5
    rerank_candidates: int = 30

@dataclass
class ChunkConfig:
    """Document chunking configuration."""
    max_bytes: int = 6 * 1024  # ~2000 tokens
    min_chunk_size: int = 100

@dataclass
class Config:
    """Main application configuration."""
    db_path: Path = field(default_factory=lambda: _default_db_path())
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()

        if url := os.environ.get("OLLAMA_URL"):
            config.ollama.base_url = url

        if path := os.environ.get("INDEX_PATH"):
            config.db_path = Path(path)

        return config

def _default_db_path() -> Path:
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_dir / "pmd" / "index.db"
```

### 3.3 Custom Exceptions (`core/exceptions.py`)

```python
class PMDError(Exception):
    """Base exception for all PMD errors."""
    pass

class DatabaseError(PMDError):
    """Database operation failed."""
    pass

class CollectionError(PMDError):
    """Collection operation failed."""
    pass

class CollectionNotFoundError(CollectionError):
    """Collection does not exist."""
    pass

class CollectionExistsError(CollectionError):
    """Collection already exists."""
    pass

class DocumentError(PMDError):
    """Document operation failed."""
    pass

class DocumentNotFoundError(DocumentError):
    """Document does not exist."""
    def __init__(self, path: str, suggestions: list[str] = None):
        self.path = path
        self.suggestions = suggestions or []
        super().__init__(f"Document not found: {path}")

class LLMError(PMDError):
    """LLM operation failed."""
    pass

class OllamaConnectionError(LLMError):
    """Cannot connect to Ollama service."""
    pass

class ModelNotFoundError(LLMError):
    """Required model not available."""
    pass

class SearchError(PMDError):
    """Search operation failed."""
    pass

class EmbeddingError(PMDError):
    """Embedding generation failed."""
    pass

class FormatError(PMDError):
    """Output formatting failed."""
    pass

class VirtualPathError(PMDError):
    """Virtual path parsing/resolution failed."""
    pass
```

### 3.4 Store Layer (`store/`)

```python
# store/database.py
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator

class Database:
    """SQLite database connection manager."""

    def __init__(self, path: Path):
        self.path = path
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Initialize database connection and load extensions."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._load_vec_extension()
        self._enable_fts5()

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database transactions."""
        cursor = self._connection.cursor()
        try:
            yield cursor
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        finally:
            cursor.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL query."""
        return self._connection.execute(sql, params)

    def _load_vec_extension(self) -> None:
        """Load sqlite-vec extension."""
        # Platform-specific loading
        pass

    def _enable_fts5(self) -> None:
        """Enable FTS5 extension."""
        pass


# store/schema.py
SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Content-addressable storage
CREATE TABLE IF NOT EXISTS content (
    hash TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Collections (indexed directories)
CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    pwd TEXT NOT NULL,
    glob_pattern TEXT NOT NULL DEFAULT '**/*.md',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Documents (file-to-content mappings)
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id),
    path TEXT NOT NULL,
    title TEXT NOT NULL,
    hash TEXT NOT NULL REFERENCES content(hash),
    active INTEGER NOT NULL DEFAULT 1,
    modified_at TEXT NOT NULL,
    UNIQUE(collection_id, path)
);

-- Full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path, body,
    content='',
    tokenize='porter unicode61'
);

-- Vector embeddings metadata
CREATE TABLE IF NOT EXISTS content_vectors (
    hash TEXT NOT NULL,
    seq INTEGER NOT NULL,
    pos INTEGER NOT NULL,
    model TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (hash, seq)
);

-- Vector storage (sqlite-vec)
-- CREATE VIRTUAL TABLE vectors_vec USING vec0(
--     hash_seq TEXT PRIMARY KEY,
--     embedding float[768]
-- );

-- Context descriptions
CREATE TABLE IF NOT EXISTS path_contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id),
    path_prefix TEXT NOT NULL,
    context TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(collection_id, path_prefix)
);

-- Ollama API response cache
CREATE TABLE IF NOT EXISTS ollama_cache (
    hash TEXT PRIMARY KEY,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_content_vectors_hash ON content_vectors(hash);
"""


# store/collections.py
from dataclasses import dataclass
from typing import Protocol

class CollectionRepository(Protocol):
    """Interface for collection operations."""

    def list_all(self) -> list[Collection]: ...
    def get_by_name(self, name: str) -> Collection | None: ...
    def get_by_id(self, id: int) -> Collection | None: ...
    def create(self, name: str, pwd: str, glob_pattern: str) -> Collection: ...
    def remove(self, collection_id: int) -> tuple[int, int]: ...  # (deleted_docs, cleaned_hashes)
    def rename(self, collection_id: int, new_name: str) -> None: ...


# store/search.py
from abc import ABC, abstractmethod

class SearchRepository(ABC):
    """Interface for search operations."""

    @abstractmethod
    def search_fts(
        self,
        query: str,
        limit: int = 5,
        collection_id: int | None = None,
        min_score: float = 0.0
    ) -> list[SearchResult]:
        """Perform BM25 full-text search."""
        pass

    @abstractmethod
    async def search_vec(
        self,
        query: str,
        model: str,
        limit: int = 5,
        collection_id: int | None = None,
        min_score: float = 0.0
    ) -> list[SearchResult]:
        """Perform vector similarity search."""
        pass
```

### 3.5 LLM Layer (`llm/`)

```python
# llm/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from ..core.types import EmbeddingResult, RerankResult

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def embed(
        self,
        text: str,
        model: str,
        is_query: bool = False
    ) -> EmbeddingResult | None:
        """Generate embeddings for text."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str | None:
        """Generate text completion."""
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[dict],
        model: str
    ) -> RerankResult:
        """Rerank documents by relevance to query."""
        pass

    @abstractmethod
    async def model_exists(self, model: str) -> bool:
        """Check if model is available."""
        pass

    @abstractmethod
    async def pull_model(
        self,
        model: str,
        on_progress: Optional[callable] = None
    ) -> bool:
        """Download/pull a model."""
        pass


# llm/ollama.py
import httpx
from .base import LLMProvider

class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=120.0)

    async def embed(
        self,
        text: str,
        model: str,
        is_query: bool = False
    ) -> EmbeddingResult | None:
        """Generate embeddings using Ollama."""
        # Format prompt based on model requirements
        if is_query:
            prompt = f"task: search result | query: {text}"
        else:
            prompt = f"title: | text: {text}"

        response = await self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": prompt}
        )

        if response.status_code == 200:
            data = response.json()
            return EmbeddingResult(
                embedding=data["embeddings"][0],
                model=model
            )
        return None

    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str | None:
        """Generate text using Ollama."""
        response = await self._client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                },
                "stream": False
            }
        )

        if response.status_code == 200:
            return response.json()["response"]
        return None

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        model: str
    ) -> RerankResult:
        """Rerank documents using Ollama with logprobs."""
        results = []

        for doc in documents:
            # Use chat format with logprobs
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self._reranker_system_prompt()},
                        {"role": "user", "content": f"Query: {query}\nDocument: {doc['body'][:1000]}"},
                        {"role": "assistant", "content": ""}
                    ],
                    "options": {"logprobs": True, "num_predict": 1}
                }
            )

            # Parse logprobs to extract Yes/No confidence
            result = self._parse_rerank_response(doc["file"], response.json())
            results.append(result)

        return RerankResult(results=results, model=model)

    def _reranker_system_prompt(self) -> str:
        return (
            "You are a relevance judge. Given a query and a document, "
            "respond with ONLY 'Yes' if the document is relevant to the query, "
            "or 'No' if it is not relevant."
        )

    def _parse_rerank_response(self, file: str, response: dict) -> RerankDocumentResult:
        # Extract logprobs and calculate confidence
        pass


# llm/query_expansion.py
class QueryExpander:
    """Expands queries into semantic variations."""

    def __init__(self, provider: LLMProvider, model: str):
        self.provider = provider
        self.model = model

    async def expand(self, query: str, num_variations: int = 2) -> list[str]:
        """Generate query variations for better recall."""
        prompt = f"""Generate {num_variations} alternative phrasings for this search query.
Return ONLY the variations, one per line, no numbering or explanation.

Query: {query}

Variations:"""

        response = await self.provider.generate(
            prompt,
            self.model,
            max_tokens=100,
            temperature=0.7
        )

        if response:
            variations = [v.strip() for v in response.strip().split('\n') if v.strip()]
            return variations[:num_variations]
        return []
```

### 3.6 Search Pipeline (`search/`)

```python
# search/pipeline.py
from dataclasses import dataclass
from typing import Optional
import asyncio

from ..core.types import SearchResult, RankedResult
from ..store.search import SearchRepository
from ..llm.base import LLMProvider
from .fusion import reciprocal_rank_fusion
from .scoring import blend_scores

@dataclass
class SearchPipelineConfig:
    fts_weight: float = 1.0
    vec_weight: float = 1.0
    rrf_k: int = 60
    rerank_candidates: int = 30
    enable_query_expansion: bool = True
    enable_reranking: bool = True

class HybridSearchPipeline:
    """Orchestrates hybrid search with FTS, vector, and reranking."""

    def __init__(
        self,
        search_repo: SearchRepository,
        llm_provider: LLMProvider,
        config: SearchPipelineConfig
    ):
        self.search_repo = search_repo
        self.llm = llm_provider
        self.config = config

    async def search(
        self,
        query: str,
        limit: int = 5,
        collection_id: Optional[int] = None,
        min_score: float = 0.0
    ) -> list[RankedResult]:
        """Execute full hybrid search pipeline."""

        # Step 1: Query expansion
        queries = [query]
        if self.config.enable_query_expansion:
            expansions = await self._expand_query(query)
            queries.extend(expansions)

        # Step 2: Parallel FTS and vector search for all queries
        all_results = await self._parallel_search(
            queries, limit * 3, collection_id
        )

        # Step 3: Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            all_results,
            k=self.config.rrf_k,
            original_query_weight=2.0
        )

        # Take top candidates for reranking
        candidates = fused[:self.config.rerank_candidates]

        # Step 4: LLM Reranking
        if self.config.enable_reranking and candidates:
            reranked = await self._rerank(query, candidates)
            final = blend_scores(candidates, reranked)
        else:
            final = candidates

        # Step 5: Filter and limit
        final = [r for r in final if r.score >= min_score]
        return final[:limit]

    async def _expand_query(self, query: str) -> list[str]:
        """Generate query variations."""
        # Implementation
        pass

    async def _parallel_search(
        self,
        queries: list[str],
        limit: int,
        collection_id: Optional[int]
    ) -> list[list[SearchResult]]:
        """Run FTS and vector search in parallel for all queries."""
        tasks = []

        for query in queries:
            tasks.append(self.search_repo.search_fts(query, limit, collection_id))
            tasks.append(self.search_repo.search_vec(query, limit, collection_id))

        results = await asyncio.gather(*tasks)
        return results

    async def _rerank(
        self,
        query: str,
        candidates: list[RankedResult]
    ) -> list[RerankDocumentResult]:
        """Rerank candidates using LLM."""
        docs = [{"file": c.file, "body": c.body} for c in candidates]
        result = await self.llm.rerank(query, docs, self.config.reranker_model)
        return result.results


# search/fusion.py
from typing import TypeVar
from collections import defaultdict

T = TypeVar('T')

def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    original_query_weight: float = 2.0,
    weights: list[float] | None = None
) -> list[RankedResult]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF Score = Σ(weight / (k + rank + 1))

    Args:
        result_lists: List of ranked result lists
        k: Smoothing constant (default 60)
        original_query_weight: Extra weight for original query results
        weights: Optional weights per result list

    Returns:
        Fused and re-ranked results
    """
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, SearchResult] = {}

    if weights is None:
        # Default: first two lists (original query) get higher weight
        weights = [original_query_weight, original_query_weight] + [1.0] * (len(result_lists) - 2)

    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0

        for rank, result in enumerate(results):
            # RRF formula
            rrf_score = weight / (k + rank + 1)

            # Top-rank bonus
            if rank == 0:
                rrf_score += 0.05
            elif rank <= 2:
                rrf_score += 0.02

            scores[result.filepath] += rrf_score

            # Keep the result with highest individual score
            if result.filepath not in docs or result.score > docs[result.filepath].score:
                docs[result.filepath] = result

    # Sort by fused score
    sorted_files = sorted(scores.keys(), key=lambda f: scores[f], reverse=True)

    return [
        RankedResult(
            file=f,
            display_path=docs[f].display_path,
            title=docs[f].title,
            body=docs[f].body or "",
            score=scores[f]
        )
        for f in sorted_files
    ]


# search/scoring.py
def blend_scores(
    rrf_results: list[RankedResult],
    rerank_results: list[RerankDocumentResult]
) -> list[RankedResult]:
    """
    Blend RRF scores with reranker scores using position-aware weighting.

    Position-aware blending:
    - Rank 1-3:   75% RRF + 25% reranker (trust initial ranking)
    - Rank 4-10:  60% RRF + 40% reranker
    - Rank 11+:   40% RRF + 60% reranker (trust reranker more for borderline)
    """
    rerank_map = {r.file: r.score for r in rerank_results}

    blended = []
    for rank, result in enumerate(rrf_results):
        rrf_score = result.score
        rerank_score = rerank_map.get(result.file, 0.5)

        # Position-aware weight
        if rank < 3:
            rrf_weight = 0.75
        elif rank < 10:
            rrf_weight = 0.60
        else:
            rrf_weight = 0.40

        final_score = rrf_weight * rrf_score + (1 - rrf_weight) * rerank_score

        blended.append(RankedResult(
            file=result.file,
            display_path=result.display_path,
            title=result.title,
            body=result.body,
            score=final_score,
            fts_score=result.fts_score,
            vec_score=result.vec_score,
            rerank_score=rerank_score
        ))

    # Re-sort by blended score
    blended.sort(key=lambda r: r.score, reverse=True)
    return blended
```

### 3.7 Formatters (`formatters/`)

```python
# formatters/base.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')

class Formatter(ABC, Generic[T]):
    """Abstract base class for output formatters."""

    @abstractmethod
    def format(self, data: T) -> str:
        """Format data to string output."""
        pass

    @abstractmethod
    def format_many(self, data: list[T]) -> str:
        """Format multiple items."""
        pass


# formatters/json.py
import json
from .base import Formatter

class JSONFormatter(Formatter[SearchResult]):
    """JSON output formatter."""

    def __init__(self, include_body: bool = False, indent: int = 2):
        self.include_body = include_body
        self.indent = indent

    def format(self, result: SearchResult) -> str:
        data = self._to_dict(result)
        return json.dumps(data, indent=self.indent)

    def format_many(self, results: list[SearchResult]) -> str:
        data = [self._to_dict(r) for r in results]
        return json.dumps(data, indent=self.indent)

    def _to_dict(self, result: SearchResult) -> dict:
        d = {
            "score": result.score,
            "file": result.filepath,
            "displayPath": result.display_path,
            "title": result.title,
            "context": result.context
        }
        if self.include_body and result.body:
            d["body"] = result.body
        if result.snippet:
            d["snippet"] = result.snippet
        return d


# formatters/csv.py
import csv
import io
from .base import Formatter

class CSVFormatter(Formatter[SearchResult]):
    """CSV output formatter."""

    def format_many(self, results: list[SearchResult]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["score", "file", "title", "context", "snippet"])

        for r in results:
            writer.writerow([
                f"{r.score:.4f}",
                r.filepath,
                r.title,
                r.context or "",
                r.snippet or ""
            ])

        return output.getvalue()


# formatters/snippet.py
import re

def extract_snippet(
    body: str,
    query: str,
    max_length: int = 200,
    chunk_pos: int | None = None
) -> SnippetResult | None:
    """
    Extract a relevant snippet from document body.

    Prioritizes sections containing query terms, falling back to
    chunk position or document start.
    """
    if not body:
        return None

    # Try to find query terms in body
    terms = query.lower().split()
    body_lower = body.lower()

    best_pos = 0
    for term in terms:
        pos = body_lower.find(term)
        if pos != -1:
            best_pos = max(0, pos - 50)  # Start 50 chars before match
            break

    # Use chunk position if available and no term match
    if best_pos == 0 and chunk_pos is not None:
        best_pos = chunk_pos

    # Extract snippet
    end_pos = min(best_pos + max_length, len(body))
    snippet = body[best_pos:end_pos]

    # Clean up: don't start/end mid-word
    if best_pos > 0:
        snippet = "..." + snippet.lstrip()
    if end_pos < len(body):
        snippet = snippet.rstrip() + "..."

    return SnippetResult(
        text=snippet,
        match_start=best_pos,
        match_end=end_pos
    )
```

### 3.8 MCP Server (`mcp/`)

```python
# mcp/server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, Resource, TextContent

from .tools import register_tools
from .resources import register_resources
from .prompts import register_prompts

class PMDMCPServer:
    """MCP server exposing PMD functionality."""

    def __init__(self, db_path: str):
        self.server = Server("pmd")
        self.db_path = db_path

        register_tools(self.server, db_path)
        register_resources(self.server, db_path)
        register_prompts(self.server)

    async def run(self):
        """Run the MCP server on stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


# mcp/tools.py
from mcp.server import Server
from mcp.types import Tool

def register_tools(server: Server, db_path: str):
    """Register all PMD tools with the MCP server."""

    @server.tool()
    async def search(
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
        collection: str | None = None
    ) -> str:
        """BM25 keyword search across indexed documents."""
        # Implementation
        pass

    @server.tool()
    async def vsearch(
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
        collection: str | None = None
    ) -> str:
        """Vector semantic search using embeddings."""
        # Implementation
        pass

    @server.tool()
    async def query(
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
        collection: str | None = None
    ) -> str:
        """Hybrid search with query expansion and reranking."""
        # Implementation
        pass

    @server.tool()
    async def get(
        file: str,
        from_line: int | None = None,
        max_lines: int | None = None
    ) -> str:
        """Retrieve a single document by path."""
        # Implementation
        pass

    @server.tool()
    async def multi_get(
        pattern: str,
        max_lines: int | None = None,
        max_bytes: int = 10240
    ) -> str:
        """Retrieve multiple documents by glob pattern or CSV list."""
        # Implementation
        pass

    @server.tool()
    async def status() -> str:
        """Get index health and collection information."""
        # Implementation
        pass


# mcp/resources.py
from mcp.server import Server
from mcp.types import Resource, ResourceTemplate

def register_resources(server: Server, db_path: str):
    """Register pmd:// URI resources."""

    @server.resource("pmd://{path}")
    async def get_document(path: str) -> str:
        """Retrieve document content by pmd:// URI."""
        # Implementation
        pass
```

### 3.9 CLI (`cli/`)

```python
# cli/main.py
import argparse
import asyncio
import sys
from typing import NoReturn

from ..core.config import Config
from .commands import search, collection, document, context, index, status

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="pmd",
        description="Python Markdown Search - Hybrid search for markdown documents"
    )
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search commands
    search_parser = subparsers.add_parser("search", help="BM25 keyword search")
    search.add_arguments(search_parser)

    vsearch_parser = subparsers.add_parser("vsearch", help="Vector semantic search")
    search.add_arguments(vsearch_parser)

    query_parser = subparsers.add_parser("query", help="Hybrid search with reranking")
    search.add_arguments(query_parser)

    # Collection commands
    coll_parser = subparsers.add_parser("collection", help="Manage collections")
    collection.add_subcommands(coll_parser)

    # Document commands
    get_parser = subparsers.add_parser("get", help="Get single document")
    document.add_get_arguments(get_parser)

    multi_parser = subparsers.add_parser("multi-get", help="Get multiple documents")
    document.add_multi_get_arguments(multi_parser)

    ls_parser = subparsers.add_parser("ls", help="List collections or files")
    document.add_ls_arguments(ls_parser)

    # Context commands
    ctx_parser = subparsers.add_parser("context", help="Manage path contexts")
    context.add_subcommands(ctx_parser)

    # Index commands
    subparsers.add_parser("update-all", help="Update all collections")
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
    embed_parser.add_argument("-f", "--force", action="store_true")
    subparsers.add_parser("cleanup", help="Clean cache and orphaned data")

    # Status
    subparsers.add_parser("status", help="Show index status")

    # MCP server
    subparsers.add_parser("mcp", help="Start MCP server")

    return parser

def main() -> NoReturn:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    config = Config.from_env()

    # Route to appropriate command handler
    handlers = {
        "search": search.handle_search,
        "vsearch": search.handle_vsearch,
        "query": search.handle_query,
        "collection": collection.handle,
        "get": document.handle_get,
        "multi-get": document.handle_multi_get,
        "ls": document.handle_ls,
        "context": context.handle,
        "update-all": index.handle_update_all,
        "embed": index.handle_embed,
        "cleanup": index.handle_cleanup,
        "status": status.handle,
        "mcp": lambda args, config: asyncio.run(run_mcp(config))
    }

    handler = handlers.get(args.command)
    if handler:
        try:
            result = handler(args, config)
            if asyncio.iscoroutine(result):
                asyncio.run(result)
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## 4. Dependencies

### 4.1 Required Dependencies

```toml
[project]
name = "pmd"
version = "1.0.0"
description = "Python Markdown Search - Hybrid search for markdown documents"
requires-python = ">=3.11"

dependencies = [
    # Database
    "sqlite-vec>=0.1.0",          # Vector similarity search for SQLite

    # HTTP Client (for Ollama)
    "httpx>=0.27.0",              # Async HTTP client

    # MCP Server
    "mcp>=1.0.0",                 # Model Context Protocol SDK

    # CLI
    "rich>=13.0.0",               # Rich terminal output, progress bars
    "click>=8.1.0",               # CLI framework (alternative to argparse)

    # Utilities
    "pydantic>=2.0.0",            # Data validation (optional, for strict typing)
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
]
```

### 4.2 External Services

| Service | Purpose | Default URL |
|---------|---------|-------------|
| Ollama | LLM inference | `http://localhost:11434` |

### 4.3 Required Ollama Models

| Model | Size | Purpose |
|-------|------|---------|
| `embeddinggemma` | ~1.6GB | Vector embeddings (768 dimensions) |
| `qwen3:0.6b` | ~400MB | Query expansion |
| `ExpedientFalcon/Qwen3-Reranker-0.6B-GGUF:Q8_0` | ~640MB | Document reranking |

---

## 5. Algorithm Specifications

### 5.1 Hybrid Search Pipeline

```
User Query
    │
    ├─→ Query Expansion (LLM, qwen3:0.6b)
    │   • Original query weighted ×2
    │   • Generate 2 semantic variations
    │
    ├─→ Parallel Search (for each query variant)
    │   ├─→ FTS5 BM25 Search
    │   │   • tokenize='porter unicode61'
    │   │   • Score normalized: abs(score) / max_score
    │   │
    │   └─→ Vector Search (sqlite-vec)
    │       • Cosine distance → 1 / (1 + distance)
    │       • Top-K with best distance per document
    │
    ├─→ Reciprocal Rank Fusion (RRF)
    │   • k = 60 (smoothing constant)
    │   • Score = Σ(weight / (k + rank + 1))
    │   • Top-rank bonus: +0.05 for #1, +0.02 for #2-3
    │   • Select top 30 candidates
    │
    ├─→ LLM Re-ranking (qwen3-reranker)
    │   • Yes/No classification with logprobs
    │   • Yes: score = 0.5 + 0.5 × confidence
    │   • No:  score = 0.5 × (1 - confidence)
    │
    └─→ Position-Aware Blending
        • Rank 1-3:   75% RRF + 25% reranker
        • Rank 4-10:  60% RRF + 40% reranker
        • Rank 11+:   40% RRF + 60% reranker
```

### 5.2 Document Chunking

```python
CHUNK_MAX_BYTES = 6 * 1024  # ~2000 tokens

Algorithm:
1. If len(content) <= max_bytes: return [(content, 0)]
2. Split preferring: \n\n → sentence end → \n → space
3. Each chunk stored with:
   - hash: SHA256(full_document)
   - seq: 0, 1, 2, ...
   - pos: character offset in original
```

### 5.3 Content-Addressable Storage

```
documents table:
  └── (collection_id, path) → hash reference

content table:
  └── hash (SHA256) → actual content

Benefits:
  • Deduplication of identical content
  • Efficient change detection
  • Clean garbage collection
```

### 5.4 Virtual Path System

```
Format: pmd://{collection-name}/{relative-path}

Example:
  Filesystem:   /home/user/notes/2025/meeting.md
  Virtual:      pmd://notes/2025/meeting.md

Bidirectional conversion supported.
```

### 5.5 Hierarchical Context Inheritance

```
Contexts are matched by longest path prefix:

  /                    → "All documents"
  /meetings            → "Meeting notes"
  /meetings/2025       → "2025 meetings"
  /meetings/2025/q1    → "Q1 2025 meetings"

Query: /meetings/2025/q1/jan.md
Match: /meetings/2025/q1 (longest)
```

---

## 6. Testing Strategy

### 6.1 Unit Tests
- Store layer: CRUD operations, query building
- Search: FTS5 queries, vector search, RRF fusion
- Formatters: Each output format
- LLM: Mock Ollama responses
- Utils: Hashing, text processing, Levenshtein

### 6.2 Integration Tests
- Full search pipeline with test fixtures
- CLI command execution
- MCP tool invocations

### 6.3 E2E Tests
- Complete workflow: index → embed → search → retrieve
- Multiple collections
- Various output formats

### 6.4 Test Fixtures
- Sample markdown files with known content
- Pre-computed embeddings for deterministic tests
- Mock Ollama server for CI/CD

---

## 7. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `XDG_CACHE_HOME` | `~/.cache` | Cache directory base |
| `INDEX_PATH` | `$XDG_CACHE_HOME/pmd/index.db` | Database path override |
| `NO_COLOR` | (unset) | Disable colored output |

---

## 8. Migration Path

1. **Phase 1**: Core infrastructure
   - Database schema and migrations
   - Basic collection management
   - Document indexing (no embeddings)

2. **Phase 2**: Search capabilities
   - FTS5 search implementation
   - Vector search with sqlite-vec
   - RRF fusion

3. **Phase 3**: LLM integration
   - Ollama client
   - Embedding generation
   - Query expansion
   - Reranking

4. **Phase 4**: CLI and formatting
   - Full CLI implementation
   - All output formats
   - Progress indicators

5. **Phase 5**: MCP server
   - Tool implementations
   - Resource handlers
   - Integration testing

---

## 9. API Compatibility Notes

The Python implementation should maintain behavioral compatibility with the TypeScript version (qmd):

- Same database schema (cross-compatible databases)
- Virtual path format uses `pmd://` (vs `qmd://` in TypeScript version)
- Same CLI command structure and options (but using `pmd` command)
- Same MCP tool signatures
- Same output format structures
