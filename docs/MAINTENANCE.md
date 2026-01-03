# Maintenance Guide

This document provides instructions for maintainers and developers working on PMD (Python Markdown Search).

---

## Table of Contents

1. [Installation](#1-installation)
2. [Running Tests](#2-running-tests)
3. [Instrumentation and Tracing](#3-instrumentation-and-tracing)
4. [Code Style Guidelines](#4-code-style-guidelines)
5. [Code Quality Standards](#5-code-quality-standards)

---

## 1. Installation

### Prerequisites

- Python 3.11 or higher
- macOS with Apple Silicon (for MLX local inference)
- uv package manager only

### Development Installation

```bash
# Clone the repository
git clone <repository-url>
cd qmd

# Create virtual environment and install with dev dependencies
uv sync --all-extras
```

### Dependencies

**Core Dependencies:**
- `sqlite-vec` - Vector similarity search extension
- `httpx` - Async HTTP client
- `mcp` - Model Context Protocol
- `rich` - Rich text formatting
- `click` - CLI framework
- `pydantic` - Data validation
- `mlx-lm` - Apple Silicon LLM inference
- `mlx-embeddings` - Apple Silicon embeddings
- `loguru` - Logging

**Tracing Dependencies:**
- `opentelemetry-sdk` - OpenTelemetry SDK
- `opentelemetry-exporter-otlp-proto-http` - OTLP HTTP exporter
- `arize-phoenix` - LLM observability

**Dev Dependencies:**
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting
- `mypy` - Static type checking
- `ruff` - Linting and formatting
- `respx` - HTTP mocking

### Verifying Installation

```bash
# Check CLI is available
pmd --version

# Run a quick test
uv run pytest tests/unit -x -q
```

---

## 2. Running Tests

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── fixtures/                # Test data
│   └── test_corpus/         # DO NOT MODIFY
├── unit/                    # Unit tests
│   ├── core/               # Core module tests
│   ├── llm/                # LLM provider tests
│   ├── search/             # Search algorithm tests
│   ├── services/           # Service layer tests
│   ├── sources/            # Document source tests
│   └── store/              # Repository tests
└── integration/             # Integration tests
```

### Running All Tests

```bash
# Run all tests with coverage
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=src/pmd --cov-report=term-missing
```

### Running Specific Tests

```bash
# Run unit tests only
uv run pytest tests/unit

# Run integration tests only
uv run pytest tests/integration

# Run a specific test file
uv run pytest tests/unit/store/test_documents.py

# Run a specific test function
uv run pytest tests/unit/store/test_documents.py::test_add_document

# Run tests matching a pattern
uv run pytest -k "embedding"

# Run tests and stop on first failure
uv run pytest -x
```

### Coverage Reports

Coverage is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=src/pmd --cov-report=term-missing --cov-report=html"
```

After running tests:
- Terminal report shows missing lines
- HTML report generated in `htmlcov/index.html`

### Test Fixtures

Common fixtures defined in `tests/conftest.py`:

| Fixture | Description |
|---------|-------------|
| `test_db_path` | Temporary database path |
| `fixtures_dir` | Path to fixtures directory |
| `db` | Connected Database instance |
| `collection_repo` | CollectionRepository instance |
| `document_repo` | DocumentRepository instance |
| `embedding_repo` | EmbeddingRepository instance |
| `fts_repo` | FTS5SearchRepository instance |
| `sample_collection` | Pre-created test collection |
| `sample_document` | Pre-created test document |

### Async Tests

Tests are automatically detected as async when using `async def`:

```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_function()
    assert result is not None
```

The `asyncio_mode = "auto"` setting handles this automatically.

---

## 3. Instrumentation and Tracing

PMD includes OpenTelemetry instrumentation for observability, exporting traces to Arize Phoenix.

### Enabling Tracing

**Via CLI:**
```bash
pmd --phoenix-tracing search "query"
pmd --phoenix-tracing --phoenix-endpoint http://localhost:6006/v1/traces search "query"
```

**Via Configuration:**
```toml
# pmd.toml
[tracing]
enabled = true
phoenix_endpoint = "http://localhost:6006/v1/traces"
service_name = "pmd"
sample_rate = 1.0
batch_export = true
```

**Via Environment:**
```bash
export PHOENIX_TRACING=1
export PHOENIX_ENDPOINT="http://localhost:6006/v1/traces"
export PHOENIX_SAMPLE_RATE=0.5
```

### Running Phoenix

Phoenix provides a UI for viewing traces:

```bash
# Install Phoenix
pip install arize-phoenix

# Start Phoenix server
phoenix serve

# Open http://localhost:6006 in browser
```

### Traced Operations

The following operations are instrumented:

| Span Name | Description |
|-----------|-------------|
| `mlx_lm.generate` | Text generation via MLX |
| `mlx_embedding.embed` | Embedding generation via MLX |
| `pmd.search` | High-level search operation |
| `pmd.index` | Indexing operation |
| `pmd.rerank` | Reranking operation |

### Span Attributes

MLX generation spans include:
- `mlx.model_id` - Model identifier
- `mlx.device` - Device (always "mps" for Metal)
- `mlx.latency_ms` - Operation latency
- `gen_ai.prompt.length` - Prompt length
- `gen_ai.request.max_tokens` - Max tokens requested
- `gen_ai.request.temperature` - Temperature setting
- `gen_ai.response.length` - Response length
- `input.value` - Input prompt (for Phoenix)
- `output.value` - Output text (for Phoenix)

MLX embedding spans include:
- `mlx.model_id` - Model identifier
- `embedding.input_length` - Input text length
- `embedding.is_query` - Query vs document embedding
- `embedding.dimension` - Embedding dimension
- `embedding.pooling_used` - Pooling strategy

### Programmatic Usage

```python
from pmd.core.instrumentation import (
    configure_phoenix_tracing,
    traced_mlx_generate,
    traced_mlx_embed,
    traced_request,
    shutdown_tracing,
    TracingConfig,
)

# Configure tracing
config = TracingConfig(
    enabled=True,
    phoenix_endpoint="http://localhost:6006/v1/traces",
    sample_rate=1.0,
)
tracer = configure_phoenix_tracing(config)

# Trace a generation
with traced_mlx_generate("model-id", prompt, 256, 0.7) as result_meta:
    output = mlx_lm.generate(...)
    result_meta["output"] = output
    result_meta["output_length"] = len(output)

# Trace an embedding
with traced_mlx_embed("model-id", text, is_query=True) as result_meta:
    embedding = generate_embedding(text)
    result_meta["embedding_dim"] = len(embedding)

# Trace a high-level operation
with traced_request("hybrid_search", attributes={"query": query}) as span:
    results = await search(query)

# Cleanup
shutdown_tracing()
```

---

## 4. Code Style Guidelines

### Formatting

PMD uses **Ruff** for linting and formatting:

```bash
# Check formatting
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/

# Format code
uv run ruff format src/
```

**Configuration** (`pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
target-version = "py311"
```

### Import Organization

Imports should be organized in three groups:
1. Standard library
2. Third-party packages
3. Local imports

```python
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from pmd.core.types import SearchResult
from pmd.store.database import Database
```

### Type Annotations

All public functions and methods should have type annotations:

```python
def search(
    query: str,
    limit: int = 10,
    collection: str | None = None,
) -> list[SearchResult]:
    """Search for documents matching query."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def embed_document(self, hash_value: str, content: str) -> int:
    """Embed a document and store the embeddings.

    Args:
        hash_value: SHA256 hash of the document content.
        content: Document text to embed.

    Returns:
        Number of chunks embedded.

    Raises:
        EmbeddingError: If embedding generation fails.
    """
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `DocumentRepository` |
| Functions | snake_case | `embed_document` |
| Variables | snake_case | `collection_name` |
| Constants | UPPER_SNAKE_CASE | `SCHEMA_VERSION` |
| Private | Leading underscore | `_internal_method` |
| Protocols | PascalCase + Protocol suffix | `DocumentSource` |

### Async Conventions

- Prefix async functions with `async def`
- Use `await` for all async calls
- Prefer `async with` for async context managers
- Use `asyncio.gather()` for parallel operations

```python
async def fetch_all(urls: list[str]) -> list[FetchResult]:
    """Fetch multiple URLs concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        return await asyncio.gather(*tasks)
```

---

## 5. Code Quality Standards

### Type Checking

PMD uses **mypy** with strict mode:

```bash
# Run type checking
uv run mypy src/pmd
```

**Configuration** (`pyproject.toml`):
```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_configs = true
```

### Test Coverage

Target coverage: **80%+** for new code.

Excluded from coverage:
- `pragma: no cover` comments
- `if TYPE_CHECKING:` blocks
- `if __name__ == "__main__":` blocks
- `raise NotImplementedError`

### Error Handling

- Use custom exceptions from `pmd.core.exceptions`
- Catch specific exceptions, not bare `except:`
- Log errors with appropriate levels
- Provide actionable error messages

```python
from pmd.core.exceptions import DocumentNotFoundError

try:
    document = repo.get(collection_id, path)
    if document is None:
        raise DocumentNotFoundError(path, suggestions=["similar.md"])
except DatabaseError as e:
    logger.error(f"Database error fetching {path}: {e}")
    raise
```

### Logging

Use **loguru** for logging:

```python
from loguru import logger

logger.debug("Processing document: {}", path)
logger.info("Indexed {} documents", count)
logger.warning("Model not found, using fallback")
logger.error("Failed to connect to database: {}", error)
```

Log levels:
- `DEBUG`: Detailed diagnostic information
- `INFO`: General operational events
- `WARNING`: Unexpected but recoverable situations
- `ERROR`: Errors that prevent operation completion

### Security Considerations

- Never log sensitive data (API keys, passwords)
- Validate all user input
- Use parameterized SQL queries (prepared statements)
- Sanitize file paths to prevent directory traversal

### Code Review Checklist

Before submitting code:

- [ ] All tests pass (`uv run pytest`)
- [ ] Type checks pass (`uv run mypy src/pmd`)
- [ ] Linting passes (`uv run ruff check src/`)
- [ ] Code is formatted (`uv run ruff format src/`)
- [ ] New code has tests
- [ ] Docstrings for public APIs
- [ ] No hardcoded secrets or credentials
- [ ] Error handling is appropriate

### Issue Tracking

This project uses **bd (beads)** for issue tracking. See `AGENTS.md` for details.

**Key Commands:**
```bash
# Check ready work
bd ready

# Create issue
bd create "Issue title" -t bug|feature|task

# Claim work
bd update <id> --status in_progress

# Complete work
bd close <id> --reason "Done"
```

**Important:**
- Do NOT use markdown TODOs
- Do NOT use external issue trackers
- Always commit `.beads/issues.jsonl` with code changes
