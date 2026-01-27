# Implementation Plan: Simple Search Script

## Goal
Create a script at `./scripts/search.py` that:
- Takes a single text parameter (the search query)
- Executes FTS, vector, and hybrid searches
- Prints results to screen
- Has no complicated argument handling or logic

## Current State

The `SearchService` class provides unified search, but requires boilerplate:

```python
from idx.search import SearchService
from idx.search.models import SearchCriteria
from idx.store.database import get_session
from idx.store.session_context import use_session

with get_session() as session:
    with use_session(session):
        service = SearchService()
        results = service.search(SearchCriteria(query="...", mode="fts"))
```

This is too much setup for a simple script.

## Proposed Changes

### 1. Add convenience function to `idx.search`

**File: `src/idx/search/__init__.py`**

Add a module-level `search()` function that handles all setup internally:

```python
def search(
    query: str,
    mode: str = "hybrid",
    limit: int = 10,
    dataset_name: str | None = None,
) -> SearchResults:
    """Execute a search with automatic session management.

    Convenience function that handles database session setup internally.
    For more control, use SearchService directly.

    Args:
        query: Search query string.
        mode: Search mode - "fts", "vector", or "hybrid".
        limit: Maximum results to return.
        dataset_name: Optional filter to specific dataset.

    Returns:
        SearchResults with matching documents.
    """
    from idx.store.database import get_session
    from idx.store.session_context import use_session

    with get_session() as session:
        with use_session(session):
            service = SearchService()
            return service.search(SearchCriteria(
                query=query,
                mode=mode,
                limit=limit,
                dataset_name=dataset_name,
            ))
```

**Update `__all__` to export it:**
```python
__all__ = [
    "search",  # New convenience function
    "FTSSearch",
    "HybridSearch",
    "SearchCriteria",
    ...
]
```

### 2. Create the script

**File: `scripts/search.py`**

```python
#!/usr/bin/env python3
# /// script
# dependencies = ["idx"]
# ///
"""Search the idx database.

Usage:
    uv run python scripts/search.py "your search query"
"""
import sys
from idx.search import search

def main():
    if len(sys.argv) < 2:
        print("Usage: search.py <query>", file=sys.stderr)
        return 1

    query = " ".join(sys.argv[1:])

    print(f"=== FTS Search: {query!r} ===\n")
    fts_results = search(query, mode="fts")
    _print_results(fts_results)

    print(f"\n=== Vector Search: {query!r} ===\n")
    vector_results = search(query, mode="vector")
    _print_results(vector_results)

    print(f"\n=== Hybrid Search: {query!r} ===\n")
    hybrid_results = search(query, mode="hybrid")
    _print_results(hybrid_results)

    return 0

def _print_results(results):
    if not results.results:
        print("  No results found.")
        return

    for i, r in enumerate(results.results, 1):
        print(f"{i}. [{r.dataset_name}] {r.path} (score: {r.score:.3f})")
        if r.chunk_text:
            snippet = r.chunk_text[:100].replace("\n", " ")
            print(f"   {snippet}...")

    if results.timing_ms:
        print(f"\n  ({len(results.results)} results in {results.timing_ms:.0f}ms)")

if __name__ == "__main__":
    sys.exit(main())
```

## Summary of Changes

| File | Change |
|------|--------|
| `src/idx/search/__init__.py` | Add `search()` convenience function |
| `scripts/search.py` | New script (13 lines of logic) |

## Script Complexity Analysis

- **Arguments**: Single positional parameter (query string)
- **Logic in script**: Only formatting/printing (13 lines)
- **Logic in idx**: Session management, search orchestration, result models

The script is trivially simple because all search logic is encapsulated in `idx.search.search()`.

## Usage Examples

```bash
# Basic search
uv run python scripts/search.py "python async"

# Multi-word query
uv run python scripts/search.py "machine learning tutorial"

# Quoted if needed for shell special chars
uv run python scripts/search.py "what is OAuth2?"
```
