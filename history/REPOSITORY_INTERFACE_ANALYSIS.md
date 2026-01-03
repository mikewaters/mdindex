# Analysis: Can EmbeddingRepository and FTS5SearchRepository Share a Base?

## Short answer
There is **partial overlap** in responsibilities (both can “search” and both touch storage), but their **interfaces are not close enough today** to justify making them derived classes of the same base without either (a) weakening the base to a trivial marker, or (b) forcing unnatural methods onto one side. A small, *search-only* interface could be shared, but a unified “store/index + search” base likely adds confusion rather than clarity.

## Current public surfaces (signatures & roles)

### FTS5SearchRepository (`src/pmd/store/search.py`)
- **Constructor**: `__init__(db: Database)`
- **Search**: `search(query: str, limit=5, collection_id=None, min_score=0.0) -> list[SearchResult]`
- **Indexing / maintenance**:
  - `index_document(doc_id: int, path: str, body: str) -> None`
  - `reindex_collection(collection_id: int) -> int`
  - `remove_from_index(doc_id: int) -> None`
  - `clear_index() -> None`
- **Notes**:
  - Indexing is part of this class’s core responsibilities.
  - Query type is **string**; semantics are FTS5 syntax.

### EmbeddingRepository (`src/pmd/store/embeddings.py`)
- **Constructor**: `__init__(db: Database)`
- **Search**: `search_vectors(query_embedding: list[float], limit=5, collection_id=None, min_score=0.0) -> list[SearchResult]`
- **Indexing / storage**:
  - `store_embedding(hash_value: str, seq: int, pos: int, embedding: list[float], model: str) -> None`
  - `delete_embeddings(hash_value: str) -> int`
  - `clear_embeddings_by_model(model: str) -> int`
  - `count_embeddings(model: str | None = None) -> int`
  - `has_embeddings(hash_value: str, model: str | None = None) -> bool`
  - `get_embeddings_for_content(hash_value: str) -> list[tuple[int, int, str]]`
- **Notes**:
  - Indexing is **chunk-level** and keyed by content hash + sequence.
  - Query type is **vector embedding**.
  - This class is not expressed as a `SearchRepository` today.

## Similarities that could justify a shared base
- Both support a **search(query, limit, collection_id, min_score)**-shaped operation.
- Both are **repository-style** classes backed by `Database`.
- Both output `list[SearchResult]`.

## Divergences that make a single “store+search” base awkward
1) **Different query types and semantics**
   - FTS expects a string; vector search expects a numeric vector.
   - A common `search` method could be generic (as `SearchRepository[QueryT]` already is), but EmbeddingRepository would need to implement that method or adapt its existing one.

2) **Indexing responsibilities are incomparable**
   - FTS indexing: `index_document(doc_id, path, body)`
   - Embedding indexing: `store_embedding(hash_value, seq, pos, embedding, model)`
   - There is no clean base method that represents both without becoming overly abstract (e.g., `index(obj: Any)`), which removes meaningful constraints.

3) **Different lifecycle and maintenance operations**
   - FTS has `reindex_collection`, `clear_index`, `remove_from_index`.
   - Embeddings have `clear_embeddings_by_model`, `count_embeddings`, `has_embeddings`, `get_embeddings_for_content`.
   - Forcing these into a base would either bloat the interface or require many `NotImplemented` stubs.

4) **SearchRepository currently expresses only search**
   - The existing abstract base in `search.py` models **search only**, not indexing. That part of your observation is correct: only FTS implements it today; embeddings do not.

## What *could* be shared cleanly
A **search-only interface** could be shared, aligning with the existing `SearchRepository`:
- Introduce (or reuse) a `SearchRepository[QueryT]` for vector search by adding `search()` to `EmbeddingRepository` as a thin wrapper over `search_vectors()`.
- This would allow the pipeline to treat FTS and vector repos polymorphically for *search only*.

However, this doesn’t simplify storage/indexing, and it risks adding indirection for minimal gain unless you want to **plug in multiple search backends** or abstract over hybrid strategies.

## Recommendation / pushback
- **Pushback:** A unified base that covers both indexing and search is not a good fit with current responsibilities. It would either be too abstract to be useful or force methods that don’t conceptually belong to the other class.
- **If simplification is still desired**, keep it small and explicit:
  - Create a *search-only* interface (you already have it) and consider letting `EmbeddingRepository` conform to it by implementing a `search()` method.
  - Leave indexing/storage concerns separate; those are fundamentally different.

## Follow-up questions (if you want to proceed)
1) Is your goal **polymorphism** in the search pipeline (inject any search backend) or **code reuse**?
2) Do you want to expose vector search through the same `search()` method signature, even if its query type is a list of floats?
3) Would you accept a design where `EmbeddingRepository` implements the search interface but keeps its storage methods separate (no shared base for indexing)?
