# Polymorphic Search Backends: Recommended Interface Shape

## Goal recap
You want polymorphism for **search operations** so pipelines can run multiple backends (or same backend with different configs) without special-casing. This is a good fit for a **search-only** interface and adapter pattern.

## Recommended approach
### 1) Keep a small search interface
Use the existing `SearchRepository[QueryT]` shape as the contract:
- `search(query, limit, collection_id=None, min_score=0.0) -> list[SearchResult]`
- Avoid indexing/storage responsibilities in this interface.

This keeps the polymorphism surface minimal and stable.

### 2) Make vector search conform to the interface (via adapter or direct method)
You have two clean options:

**Option A: Adapter class (clean separation)**
- Create a thin `VectorSearchRepository` wrapper that takes an `EmbeddingRepository` and implements `SearchRepository[list[float]]`.
- This avoids changing `EmbeddingRepository` and makes intent explicit: “this object is a search backend.”

**Option B: Add a `search()` method to `EmbeddingRepository`**
- Implement `search()` as a simple call to `search_vectors()`.
- This makes `EmbeddingRepository` itself a `SearchRepository`.

Either route gives you polymorphism without forcing indexing methods into the shared interface.

### 3) Enable multiple backends with config
To support multiple backend instances/configs:
- Treat each backend as its own `SearchRepository` instance.
- Pass a list of backends to the pipeline (e.g., `List[SearchRepository]`) and execute them uniformly.
- Weighting/merging can remain in fusion logic (RRF already exists).

## Why this fits your goal
- **Polymorphism**: pipelines can call `search()` on any backend.
- **Extensibility**: add new backends (BM25 variants, alternate vector stores, “title-only” vector, etc.) without touching pipelines.
- **Simplicity**: avoids bloating the base class with unrelated indexing/maintenance concerns.

## Pushback on a unified “store+search” base
Even with polymorphism as the goal, unifying **indexing/storage** still doesn’t fit:
- It adds required methods that many search backends won’t need or can’t implement cleanly.
- It makes the base contract unstable as new backends appear (every new capability pressures the base to grow).

So: keep the base interface **search-only**.

## Next-step decision points
1) Do you prefer an **adapter** (explicit, minimal impact) or making `EmbeddingRepository` implement `search()` directly?
2) Should the pipeline accept a **list of backends** (for hybrid), or a **dict of named backends** (for per-backend configuration/weights)?
3) Do you want a single `SearchBackendConfig` that pipelines can consume to instantiate and parameterize each backend?
