# Integration Test Cases: `pmd.store`

Scope covers `Database`, `CollectionRepository`, `DocumentRepository`, `EmbeddingRepository`, and `FTS5SearchRepository` using the pytest fixtures in `tests/conftest.py`. These are definitions only (no implementations yet).

## Database / Schema
- **Initializes schema**: connect to fresh temp DB, verify all core tables exist (`content`, `collections`, `documents`, `documents_fts`, `content_vectors`, `path_contexts`, `ollama_cache`).
- **Vector table gated by extension**: patch `Database._vec_available` to `False` after connect; ensure vector virtual table is not created; when forced to `True`, `content_vectors_vec` should exist.
- **Transaction commit/rollback**: within `Database.transaction()`, confirm successful statements commit and exceptions roll back (row absence after forced error).

## CollectionRepository
- **Create and list**: create two collections, assert unique IDs, `list_all` sorted by name, stored `glob_pattern` default of `**/*.md`.
- **Duplicate name rejected**: attempting to create collection with existing name raises `CollectionExistsError` and leaves DB unchanged.
- **Getters by id/name**: retrieving non-existent returns `None`; existing matches created attributes.
- **Rename constraints**: rename updates `updated_at`, but renaming to an existing name raises `CollectionExistsError` without altering records.
- **Update path/glob**: update only path preserves glob; update both writes both; timestamps move forward.
- **Remove cascade and cleanup**: after adding documents/embeddings for a collection, `remove` deletes collection, related documents, path contexts, and orphaned `content`/`content_vectors` rows; return tuple matches counts.

## DocumentRepository
- **Add new document**: `add_or_update` inserts content-addressable entry, sets `is_new=True`, stores hash, body length, and active=1.
- **Update existing path**: calling `add_or_update` on same path with different content returns `is_new=False`, updates `hash`, `modified_at`, `title`, keeps one active row.
- **Re-activate after soft delete**: delete document (active=0), then `add_or_update` same path should flip active back to 1 and reuse/insert content hash as needed.
- **List and get respect active flag**: `list_by_collection(active_only=True)` omits soft-deleted docs; `get` returns `None` for inactive paths; `active_only=False` includes them.
- **Delete nonexistent is false**: deleting unknown path returns `False` and leaves data unchanged.
- **Hash comparison**: `check_if_modified` returns `True` for new paths or different hash; `False` when hash matches stored doc.
- **Content length**: `get_content_length` returns exact byte length for active doc and `None` when missing.
- **Path prefix query**: `get_by_path_prefix` filters by prefix and keeps sort order.

## EmbeddingRepository
- **Store metadata only when vec unavailable**: with `db._vec_available=False`, `store_embedding` inserts metadata rows and skips `content_vectors_vec` creation; `has_embeddings` reflects stored rows by hash/model.
- **Store with vector table**: with `db._vec_available=True`, `store_embedding` writes to both metadata and vector table (one row per seq), replacing existing vectors on same hash/seq.
- **Delete embeddings**: `delete_embeddings` removes metadata (and vector rows when available) and returns deleted count.
- **Model-wide clear**: `clear_embeddings_by_model` deletes only matching model rows; other models remain.
- **Count helpers**: `count_embeddings` with/without model filter matches inserted rows.
- **Vector search basics**: when vec enabled and embeddings inserted for multiple docs/chunks, `search_vectors` returns at most one result per document (best chunk), ordered by similarity, respects `limit` and `min_score`, and returns empty list if vec disabled or query embedding empty.

## FTS5SearchRepository
- **Index and search**: `index_document` populates FTS table for a doc; `search_fts` returns that doc with normalized score ≥0 and correct source `FTS`.
- **Collection scoping**: searching with `collection_id` filters results to that collection and excludes others.
- **Score normalization**: multiple docs with different ranks produce normalized scores where highest relevance normalizes to 1.0; `min_score` threshold filters lower scores.
- **Reindex collection**: after modifying document content, `reindex_collection` rewrites FTS rows and returns count of indexed docs; old content no longer appears.
- **Remove from index**: `remove_from_index` removes a row; subsequent search returns empty list while document still exists in `documents`.
- **Clear index**: `clear_index` wipes all FTS rows; search returns empty.
- **Vector delegation**: `search_vec` delegates to provided `EmbeddingRepository`; when repository is `None`, returns empty list.

## End-to-End Flows
- **Content-addressable reuse across collections**: create two collections sharing identical document content; ensure `content` table stores single hash, while `documents` rows are distinct per collection.
- **Document lifecycle with search**: create collection and document, index into FTS, add embeddings, verify both `search_fts` and `search_vec` return the doc; after soft delete, both searches omit the doc; after re-add/update, searches reflect latest body/hash.
- **Cleanup after collection removal**: build collection with docs, embeddings, FTS entries; call `remove`; assert subsequent `search_fts`/`search_vec` return empty and no orphaned hashes or vectors remain.
