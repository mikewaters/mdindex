# Plan: Promote `uri` to a Required Document Attribute

## Goal
Make `documents.uri` a required attribute for all documents, including filesystem sources (using `file://` URIs), while preserving the existing unique key `(collection_id, path)`.

## Current State (Summary)
- `documents` table uses `path` as the identity within a collection.
- Canonical source URI is stored only in `source_metadata.source_uri`.
- Many queries and APIs reference `documents.path` (FTS, search results, MCP get_document).

## Proposed Schema Changes
1) Add a new required column to `documents`:
   - `uri TEXT NOT NULL`
2) Add an index for lookup efficiency:
   - `CREATE INDEX IF NOT EXISTS idx_documents_uri ON documents(uri)`
3) Keep existing uniqueness constraint:
   - `UNIQUE(collection_id, path)` remains unchanged.

No global uniqueness constraint on `uri` is required.

## Migration Strategy
1) **Schema migration**
   - Add `uri` column (initially nullable for migration safety, then backfill, then enforce NOT NULL).
2) **Backfill**
   - Primary source: `source_metadata.source_uri` where available.
   - Fallback: compute from collection config + `documents.path`.
     - Filesystem: `Path(collection.pwd) / path` then `.resolve().as_uri()`.
     - HTTP/HTTPS: join `source_config.base_url` (or `collection.pwd`) + path.
     - Entity: use `source_config.uri` (or `collection.pwd`) + path if applicable.
3) **Finalize**
   - Enforce NOT NULL after backfill.
   - Add `idx_documents_uri`.

## Code Changes

### Repositories
- `DocumentRepository.add_or_update` should require and persist `uri`.
- `DocumentRepository.get` and list queries should include `uri` in `DocumentResult` (extend dataclass if needed).
- Any direct inserts to `documents` (if any) must now include `uri`.

### Services
- Loading/Indexing pipelines must construct a stable URI for each document.
- For filesystem sources, use `file://` URIs by resolving the full path.
- For LlamaIndex adapter, construct `uri` as described in the adapter rules.

### Types
- Update `DocumentResult` (and any related types) to include `uri`.

### Metadata Sync
- `source_metadata.source_uri` can be kept as a mirror of `documents.uri` for fetch tracking.
- Optionally, treat mismatches as a warning during indexing.

## Compatibility Considerations
- Existing APIs that rely on `path` continue to work (no breaking changes to CLI/MCP path-based calls).
- New APIs can optionally expose `uri` where appropriate.

## Testing Plan (future)
- Migration backfill correctness for filesystem and non-filesystem collections.
- Insert/update requires `uri` for new documents.
- Query coverage for `documents.uri` and index usage.

## Open Questions
- Should `uri` be included in search results and MCP document retrieval payloads by default?
- Should `source_metadata.source_uri` be deprecated or kept as a fetch-specific mirror?
