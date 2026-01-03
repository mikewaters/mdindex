# Implementation Plan: Keep Title-Only Docs Lexical-Only

## Reasoning (ultrathink)
The current pipeline embeds raw document content, even if it is empty or title-only. When the text is just a title, embedding models often return a near-centroid vector, which looks “kind of similar” to many queries and pollutes vector retrieval. In PMD, vector search only considers rows in `content_vectors_vec`, so the most reliable fix is to **never store embeddings for title-only docs**. That ensures they can’t appear in vector search at all.

You want title-only docs to remain discoverable via **lexical (FTS) only**, so we will index a synthetic “title-only body” into FTS when the body is otherwise empty. This gives you the best of both worlds: title-only docs are searchable with BM25 but excluded from vector similarity.

Given PMD’s architecture, we can do this without schema changes:
- Embeddings: skip for title-only → no rows in `content_vectors` / `content_vectors_vec`.
- FTS: index the title into `documents_fts.body` when there is no body content.
- No migrations/backfill required (pre-alpha).

## Clarified requirements captured
- “Empty” for embeddings = **title-only markdown** (single `# Heading` with no meaningful body).
- Title-only docs are **lexical-only**.
- When body is empty, index a **title-only body** in FTS.
- No database migrations or backfill required.

## Implementation steps

### 1) Add normalization to detect title-only docs
Create a small helper to determine embeddable body:
- If content contains a leading `# ` heading and nothing else (ignoring whitespace), treat it as **title-only**.
- The helper should return:
  - `embeddable_body`: empty string if title-only
  - `fts_body`: title string if title-only, otherwise the original content

Suggested location:
- Add a utility in `src/pmd/search/chunking.py` or a new module `src/pmd/search/text.py` (prefer `text.py` to keep chunking focused on chunking).

Pseudo-logic:
```
- Strip BOM + whitespace
- If empty: treat as title-only
- If first non-empty line starts with '# ':
    title = line[2:].strip()
    remaining = content with that line removed
    If remaining is empty/whitespace: title-only
```

### 2) Skip embeddings for title-only docs
Update `EmbeddingGenerator.embed_document()` in `src/pmd/llm/embeddings.py`:
- Compute `embeddable_body` using the helper.
- If `embeddable_body` is empty:
  - `embedding_repo.delete_embeddings(hash_value)` (defensive cleanup).
  - Log debug: “Skipping embeddings for title-only doc”.
  - Return 0.
- Otherwise, chunk/embed **embeddable_body** (not raw content).

### 3) Prevent empty chunk creation
Update `chunk_document()` in `src/pmd/search/chunking.py`:
- If `content.strip()` is empty, return `ChunkingResult(chunks=[], total_bytes=0)`.
- This avoids ever emitting empty chunks even if the caller forgets to guard.

### 4) Index “title-only body” into FTS
Update `CollectionRepository.index_documents()` in `src/pmd/store/collections.py`:
- Compute `fts_body` using the helper.
- When indexing into FTS, pass `fts_body` instead of raw content.
- This ensures title-only docs remain searchable in BM25.

### 5) Tests
Add unit tests to verify behavior:
- `chunk_document("")` returns zero chunks.
- `EmbeddingGenerator.embed_document()` returns 0 for title-only content and does not create embeddings.
- FTS indexing uses title as body for title-only docs (can be tested via a small integration test against a test DB).

No migration or backfill steps are needed per requirement.

## Acceptance criteria
- Title-only markdown files never generate embeddings.
- Vector search results never include title-only docs.
- Title-only docs remain discoverable via FTS using their titles.
- No migrations/backfills required.
