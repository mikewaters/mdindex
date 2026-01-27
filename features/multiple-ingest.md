January 27, 2026
ChatGPT — GPT-5.2 Thinking

Below is a set of **developer requirements** you can hand to someone (or use as your own spec) to build the Python module you described. I’m writing these as **testable requirements + acceptance criteria**, and I’m grounding the LlamaIndex-specific behaviors (readers → `Document`, ingestion pipeline caching, docstore strategy, and persistence) in the official docs. ([LlamaIndex][1])

---

## 1) Module goal and scope

### Goal

Provide a Python module that:

1. **Loads a corpus** via a LlamaIndex `Reader` (producing `Document` objects). ([LlamaIndex][1])
2. Runs a **Preparatory IngestionPipeline** that:

   * performs **metadata extraction only** (no chunking, no embeddings),
   * writes **canonical “bronze” artifacts** (raw/original content + extracted metadata) to a centralized **DuckDB-backed docstore** (see storage requirements below). ([LlamaIndex][2])
3. Runs a **Downstream IngestionPipeline** that:

   * reads from bronze (DuckDB) as its document source,
   * performs **chunking + embedding + vector storage**, and
   * maintains its own independent persistence (docstore/cache/vector store storage context). ([LlamaIndex][2])

---

## 2) Data contract (IDs, hashes, versioning)

### Required identifiers

For each logical source document:

* `source_id` (stable identifier, e.g., URL/path/DB PK)
* `content_hash` = hash of original bytes/text (immutable snapshot identity)

### Required metadata versioning

Because the preparatory pipeline will evolve (metadata extraction improves without changing content), the bronze layer must include:

* `bronze_meta_version` (string or semver)
* `bronze_meta_hash` = hash of extracted metadata payload (changes when metadata changes)

### Contract invariant

Downstream pipelines **must not decide skip/re-run using `content_hash` alone**. Downstream “document management” must consider changes in metadata (e.g., by including `bronze_meta_hash` and/or `bronze_meta_version` in the metadata that participates in the document/node hash). This aligns with LlamaIndex’s doc management model: it stores `doc_id -> document_hash` and skips if the hash hasn’t changed. ([LlamaIndex][3])

**Acceptance criteria**

* If only `bronze_meta_version` changes (same content), downstream pipeline **reprocesses** the doc (or at minimum refreshes stored metadata and any embeddings that incorporate it).
* If neither content nor bronze metadata changed, downstream pipeline **skips** work (benefiting from ingestion caching and/or docstore UPSERTS). ([LlamaIndex][2])

---

## 3) Storage requirements (bronze vs derived)

### Bronze storage (centralized; DuckDB)

**Requirement:** Bronze artifacts are the canonical store of “original corpus + extracted metadata” and live in DuckDB.

**Implementation requirement:** Use a storage approach compatible with LlamaIndex persistence and/or fsspec-backed storage. LlamaIndex supports persisting to backends supported by fsspec (which includes DuckDB via DuckDB DocStore). ([LlamaIndex][4])

**Bronze docstore contents**

* Store the “as-is” `Document` content (or its exact source bytes/text) + extracted metadata fields.
* Store the version fields (`content_hash`, `bronze_meta_hash`, `bronze_meta_version`, timestamps, extractor config) as part of `Document.metadata`.

> Note: if the team can’t directly persist the docstore to DuckDB in your current LlamaIndex version, the fallback requirement is “persist locally then upload to DuckDB,” which is a known ask/constraint raised by users. ([GitHub][5])

### Derived storage (per pipeline; independent)

Each ingestion pipeline must have **its own** persistence layer:

* Prep pipeline persistence: its own cache and (optionally) docstore for dedupe bookkeeping
* Downstream pipeline persistence: its own cache + docstore (if using document management) + vector store/index storage context ([LlamaIndex][2])

**Acceptance criteria**

* Deleting any derived storage does not affect bronze.
* Given only bronze + pipeline configs, derived outputs can be rebuilt deterministically.

---

## 4) Pipeline requirements

### 4.1 Reader ingestion (source → Documents)

**Requirement:** Provide a pluggable reader interface so callers can supply any LlamaIndex `Reader` (or a wrapper), producing `Document` objects. LlamaIndex readers create `Document`s with text + metadata; `SimpleDirectoryReader` supports setting IDs from filenames (`filename_as_id`) and injecting file metadata. ([LlamaIndex][1])

**Acceptance criteria**

* Caller can pass a reader instance and reader config, and receive a list/iterator of `Document`s.
* Each `Document` has a stable `doc_id` strategy (typically `source_id`) to support downstream refresh. ([LlamaIndex][3])

### 4.2 Preparatory IngestionPipeline (Documents → bronze)

**Requirement:** The preparatory pipeline:

* only runs metadata extractors / enrichment transforms (no chunking, no embedding),
* writes updated bronze artifacts to DuckDB docstore.

**Document management behavior (prep)**

* Prep should support a “refresh” mode:

  * If content unchanged but metadata extractor version changed, it **updates bronze metadata** and writes a new `bronze_meta_hash`.
  * If content changed, it writes a new `content_hash` snapshot.

*(Implementation detail: don’t rely on LlamaIndex docstore UPSERTS alone for this, since your meaning of “changed” is content+metadata-version; explicitly include `bronze_meta_version/hash` in metadata.)*

### 4.3 Downstream IngestionPipeline (bronze → chunks/embeddings/vector)

**Requirement:** Downstream pipeline:

* reads bronze Documents from DuckDB docstore,
* chunks + embeds + upserts to a configured vector store,
* uses ingestion pipeline caching and/or docstore strategies to skip unchanged docs. ([LlamaIndex][2])

**Docstore strategy requirement**

* Must support `DocstoreStrategy.UPSERTS` (or `UPSERTS_AND_DELETE` if you want deletions) to keep the vector store consistent with the current bronze set. ([LlamaIndex][6])

**Acceptance criteria**

* When `bronze_meta_hash` changes for a doc (same content), downstream pipeline does not silently skip if downstream uses bronze metadata.
* When neither `content_hash` nor `bronze_meta_hash` changed, downstream does skip (cache hit / docstore skip).

---

## 5) Persistence and reload requirements

### StorageContext recreation

**Requirement:** The module must allow reloading each pipeline’s state by recreating `StorageContext` with the same backing stores / configuration (persist dir, vector store client, etc.). ([LlamaIndex][7])

### Ingestion cache

**Requirement:** Each pipeline uses its own `IngestionCache` namespace/collection to avoid cross-pipeline cache contamination. Caching is defined as “each node + transformation combination is hashed and cached.” ([LlamaIndex][2])

**Acceptance criteria**

* Two pipelines run against the same bronze input do not share cache entries unless explicitly configured to do so.

---

## 6) Public API requirements (what the module exports)

### Required functions/classes

1. `ingest_to_bronze(reader, reader_cfg, prep_cfg, bronze_store_cfg) -> BronzeIngestReport`
2. `build_vector_index(bronze_store_cfg, downstream_cfg, vector_store_cfg) -> VectorIngestReport`
3. `reprocess_source(reader, ...)` convenience wrapper that:

   * reruns prep,
   * decides which downstream pipelines to run based on change detection.

### Required reports

Both reports must include:

* counts: loaded / new / updated / skipped
* IDs affected
* hashes/versions used (`content_hash`, `bronze_meta_hash`, versions)

---

## 7) Change detection rules (must be explicit)

### Required rule table

For each `source_id`:

* If `content_hash` changed → **always** re-run downstream
* Else if `bronze_meta_hash` changed and downstream “uses bronze metadata” → re-run downstream
* Else → skip downstream

This is the “content vs metadata” flow you described, and it’s consistent with LlamaIndex doc management being “skip if hash unchanged” *as long as the hash incorporates the relevant metadata signal.* ([LlamaIndex][3])

---

## 8) Testing requirements (acceptance tests)

1. **Metadata iteration test**

* Same content ingested twice, prep extractor version increments, metadata changes.
* Expect bronze meta updated + downstream re-run when configured to use bronze metadata.

2. **No-op test**

* Same content + same bronze metadata → downstream skip (cache/docstore behavior). ([LlamaIndex][2])

3. **Content change test**

* Content changes for same `source_id` → downstream upsert occurs (and delete handling if enabled). ([LlamaIndex][6])

4. **Pipeline isolation test**

* Clearing downstream persistence does not impact bronze.
* Clearing prep persistence does not erase bronze.

---

If you want this to be even more “requirement-shaped,” I can rewrite it in a more formal template (Actors / Preconditions / Postconditions / Nonfunctional requirements). Also, if you tell me which vector store you’re targeting (Chroma/Qdrant/Pinecone/pgvector/etc.), I’ll add concrete persistence requirements for that store using LlamaIndex’s storage customization guidance. ([LlamaIndex][8])

[1]: https://llamaindexxx.readthedocs.io/en/latest/understanding/loading/loading.html?utm_source=chatgpt.com "Loading Data (Ingestion) - LlamaIndex v0.10.10"
[2]: https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/?utm_source=chatgpt.com "Ingestion Pipeline | LlamaIndex Python Documentation"
[3]: https://developers.llamaindex.ai/python/examples/ingestion/document_management_pipeline/?utm_source=chatgpt.com "Ingestion Pipeline + Document Management"
[4]: https://developers.llamaindex.ai/python/framework/module_guides/storing/?utm_source=chatgpt.com "Storing | LlamaIndex Python Documentation"
[5]: https://github.com/run-llama/llama_index/issues/19760?utm_source=chatgpt.com "[Question]: Saving docstore to file storage via persist()"
[6]: https://developers.llamaindex.ai/python/framework-api-reference/ingestion/?utm_source=chatgpt.com "DocstoreStrategy"
[7]: https://developers.llamaindex.ai/python/framework/module_guides/storing/save_load/?utm_source=chatgpt.com "Persisting & Loading Data"
[8]: https://developers.llamaindex.ai/python/framework/module_guides/storing/customization/?utm_source=chatgpt.com "Customizing Storage | LlamaIndex Python Documentation"
