# Metadata Implementation Plan

## Purpose and scope

- Add metadata-aware retrieval and ranking to improve search accuracy without requiring users to know tags.
- Keep behavior schema-driven and configurable for multiple source types.
- Avoid hard filters by default; use metadata as soft ranking signals unless explicitly configured otherwise.
- No labeled test data available; emphasize deterministic inference, safe defaults, and measurable heuristics.

## Context summary (current state)

- Vector store supports vector and hybrid search; embeddings are from ModernBERT (nomic).
- Source metadata is stored in `src/pmd/store/source_metadata.py` and includes an `extra_metadata` payload; this is currently used for fetch metadata and is not surfaced in search.
- Tags are namespaced (e.g., `#domain/entity-type`, `#subject/topic/child`) and schemas are known per source type.
- Non-tag attributes exist (e.g., source type), but efforts/initiatives are out of scope for now.

## Desired behavior

- Query-time inference should detect relevant tags/attributes without user guidance.
- Tags should influence ranking (soft boosts) rather than exclude results.
- Schema config should allow adding new tag namespaces and rules without code changes.

---

## Architecture overview

### Core components (conceptual)

1) Metadata schema and ontology
- **MetadataSchema**: Valid tag prefixes, field types, and allowed values per source type.
- **Ontology**: Tag relationships (parent/child/synonym) and alias resolution.

2) Query-time metadata inference
- **QueryMetadataInferencePipeline**: Executes ordered inference steps; merges tag candidates.
- **InferenceStep** interface:
  - Lexical / dictionary match step
  - Ontology expansion step
  - Embedding-based tag similarity step (tag description embeddings)

3) Retrieval and ranking
- **CandidateRetriever**: Vector, keyword/BM25, and tag-based retrieval channels.
- **Ranker / ScoringPolicy**: Combines vector, keyword, and metadata signals with configurable weights.

4) Orchestration
- **SearchService**: Builds a `QueryContext`, runs inference, retrieves candidates, reranks results.

---

## Configuration design (schema-first)

### Schemas
- Per-source-type config with:
  - allowed tag namespaces
  - tag hierarchy scope (ontology subset)
  - attribute types (categorical/scalar/text)

### Inference pipeline
- Ordered list of inference steps with:
  - enabled/disabled flags
  - min confidence thresholds
  - max candidates per step

### Ranking policy
- Weights for:
  - vector similarity
  - keyword/BM25 score
  - exact tag match
  - ontology-expanded tag match
  - inferred tag confidence
  - source-type match
- Per-schema overrides supported

---

## Module alignment (current code boundaries)

- Hybrid retrieval and fusion live in `src/pmd/search/pipeline.py` and `src/pmd/search/fusion.py`.
- Result scoring and normalization live in `src/pmd/search/scoring.py`.
- FTS retrieval lives in `src/pmd/store/search.py`.
- Vector search lives in `src/pmd/store/vector_search.py` and `src/pmd/store/embeddings.py`.
- Query expansion and reranking are in `src/pmd/llm/query_expansion.py` and `src/pmd/llm/reranker.py`.
- Search service entry points are in `src/pmd/services/search.py` and CLI wiring in `src/pmd/cli/commands/search.py`.
- Metadata storage for documents is currently only represented via `source_metadata.extra_metadata` in `src/pmd/store/source_metadata.py`.

This plan assumes metadata signals will be injected into the hybrid pipeline as an additional retrieval channel and as scoring features during ranking.

---

## Proposed implementation phases

### Phase 0: Baseline inventory and schema mapping

Goals:
- Identify all existing metadata fields and how they are stored.
- Enumerate tag namespaces and source-type schemas.

Work in this phase centers on auditing how metadata is currently recorded, agreeing on a schema file structure, and selecting an ontology file shape suitable for namespace expansion. The immediate artifacts are a schema config skeleton (for two representative source types) and a small ontology stub used for early pipeline tests.

### Phase 1: Query-time inference pipeline (deterministic)

Goals:
- Produce inferred tags for a query, with confidence and evidence.
- No ML training required.

Work here defines the inference step contract, the merge rules for candidate tags, and the initial deterministic inference steps (lexical match and ontology expansion). The output is a pipeline specification that is compatible with the hybrid search pipeline and can be wired into `src/pmd/search/pipeline.py`.

### Phase 2: Tag-aware retrieval channel

Goals:
- Allow exact and expanded tag matches to contribute candidates.

This phase establishes a tag-aware retrieval channel using the existing metadata store as the initial index. It defines recall caps and result mixing rules to prevent tag-only results from dominating hybrid retrieval.

### Phase 3: Ranking policy with metadata signals

Goals:
- Combine vector + keyword + metadata signals in a tunable way.

This phase introduces a scoring policy that blends metadata signals with RRF outputs. Missing metadata never reduces scores; only positive boosts apply. Schema-specific overrides are allowed but optional.

### Phase 4: Embedding-assisted tag inference (optional)

Goals:
- Improve tag inference coverage using tag-definition embeddings.

This phase adds an embedding-assisted inference step that retrieves the most semantically similar tags to a query using tag definition embeddings. It is optional and can be enabled after deterministic inference is stable.

### Phase 5: Evaluation and tuning without labels

Goals:
- Establish a lightweight evaluation loop.

This phase defines a manual evaluation loop using real queries and lightweight annotations to tune weights and thresholds.

---

## Key design decisions (open)

- Should ontology expansion be limited to 1 hop by default?
- Should source-type inference be used as a strong prior or a mild boost?
- What is the maximum metadata boost allowed to prevent overriding semantic similarity?
- Do we need a distinct tag namespace for “effort” even if out of scope now?

---

## Integration points (to inspect)

- `src/pmd/store/source_metadata.py` (current metadata storage model)
- Hybrid search components: `src/pmd/search/pipeline.py`, `src/pmd/search/fusion.py`, `src/pmd/search/scoring.py`
- Retrieval backends: `src/pmd/store/search.py`, `src/pmd/store/vector_search.py`, `src/pmd/store/embeddings.py`
- Service and CLI: `src/pmd/services/search.py`, `src/pmd/cli/commands/search.py`

---

## Non-goals (for now)

- Hard filtering by metadata unless explicitly requested.
- Training a learned reranker (no labels yet).
- Effort/initiative tags (out of scope per current request).

---

## Initial configuration sketch (draft)

This draft uses TOML because `src/pmd/core/config.py` already loads TOML config. It is intentionally minimal and focuses on schema, ontology, inference pipeline, and ranking policy.

```toml
[metadata]
enabled = true

[metadata.schemas.default]
allowed_namespaces = ["#domain", "#subject"]
source_types = ["filesystem", "http", "entity"]

[metadata.schemas.default.fields]
source_type = { kind = "categorical" }
topics = { kind = "tag_list", namespace = "#subject" }
entities = { kind = "tag_list", namespace = "#domain" }

[metadata.ontology]
format = "adjacency"
file = "config/metadata_ontology.json"
max_hops = 1

[metadata.inference]
steps = ["lexical", "ontology_expand", "embedding_tags"]
max_candidates = 8
min_confidence = 0.2

[metadata.inference.lexical]
alias_file = "config/metadata_aliases.json"
confidence = 0.7

[metadata.inference.ontology_expand]
confidence = 0.5

[metadata.inference.embedding_tags]
enabled = false
top_k = 5
confidence = 0.4

[metadata.ranking]
max_total_boost = 0.35
exact_tag_weight = 0.20
expanded_tag_weight = 0.12
inferred_tag_weight = 0.08
source_type_weight = 0.10
```

## Ontology stub example (draft)

The ontology can start as a small adjacency list, with a separate alias map for lexical inference.

```json
{
  "#subject/ml": ["#subject/ml/supervised", "#subject/ml/unsupervised"],
  "#subject/ml/supervised": ["#subject/ml/classification", "#subject/ml/regression"],
  "#domain/model": ["#domain/model/bert", "#domain/model/transformer"],
  "#domain/model/transformer": ["#domain/model/bert"]
}
```

```json
{
  "machine learning": "#subject/ml",
  "ml": "#subject/ml",
  "transformer": "#domain/model/transformer",
  "bert": "#domain/model/bert"
}
```

## First-pass scoring policy (draft)

This policy adds metadata as soft boosts on top of fused retrieval scores. The weights are bounded to prevent metadata from overriding strong semantic matches.

| Signal | Default weight | Notes |
| --- | --- | --- |
| Exact tag match | 0.20 | Applies only when document has the tag |
| Expanded tag match | 0.12 | Parent/child/synonym matches |
| Inferred tag confidence | 0.08 | Scales by inference confidence |
| Source type match | 0.10 | Optional prior when inferred |
| Max total metadata boost | 0.35 | Cap for combined metadata signals |

Weights can be applied post-RRF or as a multiplier on fused scores. The recommended default is additive boosts after RRF so metadata does not reduce recall from the primary retrieval signals.

## Immediate decisions and inputs

This plan needs a chosen schema config shape and ontology file location. It also needs a small set of real queries for manual evaluation to tune the initial weights and confidence thresholds.

---

## Proposed class mapping and entry points (no code)

This maps the conceptual classes to current modules and highlights where new interfaces should live.

### Search pipeline integration

- **Entry point:** `src/pmd/search/pipeline.py`
  - Add a metadata inference step before retrieval.
  - Add a tag-aware retrieval channel in parallel with FTS and vector search.
  - Apply metadata boosts after RRF fusion (or as a configurable post-fusion scoring stage).

### Proposed classes and locations

- `pmd.search.metadata.schema.MetadataSchema`
  - Loads schema config for source types, namespaces, and field types.
  - Provides validation and scope checks.

- `pmd.search.metadata.ontology.Ontology`
  - Loads adjacency lists and alias maps.
  - Exposes `expand(tag, hops)` and `resolve_alias(term)`.

- `pmd.search.metadata.inference.QueryMetadataInferencePipeline`
  - Orchestrates inference steps and merges candidates.

- `pmd.search.metadata.inference.InferenceStep` (interface)
  - Implementations:
    - `LexicalTagInference`
    - `OntologyExpansionInference`
    - `EmbeddingTagInference` (optional)

- `pmd.search.metadata.retrieval.TagRetriever`
  - Provides candidates via exact tag match and expansion.
  - Uses existing metadata storage as the initial index.

- `pmd.search.metadata.scoring.MetadataScoringPolicy`
  - Computes metadata boosts from inferred tags and document metadata.
  - Applies caps and per-schema overrides.

### Service layer integration

- **Entry point:** `src/pmd/services/search.py`
  - Instantiates the metadata pipeline and registers it with the search pipeline.
  - Loads metadata config from TOML alongside existing search config.

---

## Config file placement (selected)

### Option B: Dedicated metadata config

- Add `metadata.toml` in a new `config/` directory.
- Add a small pointer in the main config to load it.

Pros: clean separation of concerns.
Cons: additional loading logic.

---

## Minimal evaluation template (non-code)

This is a manual evaluation checklist to use before any learned models:

1) Sample 10–20 real queries (prefer a mix of short/long).
2) For each query, define:
   - expected topic tag(s) (if any)
   - expected source type (if any)
3) Run baseline hybrid search and record:
   - top-5 results
   - which results contain expected tags
4) Run metadata-augmented search and record the same fields.
5) Compare:
   - delta in top-5 tag coverage
   - any obvious regressions in relevance

Store results in a simple table or spreadsheet; no code required.

---

## Implementation requirements (explicit)

- Add a dedicated document metadata table to the schema (design to be finalized by the development team).
- The development team can decide whether a migration is required based on existing deployments.

## Document metadata table design sketch (draft)

This is a non-binding schema sketch to guide discussions. The goal is to store query-time metadata that is separate from source fetch metadata.

### Suggested fields

- `id` (primary key)
- `document_id` (foreign key to `documents.id`, unique)
- `source_type` (string, optional; mirrors collection source type if useful)
- `tags` (JSON array of strings, e.g., `["#subject/ml", "#domain/model/bert"]`)
- `attributes` (JSON object for non-tag fields; e.g., `{ "format": "notebook", "publisher": "arxiv" }`)
- `schema_version` (integer; allows schema evolution if tag namespaces change)
- `created_at`, `updated_at` (timestamps)

### Suggested indexes

- `document_id` (unique)
- `source_type` (optional, if used as a query prior)
- JSON index strategy as supported by the chosen database (if available)

### Data flow notes

- This table should be populated at indexing time from source metadata and/or from upstream document metadata inputs.
- Tag normalization and namespace validation should be applied before insert/update.

## Assumptions (explicit)

- Document metadata used for search will be derived from `source_metadata.extra_metadata` until a dedicated table is introduced.
- Metadata signals are additive boosts, not filters, unless the query explicitly requires otherwise.
- Ontology expansion is limited to 1 hop by default.
