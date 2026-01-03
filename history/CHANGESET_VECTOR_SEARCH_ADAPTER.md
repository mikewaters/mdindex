# Proposed Changeset: Vector Search Adapter (Option 1)
 # Introducing Tagging & Ontology into Vector Search

  ## Executive Summary

  A robust tagging + ontology layer should complement, not replace, vector search. The most reliable pattern is a hybrid retrieval + fusion approach: keep embeddings content-driven, store tags in a structured
  metadata store, and use ontology-aware inference to filter or boost candidates at query time. This preserves the strengths of dense retrieval while adding symbolic control, explainability, and precision.

  ———

  ## Goals

  - Precision: promote results that match the user’s intent via tags and ontology structure.
  - Recall: expand query intent to synonyms/parents so results aren’t missed.
  - Control: allow explicit constraints (filters) without retraining embedding models.
  - Explainability: “why this result?” can cite tag/ontology matches.

  ———

  ## Core Concepts

  ### 1) Tag Normalization

  A deterministic normalization pipeline is mandatory:

  - Canonical casing and separators (e.g., lowercase, hyphen rules).
  - Alias mapping (e.g., ml → machine-learning).
  - Remove noise (stopwords, extremely generic tags).

  ### 2) Ontology Graph

  Represent tags and relations as a graph:

  - Nodes: tags or concepts
  - Edges: is-a (hierarchy), related-to, synonym
  - Control traversal depth for expansion (e.g., parents up to depth=2)

  ### 3) Two-Channel Retrieval

  Use parallel retrieval channels:

  - Dense: vector similarity search.
  - Symbolic: tag/ontology lookup (inverted index or metadata DB).
    Fusion produces final results.

  ———

  ## Design Approach

  ### A) Index-Time Enrichment

  During indexing:

  1. Extract raw tags (frontmatter, inline).
  2. Normalize tags (canonical form).
  3. Expand tags using ontology:
      - Add parent tags.
      - Add synonyms (optionally flagged as “soft”).
  4. Store:
      - Raw tags
      - Normalized tags
      - Expanded tags (distinguish source)

  Benefit: query-time matching becomes fast and deterministic.

  ### B) Query-Time Inference & Expansion

  At search time:

  1. Extract candidate tags from query (lexical matcher, synonyms).
  2. Expand tags via ontology rules (controlled depth).
  3. Decide how to use them:
      - Hard filter (strict): only include docs with matching tags.
      - Soft boost (recommended): adjust scores using match weights.

  ### C) Fusion & Scoring

  Combine results with weighted fusion:

  - RRF (Reciprocal Rank Fusion) is a good default.
  - Tag matches can add a score bonus or alter rank order.
  - Keep the semantic signal dominant unless tags are highly reliable.

  ———

  ## Scoring Strategy (Practical)

  Use weighted features:

  1. Vector similarity (primary)
  2. Exact tag match count
  3. Parent tag match count (lower weight)
  4. Synonym match (lower weight)
  5. Ontology distance (penalize larger hops)

  Example (conceptual):

  - Exact match = +0.15
  - Parent match = +0.08
  - Synonym match = +0.05
  - Each ontology hop = -0.02

  ———

  ## Where It Fits in Your Pipeline

  Given your current structure:

  - Embeddings store remains unchanged.
  - Document metadata store holds normalized and expanded tags.
  - Search pipeline performs:
      1. Vector search
      2. Tag-based retrieval (from metadata)
      3. Fusion
      4. Optional metadata boost

  ———

  ## Optional: Embedding Augmentation

  If you want tags to influence embeddings:

  - Prepend a compact, normalized tag summary to the content before embedding.
  - Use a stable template like:
    Tags: tag1, tag2, tag3.
  - Do this only if tag quality is high and consistent.

  Caution: mixing noisy tags into embeddings can degrade semantic similarity.

  ———

  ## Operational Considerations

  ### Governance

  - Curate a controlled vocabulary.
  - Enforce canonical tags.
  - Track unused/rare tags.

  ### Monitoring & Evaluation

  - Measure search quality changes with offline evaluation.
  - Track boosting effects (precision gains vs. missed recall).

  ### Explainability

  - Surface tag matches and ontology expansions in “why this result.”

  ———

  ## Recommended Next Steps

  1. Define a minimal ontology: 20–50 top concepts + hierarchy.
  2. Implement tag normalization and synonym mapping.
  3. Add tag expansion at index time (parents + synonyms).
  4. Add tag-based boosting in query-time fusion.
  5. Validate search quality with a small labeled test set.

———

## Summary
Introduce a **VectorSearchRepository adapter** that implements `SearchRepository[list[float]]` and delegates to `EmbeddingRepository.search_vectors()`. This enables polymorphic search backends without modifying `EmbeddingRepository` itself.

## New file
- `src/pmd/store/vector_search.py`
  - Defines `VectorSearchRepository(SearchRepository[list[float]])`
  - Constructor accepts `EmbeddingRepository` (or `Database` to build one)
  - Implements `search(query_embedding, limit, collection_id=None, min_score=0.0)` by calling `search_vectors()`

## Modified files
1) `src/pmd/store/__init__.py`
   - Export `VectorSearchRepository` (optional but recommended).

2) `src/pmd/search/pipeline.py` (optional, depending on how you want to wire backends)
   - If you want full polymorphism in the pipeline, add a path to accept a list of `SearchRepository` instances.
   - Otherwise leave as-is and use the adapter in calling code.

3) `README.md` / `ARCHITECTURE.md` (optional)
   - Document the new adapter as the recommended way to plug vector search into polymorphic pipelines.

## Tests (minimal)
- Add a unit test to ensure the adapter delegates correctly:
  - Given a mock `EmbeddingRepository`, verify `search()` calls `search_vectors()` with the same parameters.

## Rationale
- Keeps `EmbeddingRepository` focused on storage + vector operations.
- Provides a clean polymorphic interface for pipelines without forcing unrelated methods into a shared base.
- Enables multiple backends/configurations by instantiating multiple adapters with different repositories or settings.

## Notes
- This changeset does **not** alter indexing behavior or embeddings schema.
- It is compatible with your current pipeline while opening a path to a multi-backend pipeline later.
