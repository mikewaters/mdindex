# Search Module Architecture

**Location:** `src/pmd/search/`

Search pipeline algorithms and orchestration.

## Files and Key Abstractions

### `pipeline.py`

**`SearchPipelineConfig`** - Pipeline parameters
- Weights, RRF k, rerank candidates
- Feature flags for expansion/reranking

**`HybridSearchPipeline`** - Multi-stage search orchestration

Pipeline stages:
1. Query expansion (optional)
2. Parallel FTS + vector search
3. Reciprocal Rank Fusion
4. LLM reranking (optional)
5. Position-aware blending
6. Score normalization

### `fusion.py`

**`reciprocal_rank_fusion()`** - Combine ranked lists

Formula: `RRF = Σ(weight / (k + rank + 1)) + bonuses`

Features:
- Top-rank bonuses (+0.05 for rank 1, +0.02 for ranks 2-3)
- Provenance tracking (fts_rank, vec_rank, sources_count)
- Weighted result lists

### `scoring.py`

**`normalize_scores()`** - Max-normalization to [0, 1]

**`blend_scores()`** - Position-aware blending

| Rank | RRF Weight | Reranker Weight |
|------|------------|-----------------|
| 1-3 | 75% | 25% |
| 4-10 | 60% | 40% |
| 11+ | 40% | 60% |

### `chunking.py`

**`chunk_document()`** - Split documents for embedding

Splitting preference:
1. Paragraph breaks (`\n\n`)
2. Sentence ends
3. Line breaks (`\n`)
4. Spaces

### `text.py`

**`is_indexable()`** - Document quality filter
- Rejects empty documents
- Rejects title-only documents
