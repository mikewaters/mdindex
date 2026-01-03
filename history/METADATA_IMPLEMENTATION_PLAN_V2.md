# Metadata Implementation Plan v2

## Purpose and scope

Add metadata-aware retrieval and ranking to improve search accuracy without requiring users to know tags. This revision prioritizes:

- Minimal viable implementation before complexity
- Operationally-defined scoring mechanics
- Realistic evaluation with proxy metrics
- Clear fallback behavior when inference fails

## Guiding principles

1. **Start simple**: Binary signals before weighted combinations
2. **Additive complexity**: Each phase must demonstrate value before adding the next
3. **Fail gracefully**: Missing metadata never degrades search quality
4. **Measure before tuning**: Collect data before optimizing weights

---

## Architecture overview

### Core components

1. **MetadataSchema**: Valid tag prefixes and source-type mappings
2. **TagIndex**: Inverted index for fast tag-based retrieval
3. **QueryTagInference**: Deterministic tag detection from query text
4. **MetadataScorer**: Multiplicative boost applied post-RRF

### Key design decisions (resolved)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Boost type | Multiplicative | Prevents metadata from dominating small RRF scores |
| Ontology expansion | Child-to-parent only | Generalization is safe; specialization risks precision loss |
| Default behavior | No metadata boost | Missing/low-confidence inference falls back to pure hybrid search |
| Single tunable weight | Yes (Phase 1-2) | Avoid overfitting without evaluation data |

---

## Phase 0: App-aware metadata extraction

**Goal**: Establish a pluggable system for extracting and normalizing metadata from different authoring applications.

### Problem statement

Documents from different applications use different metadata conventions, even when accessed via the same source type (e.g., filesystem):

| Application | Metadata location | Tag format | Example |
|-------------|-------------------|------------|---------|
| Obsidian | YAML frontmatter in content | Hierarchical: `A/B/C` | `tags: [ml/supervised, domain/nlp]` |
| Drafts | Attached JSON file or API metadata | Flat with delimiter: `A-B-C` | `{"tags": ["ml-supervised", "domain-nlp"]}` |
| Notion export | Frontmatter or properties block | Flat strings | `tags: ["Machine Learning", "NLP"]` |
| Bear | Inline hashtags in content | Hierarchical: `#A/B/C` | `#ml/supervised #domain/nlp` |
| Logseq | Page properties or inline | Hierarchical with `::` | `tags:: ml/supervised, domain/nlp` |

Without normalization, the same semantic tag `#subject/ml/supervised` might be stored as:
- `ml/supervised` (Obsidian)
- `ml-supervised` (Drafts)
- `Machine Learning - Supervised` (Notion)
- `#ml/supervised` (Bear)

### Design: MetadataProfile

Introduce `MetadataProfile` as a configuration layer that sits between `DocumentSource` (how we access) and metadata extraction (how we interpret):

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ DocumentSource  │────▶│ MetadataProfile  │────▶│ Normalized Tags │
│ (access method) │     │ (interpretation) │     │ (canonical form)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
     filesystem              obsidian              #subject/ml
     http                    drafts                #domain/nlp
     entity                  notion
```

### 0.1 MetadataProfile protocol

```python
from dataclasses import dataclass, field
from typing import Protocol, Any

@dataclass(frozen=True)
class ExtractedMetadata:
    """Normalized metadata extracted from a document."""
    tags: list[str]                    # Normalized tag list
    source_tags: list[str]             # Original tags before normalization
    attributes: dict[str, Any]         # Non-tag metadata (author, date, etc.)
    extraction_source: str             # Where metadata was found: "frontmatter", "attached", "inline"


class MetadataProfile(Protocol):
    """Protocol for app-specific metadata extraction and normalization."""

    @property
    def profile_id(self) -> str:
        """Unique identifier for this profile (e.g., 'obsidian', 'drafts')."""
        ...

    def extract_metadata(
        self,
        content: str,
        fetch_metadata: dict[str, Any],
    ) -> ExtractedMetadata:
        """Extract and normalize metadata from document content and fetch metadata.

        Args:
            content: Document text content (may contain frontmatter, inline tags, etc.)
            fetch_metadata: Metadata from FetchResult (may contain attached metadata)

        Returns:
            ExtractedMetadata with normalized tags and attributes.
        """
        ...

    def normalize_tag(self, raw_tag: str) -> str:
        """Convert app-specific tag format to canonical form.

        Args:
            raw_tag: Tag in app-specific format (e.g., "ml-supervised")

        Returns:
            Canonical tag (e.g., "#subject/ml/supervised")
        """
        ...
```

### 0.2 Profile implementations

**ObsidianProfile**:
```python
class ObsidianProfile:
    """Metadata extraction for Obsidian vaults."""

    profile_id = "obsidian"

    def __init__(self, config: ObsidianProfileConfig):
        self.tag_namespace_map = config.tag_namespace_map  # {"ml": "#subject/ml", ...}
        self.hierarchy_separator = "/"

    def extract_metadata(self, content: str, fetch_metadata: dict) -> ExtractedMetadata:
        # 1. Parse YAML frontmatter
        frontmatter = self._parse_frontmatter(content)
        raw_tags = frontmatter.get("tags", [])

        # 2. Also extract inline #tags from content
        inline_tags = self._extract_inline_tags(content)
        raw_tags.extend(inline_tags)

        # 3. Normalize all tags
        normalized = [self.normalize_tag(t) for t in raw_tags]

        return ExtractedMetadata(
            tags=normalized,
            source_tags=raw_tags,
            attributes=self._extract_attributes(frontmatter),
            extraction_source="frontmatter" if frontmatter else "inline",
        )

    def normalize_tag(self, raw_tag: str) -> str:
        # Remove leading # if present
        tag = raw_tag.lstrip("#")

        # Map top-level namespace if configured
        parts = tag.split("/")
        if parts[0] in self.tag_namespace_map:
            parts[0] = self.tag_namespace_map[parts[0]]

        # Ensure canonical format: #namespace/path
        result = "/".join(parts)
        if not result.startswith("#"):
            result = f"#{result}"
        return result.lower()
```

**DraftsProfile**:
```python
class DraftsProfile:
    """Metadata extraction for Drafts app exports."""

    profile_id = "drafts"

    def __init__(self, config: DraftsProfileConfig):
        self.hierarchy_separator = "-"  # Drafts uses hyphens
        self.tag_namespace_map = config.tag_namespace_map

    def extract_metadata(self, content: str, fetch_metadata: dict) -> ExtractedMetadata:
        # Drafts stores metadata in fetch_metadata (from API or sidecar JSON)
        raw_tags = fetch_metadata.get("tags", [])

        normalized = [self.normalize_tag(t) for t in raw_tags]

        return ExtractedMetadata(
            tags=normalized,
            source_tags=raw_tags,
            attributes={
                "created": fetch_metadata.get("created_at"),
                "modified": fetch_metadata.get("modified_at"),
                "flagged": fetch_metadata.get("flagged", False),
            },
            extraction_source="attached",
        )

    def normalize_tag(self, raw_tag: str) -> str:
        # Convert hyphen hierarchy to slash hierarchy
        # e.g., "ml-supervised-classification" -> "#subject/ml/supervised/classification"
        parts = raw_tag.split("-")

        # Map top-level namespace
        if parts[0] in self.tag_namespace_map:
            parts[0] = self.tag_namespace_map[parts[0]]

        result = "/".join(parts)
        if not result.startswith("#"):
            result = f"#{result}"
        return result.lower()
```

**GenericProfile** (fallback):
```python
class GenericProfile:
    """Default profile for documents without app-specific handling."""

    profile_id = "generic"

    def extract_metadata(self, content: str, fetch_metadata: dict) -> ExtractedMetadata:
        # Try frontmatter first, then fall back to fetch_metadata
        frontmatter = self._parse_frontmatter(content)
        raw_tags = frontmatter.get("tags", fetch_metadata.get("tags", []))

        # Minimal normalization: lowercase, ensure # prefix
        normalized = [self._minimal_normalize(t) for t in raw_tags]

        return ExtractedMetadata(
            tags=normalized,
            source_tags=raw_tags,
            attributes={},
            extraction_source="frontmatter" if frontmatter else "attached",
        )

    def _minimal_normalize(self, tag: str) -> str:
        tag = tag.lstrip("#").lower()
        return f"#{tag}"
```

### 0.3 Profile configuration

```toml
[metadata.profiles.obsidian]
enabled = true
# Map app tag prefixes to canonical namespaces
tag_namespace_map = { "ml" = "#subject/ml", "nlp" = "#domain/nlp", "project" = "#effort" }
# Where to look for metadata
extraction_sources = ["frontmatter", "inline"]

[metadata.profiles.drafts]
enabled = true
tag_namespace_map = { "ml" = "#subject/ml", "nlp" = "#domain/nlp" }
hierarchy_separator = "-"
extraction_sources = ["attached"]

[metadata.profiles.generic]
# Fallback profile used when no specific profile matches
enabled = true
```

### 0.4 Profile selection

Profiles can be selected:

1. **Per-collection** (explicit): Collection config specifies `metadata_profile = "obsidian"`
2. **By detection** (heuristic): Auto-detect based on content patterns:
   - `.obsidian/` directory present → Obsidian
   - Drafts API metadata schema → Drafts
   - Default → Generic

```python
class MetadataProfileRegistry:
    def __init__(self, profiles: dict[str, MetadataProfile]):
        self.profiles = profiles
        self.default_profile = profiles.get("generic", GenericProfile())

    def get_profile(self, profile_id: str | None) -> MetadataProfile:
        if profile_id and profile_id in self.profiles:
            return self.profiles[profile_id]
        return self.default_profile

    def detect_profile(self, source_config: SourceConfig) -> MetadataProfile:
        """Auto-detect appropriate profile based on source characteristics."""
        # Check for explicit profile in config
        explicit = source_config.get("metadata_profile")
        if explicit:
            return self.get_profile(explicit)

        # Heuristic detection
        uri = source_config.uri
        if ".obsidian" in uri or source_config.get("vault_path"):
            return self.get_profile("obsidian")

        # Default fallback
        return self.default_profile
```

### 0.5 Integration with indexing pipeline

Modify the indexing pipeline to use profiles:

```python
# In src/pmd/services/indexing.py

class IndexingService:
    def __init__(
        self,
        ...,
        profile_registry: MetadataProfileRegistry,
    ):
        self.profile_registry = profile_registry

    async def index_document(
        self,
        ref: DocumentReference,
        source: DocumentSource,
        collection: Collection,
    ) -> Document:
        # Fetch content
        result = await source.fetch_content(ref)

        # Get appropriate metadata profile
        profile = self.profile_registry.get_profile(
            collection.config.get("metadata_profile")
        )

        # Extract and normalize metadata
        extracted = profile.extract_metadata(result.content, result.metadata)

        # Store document with normalized metadata
        doc = self.document_repo.upsert(
            collection_id=collection.id,
            path=ref.path,
            title=self._extract_title(result.content, ref),
            body=result.content,
        )

        # Store normalized tags in document_metadata table
        self.metadata_repo.upsert(DocumentMetadata(
            document_id=doc.id,
            tags=extracted.tags,
            source_type=collection.source_type,
        ))

        return doc
```

### 0.6 Deliverables

- [ ] `MetadataProfile` protocol in `src/pmd/search/metadata/profiles.py`
- [ ] `ObsidianProfile` implementation
- [ ] `DraftsProfile` implementation
- [ ] `GenericProfile` fallback implementation
- [ ] `MetadataProfileRegistry` with detection logic
- [ ] YAML frontmatter parser utility
- [ ] Inline tag extractor utility
- [ ] Integration with `IndexingService`
- [ ] Collection config extension for `metadata_profile` field
- [ ] Profile configuration in TOML

### 0.7 Testing checklist

- [ ] Obsidian vault with frontmatter tags extracts correctly
- [ ] Obsidian vault with inline `#tags` extracts correctly
- [ ] Drafts export with attached JSON extracts correctly
- [ ] Tag normalization: `ml-supervised` → `#subject/ml/supervised`
- [ ] Tag normalization: `ML/Supervised` → `#subject/ml/supervised`
- [ ] Mixed sources in same collection use correct profiles
- [ ] Unknown profile falls back to generic
- [ ] Empty/missing metadata produces empty tag list (not error)

---

## Phase 1: Exact tag matching with single boost

**Goal**: Prove that metadata improves search quality with minimal machinery.

**Prerequisite**: Phase 0 delivers working tag extraction for at least one profile.

### 1.1 Document metadata table

Create a dedicated table for query-time metadata (separate from fetch metadata):

```sql
CREATE TABLE document_metadata (
    id INTEGER PRIMARY KEY,
    document_id INTEGER UNIQUE NOT NULL REFERENCES documents(id),
    tags TEXT NOT NULL DEFAULT '[]',  -- JSON array of normalized tags
    source_type TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Inverted index for tag lookups
CREATE TABLE document_tags (
    document_id INTEGER NOT NULL REFERENCES documents(id),
    tag TEXT NOT NULL,
    PRIMARY KEY (document_id, tag)
);
CREATE INDEX idx_document_tags_tag ON document_tags(tag);
```

**Rationale**: Separate junction table enables fast tag-based retrieval without JSON parsing.

### 1.2 Tag extraction at index time

During document indexing, extract tags from:
- Explicit frontmatter/metadata fields
- Configurable extraction patterns per source type

```python
@dataclass
class DocumentMetadata:
    document_id: int
    tags: list[str]  # Normalized, e.g., ["#subject/ml", "#domain/model/bert"]
    source_type: str | None
```

### 1.3 Query tag inference (lexical only)

Simple dictionary-based tag detection:

```python
class LexicalTagMatcher:
    def __init__(self, alias_map: dict[str, str]):
        """alias_map: {"machine learning": "#subject/ml", "ml": "#subject/ml", ...}"""
        self.alias_map = alias_map

    def infer_tags(self, query: str) -> list[str]:
        """Returns list of tags found in query via exact/substring match."""
        query_lower = query.lower()
        matched = []
        for alias, tag in self.alias_map.items():
            if alias in query_lower:
                matched.append(tag)
        return list(set(matched))  # Deduplicate
```

No confidence scores in Phase 1. Tags are either matched or not.

### 1.4 Scoring integration

Apply metadata boost as a multiplicative factor after RRF fusion:

```python
def apply_metadata_boost(
    results: list[RankedResult],
    query_tags: list[str],
    boost_factor: float = 1.15,  # Single tunable parameter
) -> list[RankedResult]:
    """
    Multiply score by boost_factor for each matching tag.

    Example: 2 matching tags with boost_factor=1.15
    final_score = rrf_score * (1.15 ** 2) = rrf_score * 1.32
    """
    boosted = []
    for result in results:
        doc_tags = get_document_tags(result.document_id)
        match_count = len(set(query_tags) & set(doc_tags))

        if match_count > 0:
            multiplier = boost_factor ** match_count
            # Cap total boost to prevent runaway scores
            multiplier = min(multiplier, 2.0)
            new_score = result.score * multiplier
        else:
            new_score = result.score

        boosted.append(result.with_score(new_score))

    boosted.sort(key=lambda r: r.score, reverse=True)
    return boosted
```

**Key properties**:
- Multiplicative: works with any RRF score magnitude
- Capped: max 2x boost prevents metadata from completely overriding semantics
- Single parameter: `boost_factor` is the only tunable weight
- No penalty: missing tags never reduce score

### 1.5 Pipeline integration point

Insert after RRF fusion, before reranking:

```
Query -> [Expansion] -> Parallel Search -> RRF Fusion -> **Metadata Boost** -> [Reranking] -> Normalize -> Results
```

### 1.6 Configuration (minimal)

```toml
[metadata]
enabled = true
boost_factor = 1.15
max_boost = 2.0

[metadata.aliases]
file = "config/tag_aliases.json"
```

### 1.7 Deliverables

- [ ] `document_metadata` and `document_tags` tables in schema
- [ ] `DocumentMetadataRepository` with CRUD and tag lookup
- [ ] `LexicalTagMatcher` for query-time inference
- [ ] `apply_metadata_boost()` function in scoring module
- [ ] Integration into `HybridSearchPipeline`
- [ ] Initial alias file with 20-30 common terms

### 1.8 Evaluation checkpoint

Before proceeding to Phase 2, collect:

1. **Manual relevance judgments**: 50 queries with top-5 results rated (relevant/not relevant)
2. **A/B comparison**: Same queries with metadata boost enabled vs disabled
3. **Success criteria**: Metadata-boosted results should show:
   - No regressions in top-1 accuracy
   - Improvement in queries containing known tag aliases

---

## Phase 2: Ontology parent matching

**Goal**: Improve recall by matching documents tagged with child concepts when query matches parent.

**Prerequisite**: Phase 1 evaluation shows positive signal.

### 2.1 Ontology structure

Simple parent-child relationships stored as adjacency list:

```json
{
  "#subject/ml": {
    "children": ["#subject/ml/supervised", "#subject/ml/unsupervised"],
    "description": "Machine learning and statistical learning methods"
  },
  "#subject/ml/supervised": {
    "children": ["#subject/ml/classification", "#subject/ml/regression"],
    "description": "Learning from labeled examples"
  }
}
```

### 2.2 Asymmetric expansion (child-to-parent only)

When a query matches tag T, also consider documents tagged with any ancestor of T:

```python
class Ontology:
    def __init__(self, adjacency: dict[str, dict]):
        self.adjacency = adjacency
        self._parent_map = self._build_parent_map()

    def get_ancestors(self, tag: str, max_hops: int = 1) -> list[str]:
        """Return parent tags up to max_hops levels."""
        ancestors = []
        current = tag
        for _ in range(max_hops):
            parent = self._parent_map.get(current)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors

    def expand_for_matching(self, query_tags: list[str]) -> dict[str, float]:
        """
        Expand query tags for document matching.
        Returns {tag: weight} where weight < 1.0 for expanded tags.
        """
        expanded = {}
        for tag in query_tags:
            expanded[tag] = 1.0  # Exact match: full weight
            for ancestor in self.get_ancestors(tag, max_hops=1):
                if ancestor not in expanded:
                    expanded[ancestor] = 0.7  # Parent match: reduced weight
        return expanded
```

**Note**: We do NOT expand parent-to-children. If query matches `#subject/ml`, we don't automatically boost `#subject/ml/supervised` documents. This prevents precision loss.

### 2.3 Updated scoring

```python
def apply_metadata_boost_v2(
    results: list[RankedResult],
    query_tags: dict[str, float],  # {tag: weight} from ontology expansion
    boost_factor: float = 1.15,
    max_boost: float = 2.0,
) -> list[RankedResult]:
    boosted = []
    for result in results:
        doc_tags = set(get_document_tags(result.document_id))

        # Sum weighted matches
        match_weight = sum(
            weight for tag, weight in query_tags.items()
            if tag in doc_tags
        )

        if match_weight > 0:
            # Exponential boost scaled by match weight
            multiplier = boost_factor ** match_weight
            multiplier = min(multiplier, max_boost)
            new_score = result.score * multiplier
        else:
            new_score = result.score

        boosted.append(result.with_score(new_score))

    boosted.sort(key=lambda r: r.score, reverse=True)
    return boosted
```

### 2.4 Configuration addition

```toml
[metadata.ontology]
file = "config/tag_ontology.json"
max_hops = 1
parent_weight = 0.7
```

### 2.5 Deliverables

- [ ] `Ontology` class with parent lookup and expansion
- [ ] Updated scoring function with weighted matches
- [ ] Ontology file with initial tag hierarchy (2-3 namespaces)
- [ ] Tests for expansion behavior

### 2.6 Evaluation checkpoint

Compare against Phase 1 baseline:

1. **Recall improvement**: Do relevant documents with child tags now appear for parent-level queries?
2. **Precision preservation**: No increase in irrelevant results in top-5
3. **Success criteria**: Measurable recall improvement without precision regression

---

## Phase 3: Tag-based retrieval channel

**Goal**: Use tags as a retrieval signal, not just a post-hoc boost.

**Prerequisite**: Phase 2 demonstrates value; alias/ontology coverage is sufficient.

### 3.1 Tag retrieval as parallel channel

Add tag-based retrieval alongside FTS and vector search:

```python
class TagRetriever:
    def __init__(self, metadata_repo: DocumentMetadataRepository):
        self.metadata_repo = metadata_repo

    def retrieve(
        self,
        tags: list[str],
        limit: int,
        collection_id: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve documents matching any of the given tags."""
        doc_ids = self.metadata_repo.find_documents_with_tags(
            tags, collection_id, limit=limit * 2
        )
        # Convert to SearchResult with tag-based score
        results = []
        for doc_id, match_count in doc_ids:
            doc = self.metadata_repo.get_document(doc_id)
            results.append(SearchResult(
                filepath=doc.filepath,
                title=doc.title,
                body=doc.body,
                score=match_count / len(tags),  # Normalize by query tag count
                source=SearchSource.TAG,
            ))
        return results[:limit]
```

### 3.2 RRF integration

Include tag results in fusion with lower weight:

```python
# In HybridSearchPipeline._parallel_search
result_lists = []

# FTS results (weight: 1.0)
result_lists.append(fts_results)

# Vector results (weight: 1.0)
result_lists.append(vec_results)

# Tag results (weight: 0.5) - lower weight prevents tag-only dominance
if query_tags:
    tag_results = self.tag_retriever.retrieve(query_tags, limit, collection_id)
    result_lists.append(tag_results)

# Adjust RRF weights
weights = [1.0, 1.0, 0.5] if query_tags else [1.0, 1.0]
fused = reciprocal_rank_fusion(result_lists, weights=weights)
```

### 3.3 Deliverables

- [ ] `TagRetriever` class
- [ ] `SearchSource.TAG` enum value
- [ ] RRF integration with configurable tag channel weight
- [ ] Provenance tracking for tag-sourced results

### 3.4 Evaluation checkpoint

1. **New document discovery**: Are tag-retrieved documents appearing that weren't found by FTS/vector?
2. **Fusion quality**: Does adding the tag channel improve or hurt overall ranking?
3. **Latency impact**: Tag retrieval should add <10ms to search latency

---

## Phase 4: Source-type inference (optional)

**Goal**: Use source type as a weak prior when query implies document type.

**Prerequisite**: Phases 1-3 are stable; source-type distribution is meaningful.

### 4.1 Source-type signals

Some queries imply expected source types:

| Query pattern | Implied source type |
|---------------|---------------------|
| "how to...", "tutorial" | filesystem (documentation) |
| "API reference", "endpoint" | http (API docs) |
| "definition of...", "what is" | entity |

### 4.2 Implementation sketch

```python
class SourceTypeMatcher:
    PATTERNS = {
        "filesystem": ["how to", "tutorial", "guide", "example"],
        "http": ["api", "endpoint", "reference", "documentation"],
        "entity": ["definition", "what is", "concept"],
    }

    def infer_source_type(self, query: str) -> str | None:
        query_lower = query.lower()
        for source_type, patterns in self.PATTERNS.items():
            if any(p in query_lower for p in patterns):
                return source_type
        return None
```

Apply as a mild multiplicative boost (1.05-1.10) for matching source types.

### 4.3 Evaluation

This is speculative. Only implement if:
1. Source type distribution is uneven (some types are rare)
2. Query patterns reliably predict source type preference
3. Phase 1-3 evaluation suggests source-type signal would help

---

## Phase 5: Embedding-based tag inference (deferred)

**Goal**: Infer tags from query semantics when lexical matching fails.

**Prerequisites**:
- Phases 1-3 are production-ready
- Alias coverage is high (>80% of common queries match)
- Evaluation data exists to measure improvement

### 5.1 Approach

Embed tag descriptions (not names) and compare to query embeddings:

```python
class EmbeddingTagInference:
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        tag_descriptions: dict[str, str],
        threshold: float = 0.75,  # High threshold to avoid false positives
    ):
        self.embeddings = self._embed_descriptions(tag_descriptions)
        self.threshold = threshold

    def infer_tags(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        query_embedding = self.embedding_generator.embed_query(query)
        similarities = cosine_similarity(query_embedding, self.embeddings)

        candidates = []
        for tag, sim in similarities.items():
            if sim >= self.threshold:
                candidates.append((tag, sim))

        return sorted(candidates, key=lambda x: -x[1])[:top_k]
```

### 5.2 Risks

- Embedding space mismatch between queries and tag descriptions
- False positives for semantically similar but incorrect tags
- Latency cost of additional embedding computation

### 5.3 Mitigation

- Use tag descriptions, not tag names
- High similarity threshold (start at 0.75, tune down if needed)
- Cache tag embeddings; only compute query embedding once
- Treat embedding-inferred tags as low-confidence (weight 0.5)

---

## Evaluation framework

### Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Top-1 accuracy | % of queries where top result is relevant | No regression |
| MRR@5 | Mean reciprocal rank of first relevant result in top 5 | +5% improvement |
| Tag coverage | % of queries where inference finds >= 1 tag | >60% |
| Precision@5 | % of top-5 results that are relevant | No regression |

### Data collection

1. **Query log**: Record all search queries (anonymized if needed)
2. **Click data**: If available, use clicks as implicit relevance signal
3. **Manual judgments**: Periodic review of random query samples

### Evaluation cadence

- **Phase gate**: 50-query evaluation before each phase
- **Regression testing**: 20-query smoke test after each code change
- **Quarterly review**: Full 100-query evaluation with fresh samples

---

## Configuration reference

```toml
[metadata]
enabled = true

[metadata.scoring]
boost_factor = 1.15          # Base boost per matching tag
max_boost = 2.0              # Maximum total multiplier
parent_weight = 0.7          # Weight for ontology parent matches

[metadata.inference]
method = "lexical"           # "lexical" | "lexical+ontology" | "lexical+ontology+embedding"

[metadata.inference.lexical]
alias_file = "config/tag_aliases.json"

[metadata.ontology]
file = "config/tag_ontology.json"
max_hops = 1

[metadata.retrieval]
enabled = false              # Enable in Phase 3
channel_weight = 0.5         # Weight in RRF fusion

[metadata.inference.embedding]
enabled = false              # Enable in Phase 5
threshold = 0.75
top_k = 3
description_file = "config/tag_descriptions.json"
```

---

## Module locations

| Component | Location |
|-----------|----------|
| Metadata profiles | `src/pmd/search/metadata/profiles.py` |
| Profile registry | `src/pmd/search/metadata/registry.py` |
| Frontmatter parser | `src/pmd/search/metadata/parsers.py` |
| Schema/repository | `src/pmd/store/document_metadata.py` |
| Tag inference | `src/pmd/search/metadata/inference.py` |
| Ontology | `src/pmd/search/metadata/ontology.py` |
| Scoring | `src/pmd/search/metadata/scoring.py` |
| Tag retrieval | `src/pmd/search/metadata/retrieval.py` |
| Config | `src/pmd/core/config.py` (extend existing) |

---

## Migration notes

- Phase 1 requires new tables; provide migration script
- Backfill `document_tags` from existing documents at migration time
- No breaking changes to existing search API

---

## Summary: Phased approach

| Phase | Focus | Key deliverable | Prerequisite |
|-------|-------|-----------------|--------------|
| 0 | App-aware extraction | MetadataProfile system | None |
| 1 | Exact tag matching | Single-parameter boost | Phase 0 extracts tags |
| 2 | Ontology parents | Asymmetric expansion | Phase 1 shows value |
| 3 | Tag retrieval channel | Tags in RRF fusion | Phase 2 stable |
| 4 | Source-type inference | Mild type-based boost | Evidence of need |
| 5 | Embedding inference | Semantic tag matching | High alias coverage |

Each phase is independently valuable and can be shipped incrementally.
