# Metadata Module Refactoring Plan

## Problem Statement

The metadata functionality is currently split across two locations:
- `pmd.sources.metadata` - loader-time extraction (profiles, parsers, registry)
- `pmd.search.metadata` - query-time inference (ontology, scoring, retrieval)

This creates issues:
1. **Ontology lives only in search** but could benefit extraction (validation, hierarchy expansion)
2. **Core types are in sources** with awkward re-exports via shims in search
3. **No single source of truth** for what tags/concepts the system recognizes
4. **Test imports are confusing** - most tests import from `pmd.search.metadata.*` even for extraction

## Current Architecture

```
pmd/
├── sources/
│   └── metadata/           # Loader-time extraction
│       ├── profiles.py     # ExtractedMetadata, MetadataProfile (CORE TYPES)
│       ├── parsers.py      # YAML frontmatter, inline #tags (CORE PARSING)
│       ├── implementations.py  # GenericProfile, ObsidianProfile, DraftsProfile
│       └── registry.py     # Profile registry, auto-detection
│
└── search/
    └── metadata/           # Query-time inference
        ├── ontology.py     # Parent-child tag hierarchy (CORE ONTOLOGY)
        ├── inference.py    # LexicalTagMatcher, TagMatch
        ├── retrieval.py    # TagRetriever (RRF channel)
        ├── scoring.py      # apply_metadata_boost, apply_metadata_boost_v2
        ├── data/
        │   ├── tag_ontology.json
        │   └── tag_aliases.json
        │
        └── [shim files re-exporting from sources]
            ├── profiles.py
            ├── parsers.py
            ├── implementations.py
            └── registry.py
```

## Proposed Architecture

```
pmd/
├── metadata/                   # NEW: First-class metadata module
│   ├── __init__.py
│   ├── types.py               # ExtractedMetadata, MetadataProfile protocol
│   ├── parsing.py             # Frontmatter & inline tag extraction
│   ├── ontology.py            # Tag hierarchy (parent-child relationships)
│   ├── aliases.py             # Tag alias mappings
│   ├── normalization.py       # Tag normalization utilities
│   └── data/
│       ├── tag_ontology.json
│       └── tag_aliases.json
│
├── sources/
│   └── metadata/              # Remains: Source-specific extraction
│       ├── __init__.py        # Re-exports + extraction profiles
│       ├── profiles/          # Concrete extraction implementations
│       │   ├── __init__.py
│       │   ├── generic.py     # GenericProfile
│       │   ├── obsidian.py    # ObsidianProfile
│       │   └── drafts.py      # DraftsProfile
│       └── registry.py        # Profile detection & registry
│
└── search/
    └── metadata/              # Remains: Search-specific inference
        ├── __init__.py        # Re-exports + search components
        ├── inference.py       # LexicalTagMatcher, TagMatch
        ├── retrieval.py       # TagRetriever (RRF channel)
        └── scoring.py         # Boost functions & config
```

## What Goes Where

### `pmd.metadata` (NEW - Core/Shared)

| Component | Rationale |
|-----------|-----------|
| `ExtractedMetadata` | Core type used by both extraction and search |
| `MetadataProfile` protocol | Defines extraction interface, import-free |
| `Ontology`, `OntologyNode` | Tag hierarchy used by search boosting AND could validate extraction |
| `parse_frontmatter()` | Core parsing, not source-specific |
| `extract_inline_tags()` | Core parsing, not source-specific |
| `tag_ontology.json` | Single source of truth for tag hierarchy |
| `tag_aliases.json` | Canonical alias mappings |
| Tag normalization utilities | Shared logic for tag cleanup |

### `pmd.sources.metadata` (Remains - Extraction-Specific)

| Component | Rationale |
|-----------|-----------|
| `GenericProfile` | Concrete extraction implementation |
| `ObsidianProfile` | App-specific (Obsidian vault conventions) |
| `DraftsProfile` | App-specific (Drafts app conventions) |
| `MetadataProfileRegistry` | Source detection/selection logic |
| `get_default_profile_registry()` | Factory function |
| Path-based profile detection | Source-specific heuristics |

### `pmd.search.metadata` (Remains - Search-Specific)

| Component | Rationale |
|-----------|-----------|
| `LexicalTagMatcher` | Query-time tag inference |
| `TagMatch` | Query inference result type |
| `TagRetriever` | RRF retrieval channel |
| `apply_metadata_boost()` | Search result scoring |
| `apply_metadata_boost_v2()` | Weighted scoring with ontology |
| `MetadataBoostConfig` | Search pipeline configuration |
| `BoostResult`, `WeightedBoostResult` | Scoring result types |

## Migration Strategy

### Phase 1: Create `pmd.metadata` core module
1. Create `src/pmd/metadata/` directory structure
2. Move `ExtractedMetadata` and `MetadataProfile` to `types.py`
3. Move parsing utilities to `parsing.py`
4. Move `Ontology` and related code to `ontology.py`
5. Create `aliases.py` for tag alias management
6. Move data files (`tag_ontology.json`, `tag_aliases.json`)

### Phase 2: Update `pmd.sources.metadata`
1. Update imports to use `pmd.metadata` for core types
2. Reorganize profiles into `profiles/` subdirectory
3. Update `__init__.py` to re-export from `pmd.metadata` for backward compat
4. Update registry to use new locations

### Phase 3: Update `pmd.search.metadata`
1. Update imports to use `pmd.metadata` for Ontology and core types
2. Remove shim files (`profiles.py`, `parsers.py`, `implementations.py`, `registry.py`)
3. Update `__init__.py` with proper re-exports for backward compat
4. Update `inference.py` to load aliases from `pmd.metadata`

### Phase 4: Update consumers
1. Update tests to import from canonical locations
2. Update `pmd.store.document_metadata` if it imports metadata types
3. Update `pmd.search.pipeline` to use new import paths
4. Run full test suite to verify

### Phase 5: Cleanup
1. Remove deprecated shim files
2. Update docstrings and comments
3. Add deprecation warnings for old import paths (optional)

## Import Structure After Refactor

```python
# Core types and ontology (canonical)
from pmd.metadata import (
    ExtractedMetadata,
    MetadataProfile,
    Ontology,
    OntologyNode,
    load_ontology,
    load_default_ontology,
    parse_frontmatter,
    extract_inline_tags,
)

# Source-specific extraction
from pmd.sources.metadata import (
    GenericProfile,
    ObsidianProfile,
    DraftsProfile,
    MetadataProfileRegistry,
    get_default_profile_registry,
)

# Search-specific inference
from pmd.search.metadata import (
    LexicalTagMatcher,
    TagMatch,
    TagRetriever,
    apply_metadata_boost,
    apply_metadata_boost_v2,
    MetadataBoostConfig,
)
```

## Backward Compatibility

Both `pmd.sources.metadata` and `pmd.search.metadata` will re-export core types from `pmd.metadata` to maintain backward compatibility:

```python
# pmd/sources/metadata/__init__.py
from pmd.metadata import ExtractedMetadata, MetadataProfile, ...
from .profiles import GenericProfile, ObsidianProfile, DraftsProfile
from .registry import MetadataProfileRegistry, get_default_profile_registry

# pmd/search/metadata/__init__.py
from pmd.metadata import ExtractedMetadata, MetadataProfile, Ontology, ...
from .inference import LexicalTagMatcher, TagMatch, ...
from .scoring import apply_metadata_boost, ...
```

## Files to Create

```
src/pmd/metadata/
├── __init__.py
├── types.py           # ExtractedMetadata, MetadataProfile
├── parsing.py         # parse_frontmatter, extract_inline_tags, FrontmatterResult
├── ontology.py        # Ontology, OntologyNode, load_ontology, load_default_ontology
├── aliases.py         # TagAliases class, load_aliases, load_default_aliases
├── normalization.py   # normalize_tag, extract_tags_from_field
└── data/
    ├── tag_ontology.json  (move from search/metadata/data/)
    └── tag_aliases.json   (move from search/metadata/data/)
```

## Files to Modify

1. `src/pmd/sources/metadata/__init__.py` - update imports
2. `src/pmd/sources/metadata/implementations.py` - update imports
3. `src/pmd/sources/metadata/registry.py` - update imports
4. `src/pmd/search/metadata/__init__.py` - update imports, remove shim re-exports
5. `src/pmd/search/metadata/inference.py` - use pmd.metadata for ontology/aliases
6. `src/pmd/search/metadata/scoring.py` - update imports
7. `src/pmd/search/metadata/retrieval.py` - update imports

## Files to Delete

After migration complete:
- `src/pmd/search/metadata/profiles.py` (shim)
- `src/pmd/search/metadata/parsers.py` (shim)
- `src/pmd/search/metadata/implementations.py` (shim)
- `src/pmd/search/metadata/registry.py` (shim)

## Test Updates

Tests currently import from `pmd.search.metadata.*` for everything. Update to:

```python
# Before
from pmd.search.metadata.profiles import ExtractedMetadata

# After
from pmd.metadata import ExtractedMetadata
```

Affected test files:
- `tests/unit/search/test_profiles.py`
- `tests/unit/search/test_parsers.py`
- `tests/unit/search/test_registry.py`
- `tests/unit/search/test_ontology.py`
- `tests/unit/search/test_inference.py`
- `tests/unit/search/test_metadata_scoring.py`
- `tests/unit/search/test_tag_retrieval.py`
- `tests/unit/search/test_pipeline_metadata_boost.py`
- `tests/integration/test_metadata_search.py`

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking imports | Re-export from old locations for backward compat |
| Circular imports | Core `pmd.metadata` has no internal deps |
| Missing functionality | Comprehensive test coverage before/after |
| Data file path changes | Use `Path(__file__).parent` pattern |

## Success Criteria

1. All existing tests pass
2. Imports from `pmd.metadata` work for core types
3. No duplicate code across modules
4. Clear separation: core vs extraction vs search
5. Single source of truth for ontology and aliases

## Open Questions

1. **Should aliases be used during extraction?** Currently only used in search inference. Could normalize during extraction for consistency.

2. **Ontology validation at extraction time?** Could warn/error if extracted tags don't exist in ontology.

3. **Profile subdirectory vs flat?** Plan shows `profiles/` subdirectory for implementations. Could keep flat if preferred.
