# Metadata Bounded Context Unification Plan

## Current State

Metadata logic is fragmented across four locations:

| Location | Purpose | Key Types |
|----------|---------|-----------|
| `pmd/metadata/` | Core ontology & aliases | `Ontology`, `TagAliases` |
| `pmd/sources/metadata/` | Extraction profiles | `ExtractedMetadata`, `*Profile` |
| `pmd/search/metadata/` | Query-time inference/boost | `LexicalTagMatcher`, `TagRetriever` |
| `pmd/store/document_metadata.py` | Persistence | `DocumentMetadataRepository` |

**Problems:**
- Unclear ownership and API boundaries
- Consumers import from multiple locations
- Duplicate/overlapping utilities
- Hard to understand the metadata domain as a whole

## Target State

A single `pmd/metadata/` package with clear submodules:

```
pmd/metadata/
├── __init__.py          # Public API surface
├── model/               # Core types & ontology
│   ├── __init__.py
│   ├── ontology.py      # Ontology, OntologyNode
│   ├── aliases.py       # TagAliases
│   ├── types.py         # ExtractedMetadata, StoredDocumentMetadata
│   └── data/            # JSON data files
├── extraction/          # Source-aware extraction
│   ├── __init__.py
│   ├── profiles.py      # MetadataProfile protocol
│   ├── generic.py       # GenericProfile
│   ├── obsidian.py      # ObsidianProfile
│   ├── drafts.py        # DraftsProfile
│   ├── registry.py      # MetadataProfileRegistry
│   └── parsing.py       # Frontmatter & inline tag parsing
├── query/               # Query-time operations
│   ├── __init__.py
│   ├── inference.py     # LexicalTagMatcher, TagMatch
│   ├── retrieval.py     # TagRetriever, TagSearchConfig
│   └── scoring.py       # apply_metadata_boost, BoostResult
└── store/               # Persistence layer
    ├── __init__.py
    └── repository.py    # DocumentMetadataRepository
```

## Implementation Phases

### Phase 1: Create Model Subpackage

1. Create `pmd/metadata/model/` directory structure
2. Move `ontology.py` and `aliases.py` from current `metadata/` root
3. Move `ExtractedMetadata` from `sources/metadata/types.py`
4. Move `StoredDocumentMetadata` from `store/document_metadata.py`
5. Move `data/` directory with JSON files
6. Update internal imports within model/
7. Export all types from `model/__init__.py`

### Phase 2: Create Extraction Subpackage

1. Create `pmd/metadata/extraction/` directory
2. Move `MetadataProfile` protocol from `sources/metadata/types.py`
3. Move `GenericProfile` from `sources/metadata/base.py`
4. Move `ObsidianProfile` from `sources/metadata/obsidian.py`
5. Move `DraftsProfile` from `sources/metadata/drafts.py`
6. Move `MetadataProfileRegistry` from `sources/metadata/registry.py`
7. Move parsing utilities from `sources/metadata/parsing.py`
8. Update imports to use `pmd.metadata.model` for types
9. Export public API from `extraction/__init__.py`

### Phase 3: Create Query Subpackage

1. Create `pmd/metadata/query/` directory
2. Move `LexicalTagMatcher`, `TagMatch` from `search/metadata/inference.py`
3. Move `TagRetriever`, `TagSearchConfig` from `search/metadata/retrieval.py`
4. Move scoring functions from `search/metadata/scoring.py`
5. Update imports to use `pmd.metadata.model` for types
6. Consolidate any duplicate tag-matching utilities
7. Export public API from `query/__init__.py`

### Phase 4: Create Store Subpackage

1. Create `pmd/metadata/store/` directory
2. Move `DocumentMetadataRepository` from `store/document_metadata.py`
3. Keep schema definitions with repository
4. Update imports to use `pmd.metadata.model` for types
5. Export public API from `store/__init__.py`

### Phase 5: Define Public API Surface

1. Design `metadata/__init__.py` with clear exports:
   - Model types: `Ontology`, `TagAliases`, `ExtractedMetadata`, etc.
   - Extraction: `MetadataProfileRegistry`, profile classes
   - Query: `LexicalTagMatcher`, `TagRetriever`, scoring functions
   - Store: `DocumentMetadataRepository`
2. Document the public API with docstrings
3. Add `__all__` lists to control exports

### Phase 6: Create Deprecation Shims

1. Create shim modules at old import paths:
   - `pmd/sources/metadata/__init__.py` → re-exports with warnings
   - `pmd/search/metadata/__init__.py` → re-exports with warnings
   - `pmd/store/document_metadata.py` → re-exports with warnings
2. Use `warnings.warn()` with `DeprecationWarning`
3. Include migration instructions in warning messages

### Phase 7: Update Consumers

1. Update `pmd/services/indexing.py` to use new imports
2. Update `pmd/search/pipeline.py` to use new imports
3. Update `pmd/sources/content/*.py` to use new imports
4. Update any CLI commands using metadata
5. Update MCP server if applicable

### Phase 8: Update Tests

1. Add tests for import shims (verify warnings are raised)
2. Add tests for cross-module interactions:
   - extraction → store flow
   - query → store flow
   - extraction → query flow (tag normalization)
3. Update existing test imports to use new paths
4. Verify all tests pass

### Phase 9: Update Documentation

1. Update `docs/ARCHITECTURE.md` with new layout
2. Update `docs/DATA_ARCHITECTURE.md` for metadata domain
3. Create/update metadata-specific README
4. Update any class diagrams showing metadata components
5. Remove obsolete documentation references

### Phase 10: Cleanup

1. Remove empty old directories after shims are in place
2. Remove any truly dead code identified during migration
3. Final verification that all imports resolve correctly
4. Run full test suite

## Public API Design

```python
# pmd/metadata/__init__.py

# === Model Types ===
from pmd.metadata.model import (
    # Ontology
    Ontology,
    OntologyNode,
    load_default_ontology,
    # Aliases
    TagAliases,
    load_default_aliases,
    # Extracted metadata
    ExtractedMetadata,
    # Stored metadata
    StoredDocumentMetadata,
)

# === Extraction ===
from pmd.metadata.extraction import (
    # Protocol
    MetadataProfile,
    # Profiles
    GenericProfile,
    ObsidianProfile,
    DraftsProfile,
    # Registry
    MetadataProfileRegistry,
    get_default_profile_registry,
    # Parsing utilities
    parse_frontmatter,
    extract_inline_tags,
)

# === Query ===
from pmd.metadata.query import (
    # Inference
    LexicalTagMatcher,
    TagMatch,
    create_default_matcher,
    # Retrieval
    TagRetriever,
    TagSearchConfig,
    create_tag_retriever,
    # Scoring
    MetadataBoostConfig,
    BoostResult,
    apply_metadata_boost,
    apply_metadata_boost_v2,
)

# === Store ===
from pmd.metadata.store import (
    DocumentMetadataRepository,
)
```

## Migration Guide (for deprecation warnings)

```
DeprecationWarning: Importing from 'pmd.sources.metadata' is deprecated.
Use 'pmd.metadata.extraction' instead. This import path will be removed in v2.0.

Before: from pmd.sources.metadata import ObsidianProfile
After:  from pmd.metadata.extraction import ObsidianProfile
```

## Risk Mitigation

1. **Incremental migration**: Each phase is independently testable
2. **Shims preserve compatibility**: No immediate breakage for external consumers
3. **Tests first**: Verify behavior before and after each phase
4. **Git commits per phase**: Easy to revert if issues arise

## Success Criteria

- [ ] All metadata code lives under `pmd/metadata/`
- [ ] Single public API surface in `metadata/__init__.py`
- [ ] Old import paths work with deprecation warnings
- [ ] No duplicate implementations remain
- [ ] All existing tests pass
- [ ] New cross-module interaction tests added
- [ ] Documentation reflects new structure
