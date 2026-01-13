# Export Reduction Plan

Goal: ensure every `__all__` in `src/pmd` only lists symbols consumed by other modules (i.e., outside that package), and switch all remaining importers to the submodules that actually define those symbols. Each section below outlines the actions for the modules that currently expose large `__all__` lists.

## General workflow
1. For a given package-level `__all__`, run `rg "from pmd.<module>"` / `rg "from pmd.<module> import"` to enumerate actual consumers outside the package.
2. Pay special attention to relative imports using `.`/`..` that still rely on the package exports (e.g., `from ..llm import create_llm_provider`). Treat them like any other consumer: switch the import to the precise submodule if the symbol is no longer part of the trimmed `__all__`.
3. Where no external consumer exists, drop `__all__` and treat the package as an implementation detail; otherwise, keep only the names that appear in those import statements (absolute or relative).
4. Update the import sites that previously relied on the broader export to import directly from the defining submodule.
5. Refresh README/api comments so contributors know exactly where to grab each symbol and that the top-level packages no longer re-export everything.
6. Run the relevant tests/CLI flows to confirm no regressions after tightening the exported surface.

## Module: `src/pmd/core/__init__.py`
- **Current state**: re-exports config DTOs, exception hierarchy, and `types` artifacts.
- **Observation**: no other module imports `pmd.core` directly—the repo consistently uses submodules like `pmd.core.config`, `pmd.core.exceptions`, or `pmd.core.types`.
- **Plan**:
  1. Confirm with `rg -n "from pmd\\.core import"` that zero importers exist; if any do, refactor them to import from their true submodule.
  2. Remove the package-level `__all__` (or shrink it to a single doc helper) so the module no longer pretends to export everything.
  3. Update `core/README` to explicitly document the preferred import targets.

## Module: `src/pmd/llm/__init__.py`
- **Current state**: exports provider base classes, factory helpers, and embedding/expander/reranker utilities.
- **Observation**: there are no `from pmd.llm import ...` usages anywhere (everything imports via relative paths inside the package).
- **Plan**:
  1. Drop `__all__` entirely and treat `pmd.llm` as an internal namespace with submodules as the API.
  2. If a convenience export is needed later, reintroduce only those symbols that have actual external consumers.

## Module: `src/pmd/services/__init__.py`
- **Current state**: exposes `ServiceContainer`, service classes, and DTOs (`IndexResult`, `EmbedResult`, `CleanupResult`).
- **Observation**: the CLI imports only `ServiceContainer` (`from pmd.services import ServiceContainer`).
- **Plan**:
  1. Prune `__all__` to only include `ServiceContainer`.
  2. Document that other services/results must be imported from their own submodules (`pmd.services.indexing`, etc.).
  3. Verify no residual imports rely on `pmd.services.IndexingService`, `SearchService`, etc.

## Module: `src/pmd/cli/commands/__init__.py`
- **Current state**: publishes all handler functions/helper utilities via `__all__`.
- **Observation**: no external module imports from this package; `cli.main` uses `from . import commands`.
- **Plan**:
  1. Remove `__all__` and consider this module private implementation detail.
  2. If some helpers genuinely need exposition, create an explicit `pmd.cli.api` module with a deliberately small export surface.

## Module: `src/pmd/store/__init__.py`
- **Current state**: exports `VectorSearchRepository`.
- **Observation**: no code currently imports `from pmd.store import VectorSearchRepository`; every consumer uses the concrete submodule.
- **Plan**:
  1. Drop `__all__` and the re-export; update the store README to instruct developers to import from `pmd.store.vector_search`.

## Module: `src/pmd/store/document_metadata.py`
- **Current state**: deprecated shim re-exporting `DocumentMetadataRepository` and `StoredDocumentMetadata`.
- **Observation**: only `DocumentMetadataRepository` is imported elsewhere via this path (the `StoredDocumentMetadata` import comes from `pmd.metadata`).
- **Plan**:
  1. Limit `__all__` to `["DocumentMetadataRepository"]`.
  2. Keep the deprecation warning but make it clear that only the repository survives on this path.
  3. Optionally re-export `StoredDocumentMetadata` elsewhere if necessary, but not through this shim.

## Module: `src/pmd/metadata/__init__.py`
- **Current state**: huge re-export of metadata types, extraction profiles, query helpers, and storage helpers.
- **Observation**: external consumers use only a subset of these names:
  - `DocumentMetadataRepository`, `StoredDocumentMetadata` (store/document_metadata).
  - `load_default_aliases` (metadata/query/inference).
  - `ExtractedMetadata`, `MetadataProfileRegistry`, `get_default_profile_registry` (source modules).
  - `LexicalTagMatcher`, `TagRetriever`, `Ontology` (search metadata/pipeline).
  - `GenericProfile`, `ObsidianProfile` (deprecated source metadata shim).
  - `MetadataProfileRegistry` + parsing helpers (filesystem source) via `pmd.metadata`.
- **Plan**:
  1. Build a definitive list of external consumers via `rg "from pmd.metadata import"` and record which names are imported.
  2. Leave only those names in `__all__` (everything else should be removed or imported directly from the appropriate submodule).
  3. Aggressively document which symbols remain exported and point readers to the submodules for the rest.

## Module: `src/pmd/metadata/model/__init__.py`
- **Observation**: no external module imports `pmd.metadata.model` directly; only `pmd.metadata` re-exports these types.
- **Plan**:
  1. Drop the `__all__` list, as this module is purely internal.
  2. If someone needs a custom import, they can import from `pmd.metadata.model.types`/`.ontology`/`.aliases` directly.

## Module: `src/pmd/metadata/extraction/__init__.py`
- **Observation**: only `pmd.metadata` and the deprecated `pmd.sources.metadata` import from here.
- **Plan**:
  1. Remove `__all__` or limit it to the handful of names that the deprecated shim still needs.
  2. Encourage external consumers to import from `pmd.metadata` or the specific extraction submodule (e.g., `.generic`, `.obsidian`, `.registry`).

## Module: `src/pmd/metadata/query/__init__.py`
- **Observation**: used by the deprecated `pmd.search.metadata` shim and via `from pmd.metadata.query import ...` in a few other spots.
- **Plan**:
  1. Identify which names are actually imported by the external consumers (e.g., `LexicalTagMatcher`, `TagRetriever`, `MetadataBoostConfig`, etc.).
  2. Keep only those in `__all__`.
  3. Update the shim and README to import only the retained names.

## Module: `src/pmd/metadata/store/__init__.py`
- **Observation**: exposes only `DocumentMetadataRepository`, which is genuinely needed by other modules.
- **Plan**:
  1. Keep `__all__` as-is but document that this repo is the only public symbol from the store package.

## Module: `src/pmd/sources/content/__init__.py`
- **Observation**: no module imports from this package; consumers import the individual submodules.
- **Plan**:
  1. Remove `__all__` entirely and treat the package as internal.
  2. Clean the README to list the available submodules and their public API surfaces.

## Module: `src/pmd/sources/__init__.py`
- **Observation**: there may be external usage (`services.container` builds sources, CLI).
- **Plan**:
  1. Run `rg -n "from pmd\\.sources import" src` to list exactly which names others pull from this module.
  2. After capturing the names (e.g., `SourceRegistry`, `get_default_registry`, `FileSystemSource` if used), trim `__all__` to that minimal set.
  3. If the module is no longer a stable public surface, remove `__all__` and add a doc note redirecting callers to the desired entry points.

## Module: `src/pmd/sources/metadata/__init__.py`
- **Observation**: deprecated shim re-exporting profiles, registry, and parsing helpers.
- **Plan**:
  1. Keep only the exports that still serve backward compatibility (`GenericProfile`, `ObsidianProfile`, `MetadataProfileRegistry`, etc.).
  2. Remove unused names from `__all__` and mention in comments that the module is scheduled for removal.

## Module: `src/pmd/search/metadata/__init__.py`
- **Observation**: deprecated compatibility layer used by older importers.
- **Plan**:
  1. Determine which names external consumers still reference; leave only those in `__all__`.
  2. Add a stronger deprecation notice pointing to `pmd.metadata.query`.
