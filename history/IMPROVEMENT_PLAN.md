# PMD Architecture Improvement Plan

Each section is a discrete work package with a requirements specification that can be assigned to a team. Scope covers `src/pmd` unless noted.


## 2) Decide and Enforce Source Construction Model
**Context**: `sources/__init__.py` states there is no registry, yet `sources/content/registry.py` exposes a global registry. This mixes patterns and hides state.
**Goal**: Choose one model (explicit construction or registry) and enforce it consistently.
**Requirements**
- Product decision: pick either (A) explicit source construction or (B) registry-driven construction. Document the decision.
- If (A): remove the default registry singleton; make `ServiceContainer`/CLI/MCP accept a `SourceFactory` provider injected at composition time; delete registry exports and update callers.
- If (B): formalize `SourceRegistry` as an interface; instantiate it in the composition root; remove the global singleton; provide a test-only constructor for isolated registries.
- In both cases, ensure CLI `collection add` and services obtain sources through the chosen mechanism without hidden globals.
- Add tests that verify source creation is deterministic and isolated across runs.


## 5) Decouple Vector Schema from a Single LLM Provider
**Context**: Database schema is parameterized with `config.mlx.embedding_dimension`, tying storage layout to one provider/model.
**Goal**: Make vector storage configuration provider-agnostic and resilient to multiple models.
**Requirements**
- Introduce a vector-store config section (e.g., `Config.vector`) that defines embedding dimension and model identifier persisted in schema metadata.
- Store the active embedding dimension and model name in the database (schema table) and validate on startup; block or migrate if mismatched.
- Ensure LLM providers declare their embedding dimension; add a compatibility check when constructing `EmbeddingGenerator`.
- Update schema/migration tooling to adjust vector tables safely; document migration steps.
- Add tests covering mismatch detection and successful initialization with different providers.


## 7) Clarify Layered Boundaries and Public Surfaces
**Context**: Services expose repositories directly, and module names blur application vs. infrastructure vs. interface boundaries.
**Goal**: Make the codebase easy to navigate as it grows by enforcing layer boundaries and explicit public APIs.
**Requirements**
- Adopt a layered layout (proposed): `interfaces/` (CLI, MCP), `application/` (services/use-cases), `domain/` (types/ports/policies), `infrastructure/` (sqlite, LLM, sources), with re-export shims for backward compatibility during transition.
- Define public APIs per layer and limit cross-layer imports (e.g., domain cannot import infrastructure).
- Add an architecture decision record describing the layers and allowed dependencies; include a simple import-linter configuration to enforce.
- Update READMEs and diagrams to reflect the layered layout.
- Add smoke tests that construct the app through the public surface and verify basic flows (index, search).
