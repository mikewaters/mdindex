# Monorepo Structure Plan

## Overview

This document outlines the plan to convert the mdindex repository into a monorepo supporting:
1. **Distributable libraries** in `src/` (e.g., `src/pmd`) - each with their own `pyproject.toml`
2. **Non-distributable application code** in top-level directories (e.g., `app/`) - managed by the root project

## Current State

```
mdindex/
├── pyproject.toml          # Single project config
├── uv.lock
├── src/
│   ├── __init__.py
│   └── pmd/                # The pmd library
│       ├── __init__.py
│       ├── cli/
│       ├── core/
│       └── ...
└── tests/
```

## Target State

```
mdindex/
├── pyproject.toml          # Workspace root + app code config
├── uv.lock                 # Single lockfile for entire workspace
├── src/
│   ├── pmd/
│   │   ├── pyproject.toml  # pmd library config (distributable)
│   │   ├── __init__.py
│   │   └── ...
│   └── future-lib/         # Future libraries go here
│       ├── pyproject.toml
│       └── ...
├── app/                    # Non-distributable application code
│   ├── __init__.py
│   └── ...
└── tests/
```

## Implementation Approach: uv Workspaces

uv natively supports workspaces, similar to npm/Cargo. This is the recommended approach for Python monorepos using uv.

### Key Concepts

1. **Workspace Root**: The root `pyproject.toml` defines `[tool.uv.workspace]` with member patterns
2. **Workspace Members**: Each library has its own `pyproject.toml` and can be published independently
3. **Single Lockfile**: One `uv.lock` manages all dependencies across the workspace
4. **Editable Installs**: Workspace members are automatically installed as editable

---

## Detailed Changes

### 1. Create `src/pmd/pyproject.toml`

This makes `pmd` an independently distributable package.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pmd"
version = "1.0.0"
description = "Python Markdown Search - Hybrid search engine for markdown documents"
readme = "../../README.md"
requires-python = ">=3.11"
authors = [
    {name = "Mike", email = "mike@example.com"},
]
license = {text = "MIT"}

dependencies = [
    "sqlite-vec>=0.1.0",
    "httpx>=0.27.0",
    "rich>=13.0.0",
    "click>=8.1.0",
    "pydantic>=2.0.0",
    "mlx-lm>=0.29.1",
    "mlx-embeddings>=0.0.5",
    "loguru>=0.7.3",
    "opentelemetry-sdk>=1.39.1",
    "opentelemetry-exporter-otlp-proto-http>=1.39.1",
    "arize-phoenix[all]>=12.27.0",
    "arize-phoenix-otel>=0.14.0",
    "pyyaml>=6.0.3",
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
    "aiosqlite>=0.20.0",
    "fsspec>=2024.0.0",
    "litellm>=1.50.0",
    "dspy>=2.5.0",
]

[project.optional-dependencies]
loaders = [
    "llama-index-core>=0.11.0",
    "llama-index-readers-web>=0.2.0",
    "llama-index-readers-file>=0.2.0",
]

[project.scripts]
pmd = "pmd.cli.main:main"

[tool.hatch.build.targets.wheel]
packages = ["."]
```

### 2. Modify Root `pyproject.toml`

The root becomes the workspace definition and optionally contains non-distributable app code.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mdindex"
version = "1.0.0"
description = "mdindex monorepo - workspace root"
requires-python = ">=3.11"
authors = [
    {name = "Mike", email = "mike@example.com"},
]
license = {text = "MIT"}

# Root-level dependencies (for app/ code and dev tools)
dependencies = [
    # Workspace member as dependency (installed editable automatically)
    "pmd",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
    "tach>=0.33.0",
    "respx>=0.22.0",
]

# ============================================================
# UV WORKSPACE CONFIGURATION
# ============================================================
[tool.uv.workspace]
members = ["src/*"]

# Declare workspace members as sources (editable installs)
[tool.uv.sources]
pmd = { workspace = true }

# ============================================================
# HATCH BUILD CONFIGURATION
# ============================================================
# Include app/ directory as part of root package (non-distributable)
[tool.hatch.build.targets.wheel]
packages = ["app"]

# ============================================================
# TOOL CONFIGURATIONS (shared across workspace)
# ============================================================
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=src/pmd --cov-report=term-missing --cov-report=html"

[tool.coverage.run]
source = ["src/pmd", "app"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
show_missing = true

[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_configs = true

[tool.ruff]
line-length = 100
target-version = "py311"

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-asyncio>=1.3.0",
    "pytest-cov>=6.0.0",
    "respx>=0.22.0",
    "tach>=0.33.0",
]
```

### 3. Remove `src/__init__.py`

The `src/` directory should NOT be a Python package in a monorepo setup - it's just a container for workspace members.

```bash
rm src/__init__.py
```

### 4. Update Import Paths

With this structure, imports work as follows:

```python
# Importing from pmd library (workspace member)
from pmd.core import SomeClass
from pmd.cli.main import main

# Importing from app/ (root package, non-distributable)
from app.some_module import something
```

### 5. Update `tach.toml`

```toml
interfaces = []
exclude = [
    "**/*__pycache__",
    "**/*egg-info",
    "**/docs",
    "**/tests",
    "**/venv",
]
source_roots = [
    "src/pmd",
    "app",
]

[[modules]]
path = "pmd"
depends_on = []

[[modules]]
path = "app"
depends_on = ["pmd"]
```

---

## Alternative Approaches Considered

### Option A: pip-tools with Editable Installs (Not Recommended)
- Manual management of editable installs
- No native workspace support
- More complex CI/CD setup

### Option B: Hatch Environments (Partial Support)
- Hatch has environment management but less mature workspace support
- Would require custom scripts for cross-package development

### Option C: Poetry Workspaces (Not Available)
- Poetry doesn't support workspaces natively
- Would require plugins and workarounds

**Recommendation: uv Workspaces** is the clear winner for this project since uv is already in use.

---

## Migration Steps

### Phase 1: Prepare the Structure

1. Create `src/pmd/pyproject.toml` with pmd's dependencies and metadata
2. Update root `pyproject.toml` with workspace configuration
3. Remove `src/__init__.py`

### Phase 2: Verify and Sync

4. Run `uv sync` to regenerate `uv.lock` with workspace awareness
5. Verify imports still work: `uv run python -c "from pmd import __version__; print(__version__)"`
6. Run tests: `uv run pytest`

### Phase 3: Update Tooling

7. Update `tach.toml` for new source roots
8. Update CI/CD workflows if necessary
9. Update any scripts that reference the old structure

### Phase 4: Add app/ Directory (When Needed)

10. Create `app/` directory with `__init__.py`
11. App code can import from `pmd` directly
12. Update coverage and test configurations as needed

---

## Working with the Monorepo

### Common Commands

```bash
# Install all workspace members (from root)
uv sync

# Run a command using a workspace member
uv run pmd --help

# Add a dependency to a specific member
uv add --package pmd some-new-dependency

# Add a dev dependency to root
uv add --dev some-dev-tool

# Run tests
uv run pytest

# Build a specific package for distribution
cd src/pmd && uv build
```

### Adding a New Library

1. Create `src/new-lib/` directory
2. Add `src/new-lib/pyproject.toml`
3. Run `uv sync` to pick up the new member
4. The new library is automatically available to other workspace members

---

## Test Organization

### Current Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── fixtures/             # Test data files
├── fakes/                # Mock implementations
├── unit/
│   ├── app/
│   ├── core/
│   ├── llm/
│   ├── metadata/
│   ├── search/
│   ├── services/
│   ├── sources/
│   └── store/
└── integration/
    ├── conftest.py
    ├── test_collection_indexing.py
    ├── test_corpus_hybrid_search.py
    └── ...
```

### Options for Test Organization

#### Option A: Keep Tests Centralized at Root (Recommended)

**Structure:**
```
mdindex/
├── tests/                    # All tests remain here
│   ├── conftest.py
│   ├── fixtures/
│   ├── fakes/
│   ├── unit/
│   │   └── pmd/              # Rename from mirroring src/pmd structure
│   └── integration/
├── src/
│   └── pmd/
└── app/
```

**Pros:**
- No migration required - tests stay where they are
- Shared fixtures (`conftest.py`, `fixtures/`, `fakes/`) work naturally
- Easy to run all tests at once: `uv run pytest`
- Integration tests that span libraries have a natural home

**Cons:**
- Tests don't travel with library when published (usually fine - tests aren't typically distributed)
- As more libraries are added, test organization needs clear conventions

**Configuration (root pyproject.toml):**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["src/pmd", "app"]
branch = true
```

#### Option B: Co-located Tests (Tests Inside Each Library)

**Structure:**
```
mdindex/
├── tests/                    # Root-level integration/E2E tests only
│   └── integration/
├── src/
│   └── pmd/
│       ├── pyproject.toml
│       ├── tests/            # pmd's unit tests
│       │   ├── conftest.py
│       │   ├── unit/
│       │   └── fixtures/
│       └── ...
└── app/
    └── tests/                # app's tests
```

**Pros:**
- Clear ownership - each library's tests are with the library
- Tests travel with library if published with `include-package-data`
- Easy to run tests for just one library: `uv run pytest src/pmd/tests`

**Cons:**
- Shared fixtures need duplication or extraction to a `tests-common` package
- More complex pytest configuration
- Current test structure would need significant reorganization

**Configuration (src/pmd/pyproject.toml):**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

#### Option C: Hybrid Approach

**Structure:**
```
mdindex/
├── tests/
│   ├── conftest.py           # Shared across all tests
│   ├── fixtures/             # Shared fixtures
│   ├── fakes/                # Shared mocks
│   ├── integration/          # Cross-library integration tests
│   └── e2e/                  # End-to-end tests
├── src/
│   └── pmd/
│       └── tests/            # pmd unit tests only
│           └── unit/
└── app/
    └── tests/
```

**Recommendation: Option A (Centralized)** for this project because:
1. Current test structure already works well
2. Shared fixtures are extensively used
3. Integration tests are important and span pmd modules
4. No immediate need to distribute tests with published packages

---

## Configuration Migration Reference

### What Moves to `src/pmd/pyproject.toml`

| Section | Move? | Reasoning |
|---------|-------|-----------|
| `[project]` name, version, description | **Yes** | Library identity |
| `[project]` readme, authors, license | **Yes** | Library metadata |
| `[project]` requires-python | **Yes** | Library requirement |
| `dependencies` | **Yes** | pmd's runtime dependencies |
| `[project.optional-dependencies].loaders` | **Yes** | pmd-specific optional deps |
| `[project.scripts]` pmd CLI | **Yes** | pmd's entry point |
| `[tool.hatch.build.targets.wheel]` | **Yes** | But simplified: `packages = ["."]` |

### What Stays at Root `pyproject.toml`

| Section | Reasoning |
|---------|-----------|
| `[tool.uv.workspace]` | Workspace definition |
| `[tool.uv.sources]` | Workspace member references |
| `[project.optional-dependencies].dev` | Workspace-wide dev tools |
| `[dependency-groups].dev` | Workspace-wide dev dependencies |
| `[tool.pytest.ini_options]` | Tests run from root |
| `[tool.coverage.*]` | Coverage spans all packages |
| `[tool.mypy]` | Workspace-wide type checking settings |
| `[tool.ruff]` | Workspace-wide linting settings |

### Special Considerations

**README handling:**
- pmd's `pyproject.toml` can reference the root README: `readme = "../../README.md"`
- Or create a pmd-specific README at `src/pmd/README.md`

**Version management:**
- Each library has its own version in its `pyproject.toml`
- Root package version is separate (if it even needs one)

**Entry points / Scripts:**
- `[project.scripts]` moves to the library that provides the CLI
- Multiple libraries can each define their own scripts

---

## Detailed File Changes

### `src/pmd/pyproject.toml` (NEW)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pmd"
version = "1.0.0"
description = "Python Markdown Search - Hybrid search engine for markdown documents"
readme = "../../README.md"
requires-python = ">=3.11"
authors = [
    {name = "Mike", email = "mike@example.com"},
]
license = {text = "MIT"}

dependencies = [
    "sqlite-vec>=0.1.0",
    "httpx>=0.27.0",
    "rich>=13.0.0",
    "click>=8.1.0",
    "pydantic>=2.0.0",
    "mlx-lm>=0.29.1",
    "mlx-embeddings>=0.0.5",
    "loguru>=0.7.3",
    "opentelemetry-sdk>=1.39.1",
    "opentelemetry-exporter-otlp-proto-http>=1.39.1",
    "arize-phoenix[all]>=12.27.0",
    "arize-phoenix-otel>=0.14.0",
    "pyyaml>=6.0.3",
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
    "aiosqlite>=0.20.0",
    "fsspec>=2024.0.0",
    "litellm>=1.50.0",
    "dspy>=2.5.0",
]

[project.optional-dependencies]
loaders = [
    "llama-index-core>=0.11.0",
    "llama-index-readers-web>=0.2.0",
    "llama-index-readers-file>=0.2.0",
]

[project.scripts]
pmd = "pmd.cli.main:main"

[tool.hatch.build.targets.wheel]
packages = ["."]
```

### Root `pyproject.toml` (MODIFIED)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mdindex"
version = "1.0.0"
description = "mdindex monorepo workspace"
requires-python = ">=3.11"
authors = [
    {name = "Mike", email = "mike@example.com"},
]
license = {text = "MIT"}

# Workspace members as dependencies (auto-editable)
dependencies = [
    "pmd",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
]

# ============================================================
# UV WORKSPACE
# ============================================================
[tool.uv.workspace]
members = ["src/*"]

[tool.uv.sources]
pmd = { workspace = true }

# ============================================================
# HATCH BUILD (for app/ if needed)
# ============================================================
[tool.hatch.build.targets.wheel]
packages = ["app"]

# ============================================================
# TESTING (centralized)
# ============================================================
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=src/pmd --cov-report=term-missing --cov-report=html"

[tool.coverage.run]
source = ["src/pmd", "app"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
show_missing = true

# ============================================================
# LINTING & TYPE CHECKING (workspace-wide)
# ============================================================
[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_configs = true

[tool.ruff]
line-length = 100
target-version = "py311"

# ============================================================
# DEV DEPENDENCIES
# ============================================================
[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-asyncio>=1.3.0",
    "pytest-cov>=6.0.0",
    "respx>=0.22.0",
    "tach>=0.33.0",
]
```

---

## Considerations

### Publishing
- Each library in `src/` can be published independently to PyPI
- The root package (`mdindex`) is typically NOT published
- Use `uv build` within each library directory to create distributions

### Versioning
- Each library manages its own version in its `pyproject.toml`
- Consider using a tool like `bump2version` or `commitizen` for coordinated releases

### IDE Support
- VS Code: May need to configure `python.analysis.extraPaths` for proper autocomplete
- PyCharm: Mark `src/pmd` and `app` as Sources Root
