# Agent Instructions

Immediately read @TASK_INSTRUCTIONS.md

# Design Instructions

## Breaking changes and backward-compatibility
- This is a pre-alpha product; if you are making a breaking change, THAT IS OK. You only need to adapt to this breaking change within this project.
- NEVER implement migrations or legacy fallbacks, even when instructed to. We should have a single version of schema, database, and business logic until told otherwise.

## Python authoring guidelines
- NEVER Re-define export from existing modules in some other module; if "pmd.store.something" exports symbol `Thing`, it should *NEVER* be included in another module's `__all__` declaration. 

## Software Architecture (python)
This project contains python libraries, scripts, and apps; libraries for this project reside in `src/`, end-user apps in `app/`, and end-user scripts in `scripts/`.  

# Operational Instructions
- Never use `pip` for installing libraries; always use `uv add <package>` for dependencies and `uv add --dev <developer-package>` for development tools.
- Never use the `python` binary directly, always use `uv run python <command>`

## Testing
You have been provided with test helpers in `Makefile`.

When testing your changes, **always** run the differential tests instead of a full test suite:
- **Run differential tests for the entire suite** - `make agent-test`
- **Run differential tests for a subset** - `make agent-test TESTPATH=tests/pmd/unit`

After completing a major refactor, you may decide or be instructed to run the full test suite:
- **Run regression tests for the entire suite** - `make agent-test`
- **Run regression tests for a subset** - `make agent-test TESTPATH=tests/pmd/unit`