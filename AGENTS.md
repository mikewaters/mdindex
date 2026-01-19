# Agent Instructions

Immediately read @TASK_INSTRUCTIONS.md

# Repository Structure
**Task management**: 
- This project uses `bd` commands instead of markdown TODOs or other methods of task tracking. 
- Only use `bd` when breaking down coding tasks (always use `bd` for coding tasks!), **not** for general reasoning.
- Always create task descriptions in beads, even if they are short.
- When creating tasks using `bd`, always identify any tasks that are parallelizeable using subagents. If two tasks will not touch the same code files, they can probablybe parallelized.
- Always follow the beads SESSION CLOSE PROTOCOL , found in `bd prime` agent instructions.

**Proposals and specifications**: 
Feature, implementation, and architecture proposals are located in `features/`

**Project documentation**: 
- Code and architecture documentation for the entire project resides in `docs/`.
- Individual code modules should contain a `README.md` explaining their usage, as a supplement for docstrings.
- All python classes, methods, functions, and modules must have rigorous but concise docstrings.

---

# Design Instructions

## Breaking changes and backward-compatibility
- This is a pre-alpha product; if you are making a breaking change, THAT IS OK. You only need to adapt to this breaking change within this project.
- NEVER implement migrations or legacy fallbacks, even when instructed to. We should have a single version of schema, database, and business logic until told otherwise.

## Python authoring guidelines
- NEVER Re-define export from existing modules in some other module; if "pmd.store.something" exports symbol `Thing`, it should *NEVER* be included in another module's `__all__` declaration. 

## Software Architecture (python)
This project contains python libraries, scripts, and apps; libraries for this project reside in `src/`, end-user apps in `app/`, and end-user scripts in `scripts/`.  

---

# Operational Instructions
- Never use `pip` for installing libraries; always use `uv add <package>` for dependencies and `uv add --dev <developer-package>` for development tools.
- Never use the `python` binary directly, always use `uv run python <command>`

## Testing
You have been provided with test helpers in `Makefile`. 

### Differential tests (test only what chnaged)
We are using a test caching tool ("testmon"), and you should trust it - it is monitoring fiule changes and allowing us to run differential tests for agent speed.
If you believe tests are being skipped by testmon incorrectly, feel free to raise that to Mike. 

When testing your changes, **always** run the differential tests instead of a full test suite:
- **Run differential tests for the entire suite** - `make agent-test`
- **Run differential tests for a subset** - `make agent-test TESTPATH=tests/pmd/unit`

### Regression test/full test suite
After completing a major refactor, you may decide or be instructed to run the full test suite:
- **Run regression tests for the entire suite** - `make agent-test`
- **Run regression tests for a subset** - `make agent-test TESTPATH=tests/pmd/unit`