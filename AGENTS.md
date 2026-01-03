# Instructions

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods. Immediately read the instructions for **bd (beads**) here: @TASK_MANAGEMENT.md

## Project Memory
Here are some important items we've encountered:
- Do not edit or create files in tests/fixtures/test-corpus
- Do not use emojis in user- or operator-facing tools. Its is OK to put emojis in project documentation when fitting, but user messages, error messages, logs, and terminal output tools should never have emoji.

## Project structure
- python3 type-safe modular coding
- never use pip, always use the `uv` command: 
  - `uv add <dependency-name>`; `uv run <script>`

## Testing requirements
- Define and document test cases before creating tests. Add them to `./tests/TEST_CASES.md`
- Requires high unit test coverage with integration tests and end-to-end tests for major features: `./tests/{unit,integration,e2e}`