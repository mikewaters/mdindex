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

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
