# Schema Centralization and Migrations Plan

## Current State Analysis

### Schema Definition
- `store/schema.py` contains `SCHEMA_VERSION = 1` (unused for actual tracking)
- All tables defined with `IF NOT EXISTS` - no real versioning
- Tables: content, collections, documents, documents_fts, content_vectors, path_contexts, ollama_cache, source_metadata, document_metadata, document_tags

### Schema Application
- `store/database.py` runs full schema via `_init_schema()` on every connect
- Uses `executescript(get_schema())` - idempotent but not versioned

### Ad-hoc Table Creation
- `metadata/store/repository.py` has `_ensure_table()` that duplicates DDL for `document_metadata` and `document_tags`
- This bypasses the central schema and could drift

## Implementation Plan

### Task 1: Create Migration Infrastructure
Create `store/migrations/` module with:
- `__init__.py` - Exports runner and utilities
- `runner.py` - Migration runner using SQLite `user_version` pragma
- `migrations/` directory for versioned migration files

Key design decisions:
- Use SQLite `PRAGMA user_version` for tracking (built-in, atomic)
- Each migration is a Python module with `up()` function
- Migrations are idempotent (use `IF NOT EXISTS` internally)
- Runner applies all migrations from current version to latest

### Task 2: Convert Schema to Versioned Migrations
- Create `0001_initial_schema.py` with current base schema
- Future changes become new migration files (0002, 0003, etc.)
- Keep `schema.py` as reference/documentation but runner uses migrations

Migration file structure:
```python
# migrations/0001_initial_schema.py
VERSION = 1
DESCRIPTION = "Initial schema with all core tables"

def up(db):
    """Apply migration."""
    db.executescript(SCHEMA_SQL)
```

### Task 3: Update Database Class
Modify `database.py`:
- Replace `_init_schema()` with call to migration runner
- Migration runner handles both fresh DB and upgrades
- Logging for migration status

### Task 4: Remove Ad-hoc Table Creation
Remove `_ensure_table()` from `DocumentMetadataRepository`:
- Delete the method entirely
- Remove call from `__init__`
- Repository now assumes tables exist (enforced by migration runner)

### Task 5: Add Migration Tests
Test scenarios in `tests/unit/store/test_migrations.py`:
1. Fresh database initialization
2. Upgrade from prior version (mock older DB)
3. No-op when already at latest version
4. Rollback safety (ensure migrations are atomic)

### Task 6: Document Migration Policy
Create `docs/MIGRATIONS.md` or update `store/README.md`:
- How to add new migrations
- Version numbering convention
- Testing requirements for migrations
- Backward compatibility considerations

## File Changes Summary

### New Files
- `src/pmd/store/migrations/__init__.py`
- `src/pmd/store/migrations/runner.py`
- `src/pmd/store/migrations/versions/0001_initial_schema.py`
- `tests/unit/store/test_migrations.py`

### Modified Files
- `src/pmd/store/database.py` - Use migration runner
- `src/pmd/store/schema.py` - Keep as reference, add deprecation note
- `src/pmd/metadata/store/repository.py` - Remove `_ensure_table()`
- `src/pmd/store/README.md` or `docs/MIGRATIONS.md` - Documentation

## Migration Runner Design

```python
class MigrationRunner:
    """Applies versioned migrations to database."""

    def __init__(self, db_connection):
        self.conn = db_connection

    def get_current_version(self) -> int:
        """Get current schema version from user_version pragma."""
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def set_version(self, version: int) -> None:
        """Set schema version."""
        self.conn.execute(f"PRAGMA user_version = {version}")

    def run_migrations(self) -> int:
        """Apply all pending migrations. Returns count applied."""
        current = self.get_current_version()
        applied = 0

        for migration in self._get_migrations_after(current):
            migration.up(self.conn)
            self.set_version(migration.VERSION)
            applied += 1

        return applied
```

## Risk Assessment

- **Low Risk**: `IF NOT EXISTS` ensures idempotency even if run multiple times
- **Medium Risk**: Removing `_ensure_table()` requires ensuring migration runs first
- **Mitigation**: Test with fresh DB and existing DBs in CI

## Success Criteria

1. Single source of truth for all DDL in migrations module
2. `PRAGMA user_version` tracks schema version
3. Fresh DB and upgrades both work via same code path
4. No ad-hoc table creation in repositories
5. Tests cover all migration scenarios
6. Documentation explains how to add new tables
