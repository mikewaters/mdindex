# ADR-0001: Ambient Session Registry via ContextVars

**Status:** Proposed
**Date:** 2026-01-20

## Context

The idx library currently passes SQLAlchemy sessions explicitly through constructor injection. This creates several pain points:

1. **Plumbing overhead**: Every repository, manager, and transform requires a `session` parameter in its constructor (e.g., `DocumentRepository(session)`, `FTSManager(session)`, `PersistenceTransform(session_factory=...)`).

2. **Workarounds for shared sessions**: When multiple components need to share a session within a transaction, we resort to patterns like `_session_passthrough()` context managers or lambda factories (see `src/idx/pipelines/ingest.py`).

3. **Testing friction**: Tests must create fixtures that provide session factories and wire them through all layers.

4. **Future async concerns**: While currently sync-only, any future async migration would benefit from having session management already using contextvars (which work correctly across async contexts).

### Current Pattern

```python
# Entry point opens session
with get_session() as session:
    # Manual plumbing through all layers
    doc_repo = DocumentRepository(session)
    fts = FTSManager(session)

    # Even transforms need session factories
    persist = PersistenceTransform(
        session_factory=lambda: _session_passthrough(session),
        ...
    )
```

## Decision

Implement an **ambient session registry** using Python's `contextvars` module:

1. Create `idx.store.session_context` module with:
   - `_current_session: ContextVar[Session | None]` - holds the ambient session
   - `set_session(session)` - set the ambient session for current context
   - `get_current_session()` - get the ambient session (raises if not set)
   - `@with_session` decorator - sets session from `get_session()` if not already set

2. Update repositories and managers to use ambient session:
   - Constructor becomes `__init__(self, session: Session | None = None)`
   - If session is None, call `get_current_session()`
   - Allows both explicit injection (for testing) and ambient lookup (for production)

3. Update tests to set ambient session per test:
   - Fixtures set session via `set_session()`
   - Tests can still inject explicitly when needed

### Target Pattern

```python
# Entry point sets ambient session
with get_session() as session:
    set_session(session)

    # No plumbing needed - repositories find session automatically
    doc_repo = DocumentRepository()
    fts = FTSManager()

    # Transforms also use ambient session
    persist = PersistenceTransform(dataset_id=1, force=False)
```

## Consequences

### Benefits

- **Reduced boilerplate**: No session plumbing through constructors
- **Cleaner APIs**: `DocumentRepository()` instead of `DocumentRepository(session)`
- **Simpler testing**: Set ambient session once per test, not per repository
- **Async-ready**: ContextVars work correctly with async/await
- **Transaction boundaries clear**: Entry point owns the session lifecycle

### Drawbacks

- **Implicit dependency**: Session is no longer visible in signatures (mitigated by raising clear error if not set)
- **Thread/async safety**: Must understand contextvars semantics (but this is actually safer than passing sessions around)
- **Migration effort**: Existing code needs updating (can be done incrementally)

### Migration Path

1. Add `session_context` module with ContextVar and helpers
2. Update repositories to accept `session=None` and fall back to ambient
3. Update entry points to set ambient session
4. Update tests to use ambient session fixture
5. Remove explicit session plumbing from internal code (keep optional param for advanced use)

## References

- Python contextvars: https://docs.python.org/3/library/contextvars.html
- SQLAlchemy session basics: https://docs.sqlalchemy.org/en/20/orm/session_basics.html
- Current session handling: `src/idx/store/database.py`
