# Move data access layer and DocumentCacher into storage

## Goal
- Consolidate the data access layer (`pmd.data`) into a single `pmd.store.facade` module.
- Rename classes to reflect their role: `IndexingData` → `IndexFacade`, etc.
- Relocate `DocumentCacher` from `pmd.services` into `pmd.store.caching`.
- Export facade classes directly from `pmd.store` (single canonical import path).

## Design rationale
- These "Data" classes are thin facades over repositories, reducing complexity for services.
- "Facade" is a clearer name for what they do.
- Flattening into `pmd.store` makes the architectural layer obvious.
- Single import path (`from pmd.store import IndexFacade`) avoids confusion.

## Non-goals
- No behavioral changes to repositories, services, or cache logic.
- No backward compatibility shims (clean break).

## Plan

### A. Consolidate data access layer into `pmd.store.facade`

1. Create `src/pmd/store/facade.py` containing all four classes:
   - `IndexFacade` (from `IndexingData`)
   - `LoadFacade` (from `LoadingData`)
   - `SearchFacade` (from `SearchData`)
   - `StatusFacade` (from `StatusData`)

2. Update class names and internal references within `facade.py`.

3. Export from `src/pmd/store/__init__.py`:
   ```python
   from pmd.store.facade import IndexFacade, LoadFacade, SearchFacade, StatusFacade
   ```

4. Update all import sites (use `rg` to find exhaustively):
   ```bash
   rg "from pmd\.data import" --type py
   rg "IndexingData|LoadingData|SearchData|StatusData" --type py
   ```

   Changes:
   - `from pmd.data import IndexingData` → `from pmd.store import IndexFacade`
   - Update variable names where appropriate (e.g., `indexing_data` → `index_facade`)

5. Update service constructors and their docstrings to use new names.

6. Delete `src/pmd/data/` package entirely.

### B. Move `DocumentCacher` into storage

1. Move `src/pmd/services/caching.py` → `src/pmd/store/caching.py`.

2. Export from `src/pmd/store/__init__.py`:
   ```python
   from pmd.store.caching import DocumentCacher
   ```

3. Update imports (use `rg` to find):
   ```bash
   rg "from pmd\.services\.caching import" --type py
   rg "from pmd\.services import.*DocumentCacher" --type py
   ```

4. Remove any re-export from `pmd/services/__init__.py`.

### C. Import path rule

**Single canonical path only.** After this change:
- `from pmd.store import IndexFacade, LoadFacade, SearchFacade, StatusFacade`
- `from pmd.store import DocumentCacher`

No alternative import paths should exist.

### D. Validation checklist

- [ ] No remaining imports from `pmd.data` anywhere
- [ ] No remaining imports from `pmd.services.caching`
- [ ] `pmd.store.__init__` exports: `IndexFacade`, `LoadFacade`, `SearchFacade`, `StatusFacade`, `DocumentCacher`
- [ ] All tests pass
- [ ] Service docstrings updated to reflect new names

### E. Sequencing

1. Create `facade.py` with renamed classes
2. Update `pmd.store.__init__.py` exports
3. Update all import sites and usages
4. Delete old `pmd.data` package
5. Move `caching.py` to `pmd.store`
6. Update caching imports
7. Run full test suite
