# Feature: Multi-Glob Pattern Support for Collections

## Summary

Extend collection creation to support multiple file glob patterns, allowing users to
specify which files to include or exclude using industry-standard syntax inspired by
`.gitignore` and tools like `ripgrep`.

## Motivation

Current collections only support a single glob pattern (e.g., `**/*.md`). Users need to:

- Index multiple file types: `*.md`, `*.txt`, `*.rst`
- Exclude specific directories: `!**/node_modules/**`
- Combine inclusion and exclusion rules
- Create precise file selections without multiple collections

## Industry Pattern Analysis

### gitignore Syntax
[Git gitignore documentation](https://git-scm.com/docs/gitignore)

- One pattern per line (newline-delimited)
- Patterns are evaluated in order; later patterns override earlier ones
- `!` prefix negates a pattern (re-includes excluded files)
- `**` matches any number of directories
- Trailing `/` matches only directories

### ripgrep `-g/--glob` Flag
[ripgrep Guide](https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md)

- Multiple `-g` flags are OR'd together (file matches if ANY pattern matches)
- `!` prefix excludes files matching the pattern
- Follows gitignore glob syntax
- No native AND logic; [AND requires external tooling](https://github.com/BurntSushi/ripgrep/issues/1046)

### Key Insight: OR vs AND Semantics

| Semantic | Meaning | Use Case |
|----------|---------|----------|
| OR (union) | File matches if it satisfies ANY pattern | Multiple file types: `*.md`, `*.txt` |
| AND (intersection) | File matches if it satisfies ALL patterns | Path + extension: `docs/**` AND `*.md` |

Most tools (ripgrep, fd, glob libraries) use **OR semantics** for multiple patterns.
AND semantics are typically achieved through:
- Compound patterns: `docs/**/*.md` (path AND extension in one pattern)
- Chained filtering (external)

## Proposed Design

### Pattern Syntax

Adopt a gitignore-inspired syntax with explicit OR semantics:

```
# Include patterns (OR'd together)
**/*.md
**/*.txt
docs/**/*.rst

# Exclude patterns (applied after includes)
!**/drafts/**
!**/node_modules/**
```

### Storage Format

Store patterns as a JSON array in `source_config.glob_patterns`:

```json
{
  "glob_patterns": [
    "**/*.md",
    "**/*.txt",
    "!**/node_modules/**"
  ]
}
```

**Rationale:**
- JSON array is explicit and unambiguous
- Easy to parse/serialize
- Supports arbitrary patterns without delimiter conflicts
- Consistent with existing `source_config` JSON pattern

### CLI Interface

Support multiple `-g/--glob` flags (ripgrep-style):

```bash
# Single pattern (current behavior, still works)
pmd collection add docs ./path -g "**/*.md"

# Multiple patterns (OR'd)
pmd collection add docs ./path -g "**/*.md" -g "**/*.txt"

# With exclusions
pmd collection add docs ./path -g "**/*.md" -g "!**/drafts/**"

# Mixed include/exclude
pmd collection add myproject ./src \
  -g "**/*.py" \
  -g "**/*.pyi" \
  -g "!**/__pycache__/**" \
  -g "!**/test_*.py"
```

**Why multiple flags over comma-separated?**
- Avoids conflicts with commas in patterns (rare but possible)
- Consistent with ripgrep, fd, and other CLI tools
- Shell quoting is clearer per-pattern
- argparse `nargs="*"` or `action="append"` handles this naturally

### Pattern Evaluation Algorithm

```python
def matches_patterns(file_path: str, patterns: list[str]) -> bool:
    """
    Evaluate file against pattern list.

    Algorithm:
    1. Separate include patterns (no prefix) from exclude patterns (! prefix)
    2. File must match at least one include pattern (OR)
    3. File must NOT match any exclude pattern

    Returns True if file should be indexed.
    """
    includes = [p for p in patterns if not p.startswith("!")]
    excludes = [p[1:] for p in patterns if p.startswith("!")]

    # Must match at least one include pattern
    if not any(fnmatch(file_path, inc) for inc in includes):
        return False

    # Must not match any exclude pattern
    if any(fnmatch(file_path, exc) for exc in excludes):
        return False

    return True
```

### Default Pattern

When no `-g` flags provided, use `["**/*.md"]` (preserves current behavior).

## Implementation Plan

### Phase 1: Core Data Model

**Files affected:**
- `src/pmd/core/types.py`
- `src/pmd/store/collections.py`
- `src/pmd/store/models.py`

**Changes:**
1. Keep `SourceCollection.glob_pattern` for display/single-pattern compat
2. Add `glob_patterns: list[str]` property that reads from `source_config`
3. Update `SourceCollectionRepository.create()` to accept `glob_patterns: list[str]`
4. Store patterns in `source_config["glob_patterns"]`

### Phase 2: CLI Updates

**Files affected:**
- `src/pmd/cli/main.py`
- `src/pmd/cli/commands/collection.py`

**Changes:**
1. Change `-g/--glob` to `action="append"` to collect multiple values
2. Update `_add_source_collection()` to pass list to repository
3. Update `collection list` to display patterns (comma-separated for brevity)
4. Update `collection show` to display full pattern list

### Phase 3: Source Implementation

**Files affected:**
- `src/pmd/sources/content/filesystem.py`
- `src/pmd/sources/content/fsspec.py` (if exists)

**Changes:**
1. Update `FileSystemConfig` to use `glob_patterns: list[str]`
2. Update `list_documents()` to:
   - Iterate all include patterns
   - Collect unique file paths (deduplication)
   - Filter out files matching exclude patterns
3. Add debug logging for pattern matching

### Phase 4: Documentation & Tests

**Files affected:**
- `docs/COMMAND_LINE.md`
- `tests/unit/store/test_collections.py`
- `tests/unit/sources/content/test_filesystem.py`

**Changes:**
1. Document multi-pattern CLI syntax
2. Add unit tests for pattern evaluation logic
3. Add integration tests for multi-pattern collections

## Code Sketches

### Updated CLI Argument

```python
add_parser.add_argument(
    "-g",
    "--glob",
    action="append",
    dest="globs",
    metavar="PATTERN",
    help="File glob pattern (can be specified multiple times, use ! prefix to exclude)",
)
```

### Pattern Matching Implementation

```python
import fnmatch
from pathlib import Path
from typing import Iterator


class MultiGlobMatcher:
    """Match files against multiple glob patterns with include/exclude semantics."""

    def __init__(self, patterns: list[str]) -> None:
        self.includes = [p for p in patterns if not p.startswith("!")]
        self.excludes = [p[1:] for p in patterns if p.startswith("!")]

        if not self.includes:
            raise ValueError("At least one include pattern required")

    def matches(self, path: str) -> bool:
        """Check if path matches the pattern set."""
        # Normalize path separators
        path = path.replace("\\", "/")

        # Must match at least one include
        if not any(self._glob_match(path, inc) for inc in self.includes):
            return False

        # Must not match any exclude
        if any(self._glob_match(path, exc) for exc in self.excludes):
            return False

        return True

    def _glob_match(self, path: str, pattern: str) -> bool:
        """Match path against a single glob pattern."""
        # Handle ** for recursive matching
        if "**" in pattern:
            # Use pathlib's match for ** support
            return Path(path).match(pattern)
        return fnmatch.fnmatch(path, pattern)


def list_documents_multi_glob(
    base_path: Path,
    patterns: list[str],
) -> Iterator[Path]:
    """
    List documents matching multiple glob patterns.

    Yields unique file paths matching any include pattern
    and not matching any exclude pattern.
    """
    matcher = MultiGlobMatcher(patterns)
    seen: set[Path] = set()

    # Iterate each include pattern
    for pattern in matcher.includes:
        for file_path in base_path.glob(pattern):
            if file_path.is_file() and file_path not in seen:
                # Check against full pattern set (for excludes)
                rel_path = file_path.relative_to(base_path)
                if matcher.matches(str(rel_path)):
                    seen.add(file_path)
                    yield file_path
```

### Updated FileSystemSource

```python
class FileSystemConfig:
    base_path: Path
    glob_patterns: list[str]  # Changed from glob_pattern: str
    encoding: str = "utf-8"

    @classmethod
    def from_source_config(cls, pwd: str, config: dict) -> "FileSystemConfig":
        # Handle both old single-pattern and new multi-pattern
        patterns = config.get("glob_patterns")
        if patterns is None:
            # Fallback to legacy single pattern
            patterns = [config.get("glob_pattern", "**/*.md")]
        return cls(
            base_path=Path(pwd),
            glob_patterns=patterns,
            encoding=config.get("encoding", "utf-8"),
        )
```

## Migration Strategy

No migration needed. The design is backward-compatible:

1. Existing collections with single `glob_pattern` continue to work
2. `FileSystemConfig.from_source_config()` handles both formats
3. New collections store patterns in `source_config["glob_patterns"]`

## Edge Cases

### Empty Pattern List
If user provides only exclude patterns (`-g "!**/test/**"`), error:
"At least one include pattern required"

### Overlapping Patterns
`-g "**/*.md" -g "docs/**/*.md"` - Same file may match multiple patterns.
Deduplication ensures each file is yielded once.

### Pattern Order
Unlike gitignore, pattern order doesn't matter for this implementation:
- All includes are OR'd
- All excludes are checked after includes

### Glob Syntax Differences
Python's `fnmatch` and `pathlib.glob` have minor differences from gitignore.
Document any differences in CLI help.

## Future Considerations

### AND Semantics (Optional Enhancement)
If users need AND logic (file must match ALL patterns), consider:
- Modifier syntax: `+docs/**` for AND (vs default OR)
- Separate flag: `--glob-and` vs `--glob`
- Compound patterns cover most AND cases: `docs/**/*.md`

### Pattern Files
Support reading patterns from a file (like `.gitignore`):
```bash
pmd collection add docs ./path --glob-file .pmdinclude
```

### Performance
For large directories with many patterns:
- Consider using `wcmatch` library for optimized glob matching
- Profile pattern evaluation for collections with 10+ patterns

## References

- [Git gitignore documentation](https://git-scm.com/docs/gitignore)
- [ripgrep GUIDE.md](https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md)
- [ripgrep AND logic discussion](https://github.com/BurntSushi/ripgrep/issues/1046)
- [Python fnmatch module](https://docs.python.org/3/library/fnmatch.html)
- [Python pathlib.Path.glob](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob)
