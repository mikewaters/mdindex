# Feature: fsspec-backed document source

## Summary
Introduce an fsspec-based document source so document URIs can be resolved via
`file://` today and `s3://` (or other schemes) later, while keeping the
DocumentSource interface and SourceMetadata storage intact.

## Goals
- Support `file://` and `s3://` URIs through a single source implementation.
- Keep `DocumentReference.uri` as a canonical URI for persistence in
  `source_metadata.source_uri`.
- Preserve existing filesystem behavior and tests as much as possible.
- Allow per-source `storage_options` for credentials and backend settings.

## Non-goals
- Writing to remote stores (read-only indexing only).
- Full-blown caching or streaming optimization beyond what fsspec provides.
- Reworking document identity semantics (still use `ref.path`).

## Proposed Design

### New source type: `fsspec`
- Add `FSSpecSource` implementing `DocumentSource`.
- Register it in `SourceRegistry` under `source_type="fsspec"`.
- Keep `filesystem` source intact initially; migrate later if desired.

### Configuration
- `SourceCollection.pwd` holds a base URI, e.g. `file:///abs/path` or
  `s3://bucket/prefix`.
- `source_config` may include:
  - `glob_pattern` (default `**/*.md`)
  - `encoding` (default `utf-8`)
  - `storage_options` (dict passed to fsspec)

### URI handling
- Use `fsspec.core.url_to_fs(base_uri, **storage_options)` to get `fs` and
  `base_path`.
- Enumerate with `fs.glob(f"{base_path}/{glob_pattern}")`.
- Build canonical URIs for refs as `f"{fs.protocol}://{path}"` when possible,
  or reconstruct from `base_uri` + relative path.
- Use `ref.path` as relative path from `base_path` for stable identity.

### Metadata and change detection
- Prefer `fs.info(path)` for size, `mtime`, and ETag/Last-Modified when
  available.
- Store these in `ref.metadata` and persist via `SourceMetadata.extra`.
- `check_modified` compares stored `etag` or `mtime` if present, otherwise
  returns True.

## Sketches

### FSSpecConfig
```python
@dataclass
class FSSpecConfig:
    base_uri: str
    glob_pattern: str = "**/*.md"
    encoding: str = "utf-8"
    storage_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_source_config(cls, config: SourceConfig) -> "FSSpecConfig":
        return cls(
            base_uri=config.uri,
            glob_pattern=config.get("glob_pattern", "**/*.md"),
            encoding=config.get("encoding", "utf-8"),
            storage_options=config.get("storage_options", {}),
        )
```

### FSSpecSource (high level)
```python
class FSSpecSource(BaseDocumentSource):
    def __init__(self, config: SourceConfig) -> None:
        self._config = FSSpecConfig.from_source_config(config)
        self._fs, self._base_path = fsspec.core.url_to_fs(
            self._config.base_uri,
            **self._config.storage_options,
        )

    def list_documents(self) -> Iterator[DocumentReference]:
        pattern = f"{self._base_path.rstrip('/')}/{self._config.glob_pattern}"
        for path in self._fs.glob(pattern):
            if self._fs.isdir(path):
                continue
            info = self._safe_info(path)
            rel_path = self._rel_path(path)
            uri = self._to_uri(path)
            yield DocumentReference(uri=uri, path=rel_path, metadata=info)

    async def fetch_content(self, ref: DocumentReference) -> FetchResult:
        with fsspec.open(ref.uri, "rt", encoding=self._config.encoding,
                         **self._config.storage_options) as handle:
            content = handle.read()
        info = self._safe_info(self._abs_path(ref.path))
        return FetchResult(content=content, content_type=..., metadata=info)

    async def check_modified(self, ref: DocumentReference,
                             stored_metadata: dict[str, Any]) -> bool:
        info = self._safe_info(self._abs_path(ref.path))
        return _changed(info, stored_metadata)
```

### SourceRegistry hook
```python
from .fsspec import FSSpecSource

def fsspec_factory(source_collection: SourceCollection) -> DocumentSource:
    config = SourceConfig(
        uri=source_collection.get_source_uri(),
        extra=source_collection.get_source_config_dict(),
    )
    return FSSpecSource(config)

registry.register("fsspec", fsspec_factory)
```

## Implementation Plan
1. Add `fsspec` dependency (and optional extra for `s3fs`).
2. Create `src/pmd/sources/content/fsspec.py` with config + source.
3. Register `fsspec` in `SourceRegistry`.
4. Add tests for:
   - file:// base URI listing and fetching
   - metadata fields populated from `fs.info`
   - change detection using mtime or etag
5. Add docs in `docs/` noting how to configure `source_config.storage_options`.
6. (Optional) migrate `filesystem` collections to `fsspec` for single-path code.

## Risks / Open Questions
- Path normalization for different FS protocols (ensure stable `ref.path`).
- `fs.info` key naming variance across backends (ETag vs etag, LastModified vs
  last_modified).
- Credentials handling for S3 (`storage_options` vs env vars).
- Performance of `fs.glob` on large buckets.

## Rollout
- Introduce as a new `source_type` without changing default behavior.
- Add a migration note for users who want to switch existing collections.
