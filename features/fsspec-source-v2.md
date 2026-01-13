# Feature: fsspec-backed document source (v2)

## Summary

Replace direct `pathlib` filesystem access with an fsspec-backed implementation using
`file://` URIs. This establishes the abstraction layer for future remote backends
while maintaining current functionality.

## Goals

- Use fsspec as the filesystem abstraction for local `file://` URIs.
- Keep `DocumentReference.uri` as a canonical URI for persistence in
  `source_metadata.source_uri`.
- Preserve existing filesystem behavior and tests.
- Integrate with the existing `MetadataProfile` system for tag/frontmatter extraction.

## Non-goals

- Remote filesystem support (S3, GCS, etc.) - future work.
- Credential management via `storage_options` - future work.
- ETag-based change detection - future work.
- Writing to filesystems (read-only indexing only).
- Reworking document identity semantics (still use `ref.path`).

## Proposed Design

### New source type: `fsspec`

- Add `FSSpecSource` implementing `DocumentSource` protocol.
- Register it in `SourceRegistry` under `source_type="fsspec"`.
- Keep `filesystem` source intact initially; migrate later if desired.

### Configuration

- `SourceCollection.pwd` holds a base URI, e.g. `file:///abs/path`.
- `source_config` may include:
  - `glob_pattern` (default `**/*.md`)
  - `encoding` (default `utf-8`)

### URI handling

- Use `fsspec.core.url_to_fs(base_uri)` to get `fs` and `base_path`.
- Enumerate with `fs.glob(f"{base_path}/{glob_pattern}")`.
- Build canonical URIs as `f"file://{absolute_path}"`.
- Use `ref.path` as relative path from `base_path` for stable identity.

### Change detection

- Use `fs.info(path)["mtime"]` for modification detection.
- Store mtime in `ref.metadata` and persist via `SourceMetadata.extra`.
- `check_modified` compares stored mtime to current mtime.

## Implementation Sketches

### FSSpecConfig

```python
@dataclass
class FSSpecConfig:
    base_uri: str
    glob_pattern: str = "**/*.md"
    encoding: str = "utf-8"

    @classmethod
    def from_source_config(cls, config: SourceConfig) -> "FSSpecConfig":
        return cls(
            base_uri=config.uri,
            glob_pattern=config.get("glob_pattern", "**/*.md"),
            encoding=config.get("encoding", "utf-8"),
        )
```

### FSSpecSource

```python
class FSSpecSource(BaseDocumentSource):
    def __init__(self, config: SourceConfig) -> None:
        self._config = FSSpecConfig.from_source_config(config)
        self._fs, self._base_path = fsspec.core.url_to_fs(self._config.base_uri)

    def list_documents(self) -> Iterator[DocumentReference]:
        pattern = f"{self._base_path.rstrip('/')}/{self._config.glob_pattern}"
        for path in self._fs.glob(pattern):
            if self._fs.isdir(path):
                continue
            info = self._fs.info(path)
            rel_path = os.path.relpath(path, self._base_path)
            uri = f"file://{path}"
            yield DocumentReference(
                uri=uri,
                path=rel_path,
                metadata={"mtime": info.get("mtime")},
            )

    async def fetch_content(self, ref: DocumentReference) -> FetchResult:
        abs_path = os.path.join(self._base_path, ref.path)
        content = await asyncio.to_thread(self._read_file, abs_path)
        info = self._fs.info(abs_path)
        extracted = self._extract_metadata(content, ref.path)
        return FetchResult(
            content=content,
            metadata={"mtime": info.get("mtime")},
            extracted_metadata=extracted,
        )

    def _read_file(self, path: str) -> str:
        with self._fs.open(path, "rt", encoding=self._config.encoding) as f:
            return f.read()

    async def check_modified(
        self,
        ref: DocumentReference,
        stored_metadata: dict[str, Any],
    ) -> bool:
        abs_path = os.path.join(self._base_path, ref.path)
        try:
            info = self._fs.info(abs_path)
        except FileNotFoundError:
            return True  # Deleted files are "modified"
        return info.get("mtime") != stored_metadata.get("mtime")

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            supports_incremental=True,
            supports_etag=False,
        )
```

### SourceRegistry hook

```python
def fsspec_factory(source_collection: SourceCollection) -> DocumentSource:
    config = SourceConfig(
        uri=source_collection.pwd,
        extra=source_collection.source_config or {},
    )
    return FSSpecSource(config)

registry.register("fsspec", fsspec_factory)
```

## Implementation Plan

1. Add `fsspec` dependency to `pyproject.toml`.
2. Create `src/pmd/sources/content/fsspec.py` with config and source.
3. Register `fsspec` in `SourceRegistry`.
4. Add tests for:
   - `file://` base URI listing and fetching
   - Metadata fields populated from `fs.info`
   - Change detection using mtime
   - Integration with `MetadataProfile` extraction
5. Add brief usage note in `docs/` for the new source type.
6. (Future) Migrate `filesystem` collections to `fsspec` for single-path code.

## Error Handling

- Wrap `fs.glob()` errors in `SourceListError`.
- Wrap `fs.open()` / read errors in `SourceFetchError`.
- Handle `FileNotFoundError` in `check_modified` (file deleted between list and check).
- Handle `PermissionError` with clear error messages.

## Future Work (S3 and Remote Backends)

When adding S3 support, extend this implementation with:

- `storage_options` parameter in `FSSpecConfig` for credentials.
- ETag-based change detection (normalize `ETag`/`etag`/`e_tag` keys).
- Credential security (environment variable references, exclude from persistence).
- Pagination/backpressure for large bucket listings.
- Protocol tuple handling for chained filesystems.
