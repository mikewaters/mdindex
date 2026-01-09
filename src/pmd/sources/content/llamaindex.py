"""Adapter for using LlamaIndex readers/loaders as PMD sources.

This source wraps a LlamaIndex reader/loader instance (anything exposing a
``load_data(**kwargs) -> list[Document]`` method). Documents are loaded once
and cached for subsequent fetches. The adapter maps LlamaIndex ``Document``
fields to PMD's ``DocumentReference``/``FetchResult`` types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Protocol

from .base import (
    BaseDocumentSource,
    DocumentReference,
    FetchResult,
    SourceCapabilities,
    SourceFetchError,
)


class SupportsLoadData(Protocol):
    """Protocol for LlamaIndex-like loaders/readers."""

    def load_data(self, **kwargs: Any) -> Iterable[Any]:
        ...


@dataclass
class LlamaIndexSource(BaseDocumentSource):
    """Wrap a LlamaIndex reader/loader as a document source.

    Args:
        loader: A configured reader/loader instance with ``load_data``.
        load_kwargs: Optional keyword arguments passed to ``load_data``.
        path_key: Metadata key used to derive the document path (default: "path").
        title_key: Metadata key used to derive the title (default: "title").
    """

    loader: SupportsLoadData
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    path_key: str = "path"
    title_key: str = "title"

    def __post_init__(self) -> None:
        self._documents: list[Any] | None = None
        self._by_path: dict[str, Any] = {}

    def _ensure_loaded(self) -> None:
        if self._documents is not None:
            return
        docs = list(self.loader.load_data(**self.load_kwargs))
        self._documents = docs
        self._by_path = {}
        for idx, doc in enumerate(docs):
            meta = getattr(doc, "metadata", {}) or {}
            path = str(meta.get(self.path_key) or meta.get("id") or f"doc-{idx}")
            self._by_path[path] = doc

    def list_documents(self) -> Iterator[DocumentReference]:
        self._ensure_loaded()
        assert self._documents is not None
        for idx, doc in enumerate(self._documents):
            meta = getattr(doc, "metadata", {}) or {}
            path = str(meta.get(self.path_key) or meta.get("id") or f"doc-{idx}")
            title = meta.get(self.title_key)
            yield DocumentReference(
                uri=f"llamaindex://{path}",
                path=path,
                title=title,
                metadata=dict(meta),
            )

    async def fetch_content(self, ref: DocumentReference) -> FetchResult:
        self._ensure_loaded()
        doc = self._by_path.get(ref.path)
        if doc is None:
            raise SourceFetchError(ref.uri, "Document not loaded", retryable=False)
        text = getattr(doc, "text", None) or getattr(doc, "get_content", lambda: "")()
        # LlamaIndex documents commonly expose .text; fall back to str(doc)
        content = text if isinstance(text, str) else str(text)
        meta = getattr(doc, "metadata", {}) or {}
        return FetchResult(content=content, content_type="text/plain", metadata=dict(meta))

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(supports_incremental=False, supports_streaming=False)

    async def check_modified(self, ref, stored_metadata):
        # LlamaIndex loaders typically pull a fresh snapshot; treat as modified
        return True
