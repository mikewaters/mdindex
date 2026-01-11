"""Adapter for using LlamaIndex readers/loaders as PMD sources.

This source wraps a LlamaIndex reader/loader instance (anything exposing a
``load_data(**kwargs) -> list[Document]`` method). Documents are loaded once
and cached for subsequent fetches. The adapter maps LlamaIndex ``Document``
fields to PMD's ``DocumentReference``/``FetchResult`` types.

Supported Loaders (with llama-index extras installed):
    - SimpleWebPageReader: Scrape web pages
    - SimpleDirectoryReader: Load files from directories
    - PDFReader: Extract text from PDFs

Example:
    # Using a web reader
    from pmd.sources.content.llamaindex import create_web_loader

    source = create_web_loader(["https://example.com/docs"])
    for ref in source.list_documents():
        content = await source.fetch_content(ref)
        print(content.content)

    # Using a directory reader
    from pmd.sources.content.llamaindex import create_directory_loader

    source = create_directory_loader("/path/to/docs", recursive=True)
    for ref in source.list_documents():
        print(ref.path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol

from loguru import logger

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


# =============================================================================
# Factory Functions for Common Loaders
# =============================================================================


def _check_llamaindex() -> None:
    """Check if llama-index is installed."""
    try:
        import llama_index.core  # noqa: F401
    except ImportError:
        raise ImportError(
            "LlamaIndex is required for this loader. "
            "Install with: pip install pmd[loaders]"
        )


def create_web_loader(
    urls: list[str],
    html_to_text: bool = True,
) -> LlamaIndexSource:
    """Create a LlamaIndex source for web pages.

    Uses SimpleWebPageReader to scrape and load web content.

    Args:
        urls: List of URLs to scrape.
        html_to_text: If True, convert HTML to plain text (default: True).

    Returns:
        LlamaIndexSource configured for web scraping.

    Raises:
        ImportError: If llama-index-readers-web is not installed.

    Example:
        source = create_web_loader(["https://docs.python.org/3/"])
        for ref in source.list_documents():
            content = await source.fetch_content(ref)
    """
    _check_llamaindex()

    try:
        from llama_index.readers.web import SimpleWebPageReader
    except ImportError:
        raise ImportError(
            "llama-index-readers-web is required. "
            "Install with: pip install llama-index-readers-web"
        )

    loader = SimpleWebPageReader(html_to_text=html_to_text)

    return LlamaIndexSource(
        loader=loader,
        load_kwargs={"urls": urls},
        path_key="url",
        title_key="title",
    )


def create_directory_loader(
    input_dir: str,
    recursive: bool = True,
    required_exts: list[str] | None = None,
    exclude_hidden: bool = True,
) -> LlamaIndexSource:
    """Create a LlamaIndex source for directory files.

    Uses SimpleDirectoryReader to load files from a directory.
    This is useful for loading various file types (PDFs, text, etc.)
    using LlamaIndex's built-in parsers.

    Args:
        input_dir: Path to directory to load.
        recursive: If True, load files recursively (default: True).
        required_exts: Optional list of file extensions to include.
        exclude_hidden: If True, exclude hidden files (default: True).

    Returns:
        LlamaIndexSource configured for directory loading.

    Raises:
        ImportError: If llama-index-readers-file is not installed.

    Example:
        source = create_directory_loader(
            "/path/to/docs",
            recursive=True,
            required_exts=[".pdf", ".md", ".txt"]
        )
    """
    _check_llamaindex()

    try:
        from llama_index.core import SimpleDirectoryReader
    except ImportError:
        raise ImportError(
            "llama-index-core is required. "
            "Install with: pip install llama-index-core"
        )

    loader = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=recursive,
        required_exts=required_exts,
        exclude_hidden=exclude_hidden,
    )

    return LlamaIndexSource(
        loader=loader,
        path_key="file_path",
        title_key="file_name",
    )


def create_custom_loader(
    loader: SupportsLoadData,
    load_kwargs: dict[str, Any] | None = None,
    path_key: str = "path",
    title_key: str = "title",
) -> LlamaIndexSource:
    """Create a LlamaIndex source with a custom loader.

    This factory allows using any LlamaHub loader that implements
    the load_data() method.

    Args:
        loader: A configured LlamaIndex loader instance.
        load_kwargs: Optional kwargs to pass to load_data().
        path_key: Metadata key for document path (default: "path").
        title_key: Metadata key for document title (default: "title").

    Returns:
        LlamaIndexSource wrapping the custom loader.

    Example:
        from llama_index.readers.github import GithubRepositoryReader

        loader = GithubRepositoryReader(
            github_token="...",
            owner="openai",
            repo="whisper",
        )
        source = create_custom_loader(loader, path_key="file_path")
    """
    return LlamaIndexSource(
        loader=loader,
        load_kwargs=load_kwargs or {},
        path_key=path_key,
        title_key=title_key,
    )
