"""LlamaIndex loader adapter for LoadingService.

This adapter wraps LlamaIndex loaders to produce LoadedDocument instances
compatible with the LoadingService, enforcing strict identity, metadata,
and no-chunking constraints.

Example:
    from llama_index.readers.web import SimpleWebPageReader

    reader = SimpleWebPageReader()
    adapter = LlamaIndexLoaderAdapter(
        loader=reader,
        content_type="text/html",
        load_kwargs={"urls": ["https://example.com"]},
    )

    async with create_application(config) as app:
        result = await app.loading.load_from_llamaindex("web-docs", adapter)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from pmd.metadata import ExtractedMetadata, get_default_profile_registry
from pmd.sources.content.base import DocumentReference, FetchResult

if TYPE_CHECKING:
    from .loading import LoadedDocument


class SupportsLoadData(Protocol):
    """Protocol for LlamaIndex-like loaders/readers."""

    def load_data(self, **kwargs: Any) -> list[Any]:
        ...


@dataclass
class LlamaIndexLoaderAdapter:
    """Adapter for LlamaIndex loaders that produces LoadedDocument instances.

    This adapter wraps a LlamaIndex loader and converts its output to
    LoadedDocument instances compatible with LoadingService.

    Key behaviors:
    - Constructs stable unique URIs for each document
    - Extracts metadata via pmd.metadata profiles
    - Preserves LlamaIndex metadata under `_llamaindex` key
    - By default, rejects loaders that return multiple documents (chunking)

    Attributes:
        loader: LlamaIndex loader instance with load_data() method.
        content_type: MIME type for loaded content (required).
        encoding: Text encoding (default: utf-8).
        uri_key: Metadata key to use for document path/URI (default: "path").
        title_key: Metadata key to use for document title (default: "title").
        namespace: URI scheme namespace (default: "llamaindex").
        allow_multiple: If True, allow loaders that return multiple documents.
        load_kwargs: Keyword arguments passed to loader.load_data().
    """

    loader: SupportsLoadData
    content_type: str
    encoding: str = "utf-8"
    uri_key: str = "path"
    title_key: str = "title"
    namespace: str = "llamaindex"
    allow_multiple: bool = False
    load_kwargs: dict[str, Any] = field(default_factory=dict)

    async def load_eager(self) -> list["LoadedDocument"]:
        """Load all documents from the loader.

        Returns:
            List of LoadedDocument instances (with source_collection_id=0, to be
            injected by LoadingService).

        Raises:
            ValueError: If allow_multiple=False and loader returns multiple documents.
            ValueError: If no unique identifier can be constructed for a document.
            ValueError: If duplicate URIs are detected when allow_multiple=True.
        """
        # Import here to avoid circular imports
        from .loading import LoadedDocument

        # Wrap sync loader in thread to avoid blocking
        start_time = time.perf_counter()
        docs = await asyncio.to_thread(self._load_data)
        fetch_duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Check multi-document constraint
        if len(docs) > 1 and not self.allow_multiple:
            raise ValueError(
                f"Loader returned {len(docs)} documents but allow_multiple=False. "
                "Set allow_multiple=True if this loader legitimately produces multiple documents, "
                "or use a different loader that returns single documents."
            )

        # Convert documents
        loaded_docs: list[LoadedDocument] = []
        seen_uris: set[str] = set()

        for idx, doc in enumerate(docs):
            loaded = self._convert_document(doc, idx, fetch_duration_ms)

            # Check for duplicate URIs
            if loaded.ref.uri in seen_uris:
                raise ValueError(
                    f"Duplicate URI detected: {loaded.ref.uri}. "
                    "Each document must have a unique identifier."
                )
            seen_uris.add(loaded.ref.uri)
            loaded_docs.append(loaded)

        return loaded_docs

    def _load_data(self) -> list[Any]:
        """Call the loader synchronously."""
        return list(self.loader.load_data(**self.load_kwargs))

    def _convert_document(
        self,
        doc: Any,
        index: int,
        fetch_duration_ms: int,
    ) -> "LoadedDocument":
        """Convert a LlamaIndex Document to a LoadedDocument.

        Args:
            doc: LlamaIndex Document object.
            index: Index of this document in the loader output.
            fetch_duration_ms: Time taken to load all documents.

        Returns:
            LoadedDocument instance.

        Raises:
            ValueError: If no unique identifier can be constructed.
        """
        from .loading import LoadedDocument

        # Extract content
        content = self._get_content(doc)

        # Get LlamaIndex metadata
        llama_metadata = getattr(doc, "metadata", {}) or {}

        # Construct URI and path
        uri, path = self._construct_uri(doc, llama_metadata, content, index)

        # Get or extract title
        title = self._get_title(doc, llama_metadata, content, path)

        # Create FetchResult
        fetch_result = FetchResult(
            content=content,
            content_type=self.content_type,
            metadata={"encoding": self.encoding, **llama_metadata},
        )

        # Create DocumentReference
        ref = DocumentReference(
            uri=uri,
            path=path,
            title=title,
            metadata=llama_metadata,
        )

        # Extract metadata via profiles, augmented with LlamaIndex metadata
        extracted_metadata = self._extract_metadata(content, path, llama_metadata)

        return LoadedDocument(
            ref=ref,
            fetch_result=fetch_result,
            title=title,
            fetch_duration_ms=fetch_duration_ms,
            source_collection_id=0,  # Will be injected by LoadingService
            extracted_metadata=extracted_metadata,
        )

    def _get_content(self, doc: Any) -> str:
        """Extract text content from a LlamaIndex Document."""
        # Try common attributes
        text = getattr(doc, "text", None)
        if text is not None:
            return str(text)

        # Try get_content method
        get_content = getattr(doc, "get_content", None)
        if callable(get_content):
            return str(get_content())

        # Fallback to string representation
        return str(doc)

    def _construct_uri(
        self,
        doc: Any,
        metadata: dict[str, Any],
        content: str,
        index: int,
    ) -> tuple[str, str]:
        """Construct URI and path for a document.

        Fallback chain:
        1. metadata[uri_key] if present
        2. doc.id_ if present
        3. Hash of namespace + content + sorted metadata

        Args:
            doc: LlamaIndex Document object.
            metadata: Document metadata dict.
            content: Document content.
            index: Document index (for error messages).

        Returns:
            Tuple of (uri, path).

        Raises:
            ValueError: If no identifier can be constructed.
        """
        # Try uri_key in metadata
        if self.uri_key in metadata:
            path = str(metadata[self.uri_key])
            return f"{self.namespace}://{path}", path

        # Try doc.id_
        doc_id = getattr(doc, "id_", None)
        if doc_id is not None:
            path = str(doc_id)
            return f"{self.namespace}://{path}", path

        # Fallback to hash
        hash_input = self.namespace + content + str(sorted(metadata.items()))
        content_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
        path = f"hash-{content_hash}"
        return f"{self.namespace}://{path}", path

    def _get_title(
        self,
        doc: Any,
        metadata: dict[str, Any],
        content: str,
        path: str,
    ) -> str:
        """Extract or derive title for a document.

        Args:
            doc: LlamaIndex Document object.
            metadata: Document metadata dict.
            content: Document content.
            path: Document path (fallback).

        Returns:
            Title string.
        """
        # Try title_key in metadata
        if self.title_key in metadata:
            return str(metadata[self.title_key])

        # Try extracting from markdown heading
        for line in content.split("\n"):
            if line.startswith("# "):
                return line[2:].strip()

        # Fallback to path
        return path

    def _extract_metadata(
        self,
        content: str,
        path: str,
        llama_metadata: dict[str, Any],
    ) -> ExtractedMetadata:
        """Extract metadata via profiles and augment with LlamaIndex metadata.

        Args:
            content: Document content.
            path: Document path.
            llama_metadata: LlamaIndex Document.metadata dict.

        Returns:
            ExtractedMetadata with LlamaIndex metadata under _llamaindex key.
        """
        try:
            registry = get_default_profile_registry()
            profile = registry.detect_or_default(content, path)
            extracted = profile.extract_metadata(content, path)

            if not extracted.extraction_source:
                extracted.extraction_source = profile.name

        except Exception:
            # If extraction fails, create minimal metadata
            extracted = ExtractedMetadata(tags=[], attributes={})

        # Augment with LlamaIndex metadata under reserved key
        if llama_metadata:
            if extracted.attributes is None:
                extracted.attributes = {}
            extracted.attributes["_llamaindex"] = dict(llama_metadata)

        return extracted
