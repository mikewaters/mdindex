"""idx.transform.llama - LlamaIndex TransformComponent wrappers.

Provides LlamaIndex-compatible transform components that wrap idx functionality
for use in ingestion pipelines.

Example usage:
    from llama_index.core.ingestion import IngestionPipeline
    from idx.transform.llama import TextNormalizerTransform, FTSIndexerTransform
    from idx.store.fts import FTSManager

    pipeline = IngestionPipeline(
        transformations=[
            TextNormalizerTransform(),
            FTSIndexerTransform(fts_manager=fts_manager),
        ]
    )
    nodes = pipeline.run(documents=documents)
"""

from collections.abc import Callable
from typing import Any

from llama_index.core.schema import BaseNode, TransformComponent
from sqlalchemy.orm import Session

from idx.store.fts import FTSManager
from idx.transform.normalize import TextNormalizer

__all__ = [
    "TextNormalizerTransform",
    "FTSIndexerTransform",
]


class TextNormalizerTransform(TransformComponent):
    """LlamaIndex TransformComponent that normalizes text content of nodes.

    Wraps idx.transform.normalize.TextNormalizer to integrate with
    LlamaIndex ingestion pipelines. Normalizes line endings, collapses
    excessive whitespace, strips BOM, and performs other text cleanup.

    Attributes:
        strip_bom: Remove UTF-8 BOM if present.
        normalize_line_endings: Convert \\r\\n and \\r to \\n.
        collapse_blank_lines: Limit consecutive blank lines.
        max_consecutive_blank_lines: Maximum blank lines to allow.
        strip_trailing_whitespace: Remove trailing whitespace from lines.
    """

    strip_bom: bool = True
    normalize_line_endings: bool = True
    collapse_blank_lines: bool = True
    max_consecutive_blank_lines: int = 2
    strip_trailing_whitespace: bool = True

    def __init__(
        self,
        *,
        strip_bom: bool = True,
        normalize_line_endings: bool = True,
        collapse_blank_lines: bool = True,
        max_consecutive_blank_lines: int = 2,
        strip_trailing_whitespace: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the text normalizer transform.

        Args:
            strip_bom: Remove UTF-8 BOM if present.
            normalize_line_endings: Convert \\r\\n and \\r to \\n.
            collapse_blank_lines: Limit consecutive blank lines.
            max_consecutive_blank_lines: Maximum blank lines to allow.
            strip_trailing_whitespace: Remove trailing whitespace from lines.
            **kwargs: Additional arguments passed to TransformComponent.
        """
        super().__init__(**kwargs)
        self.strip_bom = strip_bom
        self.normalize_line_endings = normalize_line_endings
        self.collapse_blank_lines = collapse_blank_lines
        self.max_consecutive_blank_lines = max_consecutive_blank_lines
        self.strip_trailing_whitespace = strip_trailing_whitespace

    def __call__(
        self,
        nodes: list[BaseNode],
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Normalize the text content of each node.

        Creates a TextNormalizer with the configured options and applies
        it to each node's text content.

        Args:
            nodes: List of nodes to normalize.
            **kwargs: Additional arguments (unused).

        Returns:
            The same nodes with normalized text content.
        """
        normalizer = TextNormalizer(
            strip_bom=self.strip_bom,
            normalize_line_endings=self.normalize_line_endings,
            collapse_blank_lines=self.collapse_blank_lines,
            max_consecutive_blank_lines=self.max_consecutive_blank_lines,
            strip_trailing_whitespace=self.strip_trailing_whitespace,
        )

        for node in nodes:
            original_text = node.get_content()
            normalized_text = normalizer.normalize(original_text)
            # Use set_content() which works for both Document and TextNode
            node.set_content(normalized_text)

        return nodes


class FTSIndexerTransform(TransformComponent):
    """LlamaIndex TransformComponent that indexes nodes in FTS5.

    Passthrough transform that updates the FTS5 full-text search index
    for each node, then returns the nodes unchanged. This allows FTS
    indexing to be integrated into LlamaIndex ingestion pipelines.

    The transform requires either an FTSManager instance or a session
    factory callable that produces SQLAlchemy sessions.

    Attributes:
        doc_id_key: Metadata key containing the document ID (default: "doc_id").
        path_key: Metadata key containing the document path (default: "path").
    """

    doc_id_key: str = "doc_id"
    path_key: str = "path"

    # These are not Pydantic fields - set in __init__
    _fts_manager: FTSManager | None = None
    _session_factory: Callable[[], Session] | None = None

    def __init__(
        self,
        *,
        fts_manager: FTSManager | None = None,
        session_factory: Callable[[], Session] | None = None,
        doc_id_key: str = "doc_id",
        path_key: str = "path",
        **kwargs: Any,
    ) -> None:
        """Initialize the FTS indexer transform.

        Provide either an FTSManager instance or a session factory.
        If a session factory is provided, a new FTSManager will be created
        for each call.

        Args:
            fts_manager: Pre-configured FTSManager instance.
            session_factory: Callable that returns a SQLAlchemy session.
            doc_id_key: Metadata key for document ID (used as FTS rowid).
            path_key: Metadata key for document path.
            **kwargs: Additional arguments passed to TransformComponent.

        Raises:
            ValueError: If neither fts_manager nor session_factory is provided.
        """
        super().__init__(**kwargs)

        if fts_manager is None and session_factory is None:
            raise ValueError(
                "Either fts_manager or session_factory must be provided"
            )

        self._fts_manager = fts_manager
        self._session_factory = session_factory
        self.doc_id_key = doc_id_key
        self.path_key = path_key

    def _get_fts_manager(self) -> FTSManager:
        """Get the FTSManager instance to use.

        Returns:
            FTSManager instance.
        """
        if self._fts_manager is not None:
            return self._fts_manager

        # Create from session factory
        if self._session_factory is not None:
            session = self._session_factory()
            return FTSManager(session)

        # Should never reach here due to __init__ validation
        raise RuntimeError("No FTSManager or session_factory available")

    def _get_doc_id(self, node: BaseNode) -> int | None:
        """Extract document ID from node metadata or ref_doc_id.

        Checks metadata first using doc_id_key, then falls back to
        ref_doc_id if available.

        Args:
            node: The node to extract doc_id from.

        Returns:
            Document ID as integer, or None if not found.
        """
        # Check metadata first
        if node.metadata and self.doc_id_key in node.metadata:
            doc_id = node.metadata[self.doc_id_key]
            if isinstance(doc_id, int):
                return doc_id
            # Try to convert string to int
            try:
                return int(doc_id)
            except (ValueError, TypeError):
                pass

        # Fall back to ref_doc_id
        if hasattr(node, "ref_doc_id") and node.ref_doc_id is not None:
            try:
                return int(node.ref_doc_id)
            except (ValueError, TypeError):
                pass

        return None

    def _get_path(self, node: BaseNode) -> str:
        """Extract document path from node metadata.

        Args:
            node: The node to extract path from.

        Returns:
            Document path, or empty string if not found.
        """
        if node.metadata and self.path_key in node.metadata:
            return str(node.metadata[self.path_key])
        return ""

    def __call__(
        self,
        nodes: list[BaseNode],
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Index each node in the FTS5 index.

        For each node, extracts doc_id and path from metadata and
        calls FTSManager.upsert() to update the FTS index. Nodes
        without valid doc_id are skipped.

        Args:
            nodes: List of nodes to index.
            **kwargs: Additional arguments (unused).

        Returns:
            The same nodes unchanged (passthrough).
        """
        fts_manager = self._get_fts_manager()

        for node in nodes:
            doc_id = self._get_doc_id(node)
            if doc_id is None:
                # Skip nodes without valid doc_id
                continue

            path = self._get_path(node)
            body = node.text if hasattr(node, "text") else node.get_content()

            fts_manager.upsert(doc_id=doc_id, path=path, body=body)

        return nodes
