"""idx.transform.llama - LlamaIndex TransformComponent wrappers.

Provides LlamaIndex-compatible transform components that wrap idx functionality
for use in ingestion pipelines. Uses ambient session via contextvars.

Example usage:
    from llama_index.core.ingestion import IngestionPipeline
    from idx.transform.llama import TextNormalizerTransform, PersistenceTransform
    from idx.store import get_session, use_session

    with get_session() as session:
        with use_session(session):
            persist = PersistenceTransform(dataset_id=1)
            pipeline = IngestionPipeline(
                transformations=[TextNormalizerTransform(), persist]
            )
            nodes = pipeline.run(documents=documents)
            print(f"Created: {persist.stats.created}")
"""

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json

from llama_index.core.schema import BaseNode, TransformComponent
from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.store.fts import FTSManager
from idx.store.fts_chunk import FTSChunkManager
from idx.store.models import Document
from idx.store.repositories import DocumentRepository
from idx.store.session_context import current_session
from idx.transform.normalize import TextNormalizer

__all__ = [
    "TextNormalizerTransform",
    "FTSIndexerTransform",
    "PersistenceTransform",
    "PersistenceStats",
    "ChunkPersistenceTransform",
    "ChunkPersistenceStats",
]

logger = get_logger(__name__)


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

    Uses ambient session via contextvars. The session must be set
    via `use_session()` before calling the transform.

    Attributes:
        doc_id_key: Metadata key containing the document ID (default: "doc_id").
        path_key: Metadata key containing the document path (default: "path").
    """

    doc_id_key: str = "doc_id"
    path_key: str = "path"

    # These are not Pydantic fields - set in __init__
    _fts_manager: FTSManager | None = None

    def __init__(
        self,
        *,
        fts_manager: FTSManager | None = None,
        doc_id_key: str = "doc_id",
        path_key: str = "path",
        **kwargs: Any,
    ) -> None:
        """Initialize the FTS indexer transform.

        Can optionally provide a pre-configured FTSManager instance.
        If not provided, a new FTSManager will be created using the
        ambient session from current_session().

        Args:
            fts_manager: Optional pre-configured FTSManager instance.
            doc_id_key: Metadata key for document ID (used as FTS rowid).
            path_key: Metadata key for document path.
            **kwargs: Additional arguments passed to TransformComponent.
        """
        raise NotImplementedError()
        super().__init__(**kwargs)

        self._fts_manager = fts_manager
        self.doc_id_key = doc_id_key
        self.path_key = path_key

    def _get_fts_manager(self) -> FTSManager:
        """Get the FTSManager instance to use.

        Returns:
            FTSManager instance. Uses explicit instance if provided,
            otherwise creates one using ambient session.
        """
        if self._fts_manager is not None:
            return self._fts_manager

        # Create using ambient session
        return FTSManager()

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


@dataclass
class PersistenceStats:
    """Statistics from a persistence operation.

    Attributes:
        created: Number of new documents created.
        updated: Number of existing documents updated.
        skipped: Number of unchanged documents skipped.
        failed: Number of documents that failed to process.
        errors: List of error messages for failed documents.
    """

    created: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        """Total number of documents processed (excluding failures)."""
        return self.created + self.updated + self.skipped

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.created = 0
        self.updated = 0
        self.skipped = 0
        self.failed = 0
        self.errors = []


def _compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class PersistenceTransform(TransformComponent):
    """LlamaIndex TransformComponent that persists nodes to the database and FTS.

    This transform handles the full persistence workflow within the pipeline:
    1. Check if document exists (by path from node metadata)
    2. Compare content hash to detect changes
    3. Create new documents or update existing ones
    4. Update FTS index
    5. Track statistics

    The transform uses the ambient session from the current context (set via
    `use_session()`). Statistics are available via the `stats` property.

    Example:
        # With ambient session (preferred)
        with get_session() as session:
            with use_session(session):
                persist = PersistenceTransform(dataset_id=dataset_id)
                pipeline = IngestionPipeline(
                    transformations=[TextNormalizerTransform(), persist]
                )
                pipeline.run(documents=documents)
                print(f"Created {persist.stats.created} documents")

    Attributes:
        stats: PersistenceStats with counts of created/updated/skipped/failed.
    """

    # Private attributes (not Pydantic fields)
    _dataset_id: int = 0
    _force: bool = False
    _path_key: str = "relative_path"
    _stats: PersistenceStats | None = None

    def __init__(
        self,
        dataset_id: int,
        *,
        force: bool = False,
        path_key: str = "relative_path",
        **kwargs: Any,
    ) -> None:
        """Initialize the persistence transform.

        Args:
            dataset_id: ID of the dataset to persist documents to.
            force: If True, update all documents even if unchanged.
            path_key: Metadata key for the document path (default: "relative_path").
            **kwargs: Additional arguments passed to TransformComponent.
        """
        super().__init__(**kwargs)
        self._dataset_id = dataset_id
        self._force = force
        self._path_key = path_key
        self._stats = PersistenceStats()

    @property
    def stats(self) -> PersistenceStats:
        """Get persistence statistics."""
        if self._stats is None:
            self._stats = PersistenceStats()
        return self._stats

    def __call__(
        self,
        nodes: list[BaseNode],
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Persist each node to the database and FTS index.

        For each node:
        - Extracts path from metadata
        - Computes content hash
        - Creates or updates the Document record
        - Updates FTS index
        - Tracks statistics

        Requires an ambient session to be set via `use_session()`.

        Args:
            nodes: List of nodes to persist.
            **kwargs: Additional arguments (unused).

        Returns:
            The same nodes with doc_id added to metadata.

        Raises:
            SessionNotSetError: If no ambient session is set.
        """
        logger.info(f"PersistenceTransform: persisting {len(nodes)} documents")

        # Reset stats for this run
        self.stats.reset()

        # Get ambient session - raises SessionNotSetError if not set
        session = current_session()

        # Use ambient session for repositories
        doc_repo = DocumentRepository()
        fts = FTSManager()

        # Pre-fetch existing documents for efficiency
        existing_paths = doc_repo.list_paths_by_dataset(
            self._dataset_id, active_only=False
        )
        existing_docs: dict[str, Document | None] = {
            path: doc_repo.get_by_path(self._dataset_id, path)
            for path in existing_paths
        }

        for node in nodes:
            try:
                self._process_node(
                    session=session,
                    doc_repo=doc_repo,
                    fts=fts,
                    node=node,
                    existing_docs=existing_docs,
                )
            except Exception as e:
                path = self._get_path(node)
                logger.error(f"Failed to persist {path}: {e}")
                self.stats.failed += 1
                self.stats.errors.append(f"{path}: {e}")

        session.flush()

        logger.info(
            f"PersistenceTransform complete: "
            f"created={self.stats.created}, updated={self.stats.updated}, "
            f"skipped={self.stats.skipped}, failed={self.stats.failed}"
        )

        return nodes

    def _get_path(self, node: BaseNode) -> str:
        """Extract document path from node metadata."""
        if node.metadata and self._path_key in node.metadata:
            return str(node.metadata[self._path_key])
        # Fall back to node_id (which we set to relative_path)
        return node.node_id

    def _get_metadata_json(self, node: BaseNode) -> str | None:
        """Extract metadata as JSON string."""
        if not node.metadata:
            return None
        # Filter out internal keys
        filtered = {
            k: v for k, v in node.metadata.items()
            if not k.startswith("_") and k not in ("file_path",)
        }
        return json.dumps(filtered) if filtered else None

    def _process_node(
        self,
        session: Session,
        doc_repo: DocumentRepository,
        fts: FTSManager,
        node: BaseNode,
        existing_docs: dict[str, Document | None],
    ) -> None:
        """Process a single node for persistence."""
        path = self._get_path(node)
        body = node.get_content()
        content_hash = _compute_content_hash(body)
        metadata_json = self._get_metadata_json(node)

        # Extract source metadata for etag/last_modified
        etag = node.metadata.get("etag") if node.metadata else None
        last_modified_str = node.metadata.get("last_modified") if node.metadata else None
        last_modified = None
        if last_modified_str:
            from datetime import datetime
            try:
                last_modified = datetime.fromisoformat(last_modified_str)
            except (ValueError, TypeError):
                pass

        existing = existing_docs.get(path)

        if existing is not None:
            # Document exists - check if changed
            if not self._force and existing.content_hash == content_hash:
                # Unchanged - skip (but ensure active if was inactive)
                if not existing.active:
                    existing.active = True
                    existing.metadata_json = metadata_json
                    session.flush()
                    fts.upsert(existing.id, path, body)
                    self.stats.updated += 1
                    # Add doc_id to node metadata for downstream use
                    node.metadata["doc_id"] = existing.id
                else:
                    self.stats.skipped += 1
                    node.metadata["doc_id"] = existing.id
                return

            # Changed or force - update
            existing.content_hash = content_hash
            existing.body = body
            existing.etag = etag
            existing.last_modified = last_modified
            existing.active = True
            existing.metadata_json = metadata_json
            session.flush()

            fts.upsert(existing.id, path, body)
            node.metadata["doc_id"] = existing.id

            logger.debug(f"Updated document: {path}")
            self.stats.updated += 1
        else:
            # New document - create
            doc = doc_repo.create(
                dataset_id=self._dataset_id,
                path=path,
                content_hash=content_hash,
                body=body,
                etag=etag,
                last_modified=last_modified,
                metadata_json=metadata_json,
            )
            session.flush()

            fts.upsert(doc.id, path, body)
            node.metadata["doc_id"] = doc.id

            # Add to existing_docs cache for any future nodes with same path
            existing_docs[path] = doc

            logger.debug(f"Created document: {path}")
            self.stats.created += 1


@dataclass
class ChunkPersistenceStats:
    """Statistics from a chunk persistence operation.

    Attributes:
        created: Number of new chunks created in FTS index.
        updated: Number of existing chunks updated.
        skipped: Number of unchanged chunks skipped (currently unused).
        failed: Number of chunks that failed to process.
        errors: List of error messages for failed chunks.
    """

    created: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        """Total number of chunks processed (excluding failures)."""
        return self.created + self.updated + self.skipped

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.created = 0
        self.updated = 0
        self.skipped = 0
        self.failed = 0
        self.errors = []


class ChunkPersistenceTransform(TransformComponent):
    """LlamaIndex TransformComponent that persists chunks to the FTS index.

    This transform handles chunk persistence after MarkdownNodeParser:
    1. Assigns stable node IDs in format `{content_hash}:{chunk_seq}`
    2. Sets metadata fields (source_doc_id, doc_id, chunk_seq, chunk_pos)
    3. Upserts chunks to FTSChunkManager (chunks_fts table)
    4. Tracks statistics (created, updated, skipped, failed)

    The transform uses the ambient session from the current context (set via
    `use_session()`). Statistics are available via the `stats` property.

    Example:
        with get_session() as session:
            with use_session(session):
                chunk_persist = ChunkPersistenceTransform(dataset_name="obsidian")
                pipeline = IngestionPipeline(
                    transformations=[
                        TextNormalizerTransform(),
                        PersistenceTransform(dataset_id=dataset_id),
                        MarkdownNodeParser(),
                        chunk_persist,
                    ]
                )
                pipeline.run(documents=documents)
                print(f"Created {chunk_persist.stats.created} chunks")

    Attributes:
        stats: ChunkPersistenceStats with counts of created/updated/skipped/failed.
    """

    # Private attributes (not Pydantic fields)
    _dataset_name: str = ""
    _path_key: str = "relative_path"
    _doc_id_key: str = "doc_id"
    _content_hash_key: str = "content_hash"
    _stats: ChunkPersistenceStats | None = None

    def __init__(
        self,
        dataset_name: str,
        *,
        path_key: str = "relative_path",
        doc_id_key: str = "doc_id",
        content_hash_key: str = "content_hash",
        **kwargs: Any,
    ) -> None:
        """Initialize the chunk persistence transform.

        Args:
            dataset_name: Name of the dataset (e.g., "obsidian").
            path_key: Metadata key for the document path (default: "relative_path").
            doc_id_key: Metadata key for the document ID (default: "doc_id").
            content_hash_key: Metadata key for content hash (default: "content_hash").
            **kwargs: Additional arguments passed to TransformComponent.
        """
        super().__init__(**kwargs)
        self._dataset_name = dataset_name
        self._path_key = path_key
        self._doc_id_key = doc_id_key
        self._content_hash_key = content_hash_key
        self._stats = ChunkPersistenceStats()

    @property
    def stats(self) -> ChunkPersistenceStats:
        """Get chunk persistence statistics."""
        if self._stats is None:
            self._stats = ChunkPersistenceStats()
        return self._stats

    def __call__(
        self,
        nodes: list[BaseNode],
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Persist each chunk to the FTS index.

        For each node:
        - Assigns stable node ID as `{content_hash}:{chunk_seq}`
        - Sets metadata fields (source_doc_id, doc_id, chunk_seq, chunk_pos)
        - Upserts to FTSChunkManager
        - Tracks statistics

        Requires an ambient session to be set via `use_session()`.

        Args:
            nodes: List of TextNodes to persist (typically from MarkdownNodeParser).
            **kwargs: Additional arguments (unused).

        Returns:
            The same nodes with assigned IDs and metadata (pass-through).

        Raises:
            SessionNotSetError: If no ambient session is set.
        """
        logger.info(f"ChunkPersistenceTransform: persisting {len(nodes)} chunks")

        # Reset stats for this run
        self.stats.reset()

        # Get FTS manager using ambient session
        fts_chunk = FTSChunkManager()

        # Group nodes by source document to assign chunk sequences
        doc_chunks: dict[str, list[BaseNode]] = {}
        for node in nodes:
            # Get source document identifier
            ref_doc_id = getattr(node, "ref_doc_id", None) or node.node_id
            source_key = self._get_source_key(node, ref_doc_id)
            if source_key not in doc_chunks:
                doc_chunks[source_key] = []
            doc_chunks[source_key].append(node)

        # Process nodes grouped by document
        for source_key, chunk_nodes in doc_chunks.items():
            for chunk_seq, node in enumerate(chunk_nodes):
                try:
                    self._process_chunk(
                        fts_chunk=fts_chunk,
                        node=node,
                        chunk_seq=chunk_seq,
                    )
                except Exception as e:
                    node_id = getattr(node, "node_id", "unknown")
                    logger.error(f"Failed to persist chunk {node_id}: {e}")
                    self.stats.failed += 1
                    self.stats.errors.append(f"{node_id}: {e}")

        logger.info(
            f"ChunkPersistenceTransform complete: "
            f"created={self.stats.created}, failed={self.stats.failed}"
        )

        return nodes

    def _get_source_key(self, node: BaseNode, ref_doc_id: str) -> str:
        """Get a key to group chunks by source document.

        Uses same path extraction logic as _get_path, falling back
        to ref_doc_id if no path metadata is found.

        Args:
            node: The chunk node.
            ref_doc_id: Reference document ID from the node.

        Returns:
            Grouping key (path or ref_doc_id).
        """
        path = self._get_path(node)
        return path if path else ref_doc_id

    def _get_content_hash(self, node: BaseNode) -> str:
        """Get content hash from node metadata or compute it.

        Args:
            node: The chunk node.

        Returns:
            Content hash string.
        """
        # First check metadata for existing hash
        if node.metadata and self._content_hash_key in node.metadata:
            return str(node.metadata[self._content_hash_key])

        # Compute from source document content if available
        # Fall back to computing from node's own text
        text = node.get_content()
        return _compute_content_hash(text)

    def _get_path(self, node: BaseNode) -> str:
        """Extract document path from node metadata.

        Tries multiple keys in order of preference:
        1. relative_path (default key)
        2. file_name (Obsidian reader)
        3. note_name + .md extension (Obsidian reader)

        Args:
            node: The chunk node.

        Returns:
            Document path string.
        """
        if not node.metadata:
            return ""

        # Try the configured path key first (default: relative_path)
        if self._path_key in node.metadata:
            return str(node.metadata[self._path_key])

        # Fallback to file_name (used by Obsidian reader)
        if "file_name" in node.metadata:
            return str(node.metadata["file_name"])

        # Fallback to note_name + .md extension
        if "note_name" in node.metadata:
            return f"{node.metadata['note_name']}.md"

        return ""

    def _get_doc_id(self, node: BaseNode) -> int | None:
        """Extract document ID from node metadata.

        Args:
            node: The chunk node.

        Returns:
            Document ID as integer, or None if not available.
        """
        if node.metadata and self._doc_id_key in node.metadata:
            doc_id = node.metadata[self._doc_id_key]
            if isinstance(doc_id, int):
                return doc_id
            try:
                return int(doc_id)
            except (ValueError, TypeError):
                pass
        return None

    def _process_chunk(
        self,
        fts_chunk: FTSChunkManager,
        node: BaseNode,
        chunk_seq: int,
    ) -> None:
        """Process a single chunk for persistence.

        Args:
            fts_chunk: FTSChunkManager instance.
            node: The chunk node.
            chunk_seq: 0-indexed chunk sequence within the document.
        """
        # Get content hash for stable ID
        content_hash = self._get_content_hash(node)
        path = self._get_path(node)
        doc_id = self._get_doc_id(node)

        # Build source_doc_id in format {dataset_name}:{path}
        source_doc_id = f"{self._dataset_name}:{path}"

        # Assign stable node ID: {content_hash}:{chunk_seq}
        node_id = f"{content_hash}:{chunk_seq}"
        node.id_ = node_id

        # Set metadata fields
        node.metadata["source_doc_id"] = source_doc_id
        node.metadata["chunk_seq"] = chunk_seq
        if doc_id is not None:
            node.metadata["doc_id"] = doc_id

        # Try to get character offset if available (from parser)
        if "start_char_idx" in node.metadata:
            node.metadata["chunk_pos"] = node.metadata["start_char_idx"]
        elif hasattr(node, "start_char_idx") and node.start_char_idx is not None:
            node.metadata["chunk_pos"] = node.start_char_idx

        # Get chunk text
        text = node.get_content()

        # Check if chunk already exists by searching for existing node_id
        # For FTS5, we use upsert which handles both insert and update
        # Since upsert does DELETE then INSERT, we count all as "updated"
        # unless we can detect it's truly new

        # Upsert to FTS
        fts_chunk.upsert(node_id=node_id, text=text, source_doc_id=source_doc_id)

        # Count as created (we can't easily distinguish update vs create with FTS5)
        # A more sophisticated implementation could query first to check existence
        self.stats.created += 1

        logger.debug(
            f"Persisted chunk {node_id} (seq={chunk_seq}) from {source_doc_id}"
        )
