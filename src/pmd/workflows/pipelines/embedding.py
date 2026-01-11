"""Embedding pipeline for document chunking and vector generation.

This module implements the EmbeddingPipeline that orchestrates:
1. list_embed_targets - Find documents needing embeddings
2. embed_documents - Chunk, embed, and store vectors for each document

The pipeline uses EmbeddingGeneratorProtocol for the actual embedding work,
which handles chunking and storage internally. This keeps the pipeline focused
on orchestration, progress tracking, and error handling.

Design notes:
- Uses existing EmbeddingGenerator.embed_document() which handles chunk->embed->store
- Adding ChunkerProtocol integration can be done later if finer control is needed
- Progress callbacks are invoked per document
- Trace context is logged at key boundaries
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Awaitable

from loguru import logger

from pmd.core.exceptions import SourceCollectionNotFoundError

if TYPE_CHECKING:
    from pmd.app.types import (
        SourceCollectionRepositoryProtocol,
        DocumentRepositoryProtocol,
        EmbeddingGeneratorProtocol,
        EmbeddingRepositoryProtocol,
        DatabaseProtocol,
    )
    from pmd.workflows.contracts import EmbedRequest, EmbedTarget


class EmbeddingPipeline:
    """Pipeline for generating and storing document embeddings.

    This pipeline orchestrates the embedding flow:
    - Find documents needing embeddings (skip if already embedded)
    - For each document: chunk, generate embeddings, store vectors
    - Track progress and handle errors per document

    The actual embedding work is delegated to EmbeddingGeneratorProtocol,
    which handles chunking and storage internally. This design keeps the
    pipeline simple while allowing the generator to be swapped for different
    embedding strategies.

    Example:
        from pmd.workflows.contracts import EmbedRequest

        pipeline = EmbeddingPipeline(
            source_collection_repo=repo,
            embedding_generator_factory=get_generator,
            embedding_repo=embed_repo,
            db=db,
        )

        request = EmbedRequest(collection_name="my-docs", force=False)
        result = await pipeline.execute(request)
        print(f"Embedded: {result.embedded}, Chunks: {result.chunks_total}")
    """

    def __init__(
        self,
        source_collection_repo: "SourceCollectionRepositoryProtocol",
        embedding_generator_factory: Callable[[], Awaitable["EmbeddingGeneratorProtocol"]],
        embedding_repo: "EmbeddingRepositoryProtocol | None",
        db: "DatabaseProtocol",
        document_repo: "DocumentRepositoryProtocol | None" = None,
        llm_available_check: Callable[[], Awaitable[bool]] | None = None,
    ):
        """Initialize EmbeddingPipeline.

        Args:
            source_collection_repo: Repository for collection lookup.
            embedding_generator_factory: Async factory for creating embedding generator.
            embedding_repo: Repository for checking existing embeddings (optional).
            db: Database for querying documents.
            document_repo: Repository for document queries.
            llm_available_check: Optional async function to check LLM availability.
        """
        self._source_collection_repo = source_collection_repo
        self._embedding_generator_factory = embedding_generator_factory
        self._embedding_repo = embedding_repo
        self._document_repo = document_repo
        self._db = db
        self._llm_available_check = llm_available_check

    async def execute(self, request: "EmbedRequest") -> "EmbedResult":
        """Execute the embedding pipeline.

        Pipeline flow:
        1. Verify prerequisites (vector storage, LLM availability)
        2. Query documents needing embeddings
        3. For each document: chunk, embed, store
        4. Track progress and accumulate results

        Args:
            request: Embedding request with collection name and options.

        Returns:
            EmbedResult with counts of embedded documents and chunks.

        Raises:
            SourceCollectionNotFoundError: If collection does not exist.
            RuntimeError: If vector storage or embedding generator not available.
        """
        from pmd.services.indexing import EmbedResult

        collection_name = request.collection_name
        force = request.force
        context = request.context
        progress_callback = request.progress_callback

        # Log with trace context if available
        trace_info = f", trace_id={context.trace_id[:8]}" if context else ""
        logger.info(
            f"EmbeddingPipeline.execute: collection={collection_name!r}, "
            f"force={force}{trace_info}"
        )
        start_time = time.perf_counter()

        # Check prerequisites
        if not self._db.vec_available:
            raise RuntimeError(
                "Vector storage not available (sqlite-vec extension not loaded)"
            )

        if self._llm_available_check and not await self._llm_available_check():
            raise RuntimeError("LLM provider not available (is it running?)")

        # Verify collection exists
        source_collection = self._source_collection_repo.get_by_name(collection_name)
        if not source_collection:
            raise SourceCollectionNotFoundError(
                f"Collection '{collection_name}' not found"
            )

        # Get embedding generator
        embedding_generator = await self._embedding_generator_factory()

        # Query documents needing embeddings
        embed_targets = self._list_embed_targets(source_collection.id, force)
        total_docs = len(embed_targets)

        logger.debug(f"Found {total_docs} documents to process for embedding")

        # Process each document
        embedded_count = 0
        skipped_count = 0
        chunks_total = 0

        for idx, target in enumerate(embed_targets):
            # Check if already embedded (unless force)
            if not force and self._embedding_repo and self._embedding_repo.has_embeddings(target.doc_hash):
                skipped_count += 1
                continue

            try:
                # Generate and store embeddings
                # The generator handles chunking, embedding, and storage internally
                chunks_embedded = await embedding_generator.embed_document(
                    target.doc_hash,
                    target.content,
                    force=force,
                )

                if chunks_embedded > 0:
                    embedded_count += 1
                    chunks_total += chunks_embedded
                    logger.debug(f"Embedded: {target.path} ({chunks_embedded} chunks)")

                # Report progress
                if progress_callback:
                    progress_callback(idx + 1, total_docs, target.path)

            except Exception as e:
                # Log error but continue with other documents
                logger.warning(f"Failed to embed document: {target.path}: {e}")
                # Could add to error list if we want to track failures

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"EmbeddingPipeline complete: collection={collection_name!r}, "
            f"embedded={embedded_count}, skipped={skipped_count}, "
            f"chunks={chunks_total}, {elapsed:.1f}ms"
        )

        return EmbedResult(
            embedded=embedded_count,
            skipped=skipped_count,
            chunks_total=chunks_total,
        )

    def _list_embed_targets(
        self,
        source_collection_id: int,
        force: bool,
    ) -> list["EmbedTarget"]:
        """Query documents needing embeddings.

        Args:
            source_collection_id: Collection to query.
            force: If True, include all documents (not just those missing embeddings).

        Returns:
            List of EmbedTarget with doc info.
        """
        from pmd.workflows.contracts import EmbedTarget

        # Query all active documents in collection with their content
        if self._document_repo:
            rows = self._document_repo.list_active_with_content(source_collection_id)
        else:
            # Fallback to direct SQL if document_repo not available
            cursor = self._db.execute(
                """
                SELECT d.path, d.hash, c.doc
                FROM documents d
                JOIN content c ON d.hash = c.hash
                WHERE d.source_collection_id = ? AND d.active = 1
                """,
                (source_collection_id,),
            )
            rows = [(row["path"], row["hash"], row["doc"]) for row in cursor.fetchall()]

        return [
            EmbedTarget(
                doc_hash=hash_val,
                path=path,
                content=content,
            )
            for path, hash_val, content in rows
        ]
