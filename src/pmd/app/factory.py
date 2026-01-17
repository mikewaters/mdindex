"""Application composition root and dependency injection.

This module provides the main factory function for creating configured
Application instances with all dependencies wired together.

Example:
    from pmd.app import create_application
    from pmd.core.config import Config

    async with create_application(Config()) as app:
        results = await app.search.hybrid_search("query")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import Config

from .application import Application


async def create_application(config: "Config") -> Application:
    """Create and wire a fully configured Application.

    This is the main composition root that wires all dependencies together.

    Args:
        config: Application configuration.

    Returns:
        Configured Application instance ready for use.

    Example:
        from pmd.core.config import Config
        from pmd.app import create_application

        async with create_application(Config()) as app:
            await app.indexing.index_collection("docs", source)
            results = await app.search.hybrid_search("query")
    """
    # Lazy imports to avoid circular dependencies
    from pmd.store.database import Database
    from pmd.store.repositories.collections import SourceCollectionRepository
    from pmd.store.repositories.documents import DocumentRepository
    from pmd.store.repositories.content import ContentRepository
    from pmd.store.repositories.fts import FTS5SearchRepository
    from pmd.store.repositories.embeddings import EmbeddingRepository
    from pmd.store.repositories.source_metadata import SourceMetadataRepository
    from pmd.services.indexing import IndexingService
    from pmd.services.loading import LoadingService
    from pmd.services.search import SearchService
    from pmd.services.status import StatusService
    from pmd.sources import get_default_registry
    from pmd.llm import create_llm_provider, EmbeddingGenerator, QueryExpander, DocumentReranker
    from pmd.metadata import (
        LexicalTagMatcher,
        Ontology,
        TagRetriever,
    )
    from pmd.store.repositories.metadata import DocumentMetadataRepository
    from pmd.services.caching import DocumentCacher

    # Create and connect database
    db = Database(config.db_path)
    db.connect()

    # Create repositories
    source_collection_repo = SourceCollectionRepository(db)
    document_repo = DocumentRepository(db)
    content_repo = ContentRepository(db)
    fts_repo = FTS5SearchRepository(db)
    embedding_repo = EmbeddingRepository(db)

    # Create LLM provider (may be None if provider unavailable)
    try:
        llm_provider = create_llm_provider(config)
    except (ValueError, RuntimeError):
        llm_provider = None

    # Create async factories for LLM components
    async def get_embedding_generator():
        if llm_provider:
            return EmbeddingGenerator(llm_provider, embedding_repo, config)
        return None

    async def get_query_expander():
        if llm_provider:
            return QueryExpander(llm_provider)
        return None

    async def get_reranker():
        if llm_provider:
            return DocumentReranker(llm_provider)
        return None

    # Create sync factories for metadata components
    def get_tag_matcher():
        return LexicalTagMatcher()

    def get_ontology():
        return Ontology()  # type: ignore

    def get_tag_retriever():
        return TagRetriever(db)  # type: ignore

    def get_metadata_repo():
        return DocumentMetadataRepository(db)

    # Check LLM availability
    async def is_llm_available():
        if llm_provider:
            return await llm_provider.is_available()
        return False

    # Create source metadata repository
    source_metadata_repo = SourceMetadataRepository(db)

    # Create source registry
    source_registry = get_default_registry()

    # Create document cacher (if enabled in config)
    cacher = DocumentCacher(config.cache) if config.cache.enabled else None

    # Create loading service
    loading = LoadingService(
        db=db,
        source_collection_repo=source_collection_repo,
        document_repo=document_repo,
        source_metadata_repo=source_metadata_repo,
        source_registry=source_registry,
    )

    # Create services with explicit dependencies
    indexing = IndexingService(
        db=db,
        source_collection_repo=source_collection_repo,
        document_repo=document_repo,
        fts_repo=fts_repo,
        loader=loading,
        content_repo=content_repo,
        embedding_repo=embedding_repo,
        embedding_generator_factory=get_embedding_generator,  # type: ignore
        llm_available_check=is_llm_available,
        source_registry=source_registry,
        cacher=cacher,
    )

    search = SearchService(
        db=db,
        fts_repo=fts_repo,
        source_collection_repo=source_collection_repo,
        embedding_repo=embedding_repo,
        embedding_generator_factory=get_embedding_generator,  # type: ignore
        query_expander_factory=get_query_expander,  # type: ignore
        reranker_factory=get_reranker,  # type: ignore
        tag_matcher_factory=get_tag_matcher,  # type: ignore
        ontology_factory=get_ontology,  # type: ignore
        tag_retriever_factory=get_tag_retriever,  # type: ignore
        metadata_repo_factory=get_metadata_repo,  # type: ignore
        fts_weight=config.search.fts_weight,
        vec_weight=config.search.vec_weight,
        rrf_k=config.search.rrf_k,
        rerank_candidates=config.search.rerank_candidates,
    )

    status = StatusService(
        document_repo=document_repo,
        embedding_repo=embedding_repo,
        fts_repo=fts_repo,
        source_collection_repo=source_collection_repo,
        db_path=config.db_path,
        llm_provider=config.llm_provider,
        llm_available_check=is_llm_available,
        vec_available=db.vec_available,
    )

    return Application(
        db=db,
        llm_provider=llm_provider,
        loading=loading,
        indexing=indexing,
        search=search,
        status=status,
        config=config,
        source_collection_repo=source_collection_repo,
        document_repo=document_repo,
        embedding_repo=embedding_repo,
    )
