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
    from ..search.adapters import (
        FTS5TextSearcher,
        EmbeddingVectorSearcher,
        TagRetrieverAdapter,
        LexicalTagInferencer,
        LLMQueryExpanderAdapter,
        LLMRerankerAdapter,
        OntologyMetadataBooster,
    )

from .application import Application


def _create_search_adapters(
    llm_provider,
    fts_repo,
    embedding_repo,
    db,
    config,
) -> dict:
    """Create search pipeline adapters.

    Returns a dict with adapter instances and related objects that can be
    passed to SearchService. Creates adapters once for reuse across searches.

    Args:
        llm_provider: LLM provider instance.
        fts_repo: FTS repository.
        embedding_repo: Embedding repository.
        db: Database instance.
        config: Application configuration.

    Returns:
        Dict with adapter instances and factories.
    """
    from pmd.search.adapters import (
        FTS5TextSearcher,
        EmbeddingVectorSearcher,
        LLMQueryExpanderAdapter,
        LLMRerankerAdapter,
        LexicalTagInferencer,
        TagRetrieverAdapter,
        OntologyMetadataBooster,
    )
    from pmd.llm import EmbeddingGenerator, QueryExpander, DocumentReranker
    from pmd.ontology.model import load_default_ontology
    from pmd.ontology.inference import LexicalTagMatcher
    from pmd.ontology.retrieval import TagRetriever
    from pmd.store.repositories.metadata import DocumentMetadataRepository

    # Text searcher
    text_searcher = FTS5TextSearcher(fts_repo)

    # LLM-dependent components
    embedding_generator = EmbeddingGenerator(llm_provider, embedding_repo, config)
    vector_searcher = EmbeddingVectorSearcher(embedding_generator)

    llm_query_expander = QueryExpander(llm_provider)
    query_expander = LLMQueryExpanderAdapter(llm_query_expander)

    llm_reranker = DocumentReranker(llm_provider)
    reranker = LLMRerankerAdapter(llm_reranker)

    # Tag-related components
    tag_matcher = LexicalTagMatcher()
    ontology = load_default_ontology()
    tag_inferencer = LexicalTagInferencer(tag_matcher, ontology)

    metadata_repo = DocumentMetadataRepository(db)
    tag_retriever = TagRetriever(db, metadata_repo)
    tag_searcher = TagRetrieverAdapter(tag_retriever)

    metadata_booster = OntologyMetadataBooster(db, metadata_repo, ontology)

    return {
        "text_searcher": text_searcher,
        "vector_searcher": vector_searcher,
        "query_expander": query_expander,
        "reranker": reranker,
        "tag_inferencer": tag_inferencer,
        "tag_searcher": tag_searcher,
        "metadata_booster": metadata_booster,
        "embedding_generator": embedding_generator,
    }


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
    from pmd.store.repositories.embeddings import EmbeddingRepository
    from pmd.store.repositories.fts import FTS5SearchRepository
    from pmd.store import IndexFacade, LoadFacade, SearchFacade, StatusFacade
    from pmd.services.indexing import IndexingService
    from pmd.services.loading import LoadingService
    from pmd.search.service import SearchService
    from pmd.services.status import StatusService
    from pmd.llm import create_llm_provider, EmbeddingGenerator
    from pmd.store.caching import DocumentCacher

    # Create and connect database
    db = Database(config.db_path)
    db.connect()

    # Create facade instances
    index_facade = IndexFacade(db)
    load_facade = LoadFacade(db)
    search_facade = SearchFacade(db)
    status_facade = StatusFacade(db)

    # Create repositories needed for search adapters
    embedding_repo = EmbeddingRepository(db)
    fts_repo = FTS5SearchRepository(db)

    # Create LLM provider (required - will raise if unavailable)
    llm_provider = create_llm_provider(config)

    # Create search adapters (pre-assembled for SearchService)
    search_adapters = _create_search_adapters(
        llm_provider=llm_provider,
        fts_repo=fts_repo,
        embedding_repo=embedding_repo,
        db=db,
        config=config,
    )

    # Create embedding generator factory for IndexingService
    # (IndexingService still uses factories for now)
    async def get_embedding_generator():
        return EmbeddingGenerator(llm_provider, embedding_repo, config)

    # Create document cacher (if enabled in config)
    cacher = DocumentCacher(config.cache) if config.cache.enabled else None

    # Create loading service
    loading = LoadingService(facade=load_facade)

    # Create services
    indexing = IndexingService(
        facade=index_facade,
        loader=loading,
        embedding_generator_factory=get_embedding_generator,  # type: ignore
        cacher=cacher,
    )

    # Create search service with pre-assembled adapters
    search = SearchService(
        facade=search_facade,
        # Pre-created adapters (LLM-dependent)
        text_searcher=search_adapters["text_searcher"],
        vector_searcher=search_adapters["vector_searcher"],
        query_expander=search_adapters["query_expander"],
        reranker=search_adapters["reranker"],
        embedding_generator=search_adapters["embedding_generator"],
        # Optional adapters (not LLM-dependent)
        tag_inferencer=search_adapters["tag_inferencer"],
        tag_searcher=search_adapters["tag_searcher"],
        metadata_booster=search_adapters["metadata_booster"],
        # Config
        fts_weight=config.search.fts_weight,
        vec_weight=config.search.vec_weight,
        rrf_k=config.search.rrf_k,
        rerank_candidates=config.search.rerank_candidates,
    )

    status = StatusService(
        facade=status_facade,
        db_path=config.db_path,
        llm_provider_name=config.llm_provider,
        llm_provider_instance=llm_provider,
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
    )
