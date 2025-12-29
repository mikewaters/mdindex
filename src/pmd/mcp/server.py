"""MCP server for PMD."""

import asyncio
import json

from ..core.config import Config
from ..llm import QueryExpander, DocumentReranker, create_llm_provider
from ..llm.embeddings import EmbeddingGenerator
from ..search.pipeline import HybridSearchPipeline, SearchPipelineConfig
from ..store.database import Database
from ..store.documents import DocumentRepository
from ..store.embeddings import EmbeddingRepository
from ..store.search import FTS5SearchRepository


class PMDMCPServer:
    """MCP server exposing PMD search functionality."""

    def __init__(self, config: Config):
        """Initialize MCP server.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.db = Database(config.db_path)
        self.embedding_repo = EmbeddingRepository(self.db)
        self.search_repo = FTS5SearchRepository(self.db)
        self.doc_repo = DocumentRepository(self.db)

        # Initialize LLM provider
        self.llm_provider = create_llm_provider(config)

        # Initialize LLM components
        self.embedding_generator = EmbeddingGenerator(
            self.llm_provider,
            self.embedding_repo,
            config,
        )
        self.query_expander = QueryExpander(self.llm_provider)
        self.reranker = DocumentReranker(self.llm_provider)

        # Initialize search pipeline with LLM components
        pipeline_config = SearchPipelineConfig(
            fts_weight=config.search.fts_weight,
            vec_weight=config.search.vec_weight,
            rrf_k=config.search.rrf_k,
            rerank_candidates=config.search.rerank_candidates,
            enable_query_expansion=True,
            enable_reranking=True,
        )
        self.pipeline = HybridSearchPipeline(
            self.search_repo,
            pipeline_config,
            query_expander=self.query_expander,
            reranker=self.reranker,
            embedding_generator=self.embedding_generator,
        )

    async def initialize(self) -> None:
        """Initialize the server (connect to database)."""
        self.db.connect()

    async def shutdown(self) -> None:
        """Shutdown the server (close connections)."""
        await self.llm_provider.close()
        self.db.close()

    async def search(
        self,
        query: str,
        limit: int = 5,
        collection: str | None = None,
    ) -> dict:
        """Execute hybrid search.

        Args:
            query: Search query.
            limit: Maximum results to return.
            collection: Optional collection name filter.

        Returns:
            Search results as dictionary.
        """
        # Get collection ID if specified
        collection_id = None
        if collection:
            from ..store.collections import CollectionRepository

            repo = CollectionRepository(self.db)
            coll = repo.get_by_name(collection)
            if coll:
                collection_id = coll.id

        # Execute search
        results = await self.pipeline.search(
            query,
            limit=limit,
            collection_id=collection_id,
        )

        return {
            "query": query,
            "collection": collection,
            "results_count": len(results),
            "results": [
                {
                    "file": r.file,
                    "title": r.title,
                    "score": r.score,
                    "fts_score": r.fts_score,
                    "vec_score": r.vec_score,
                    "rerank_score": r.rerank_score,
                }
                for r in results
            ],
        }

    async def get_document(
        self,
        collection: str,
        path: str,
    ) -> dict:
        """Retrieve a document by collection and path.

        Args:
            collection: Collection name.
            path: Document path relative to collection.

        Returns:
            Document content and metadata.
        """
        from ..store.collections import CollectionRepository

        repo = CollectionRepository(self.db)
        coll = repo.get_by_name(collection)

        if not coll:
            return {"error": f"Collection '{collection}' not found"}

        doc = self.doc_repo.get(coll.id, path)

        if not doc:
            return {"error": f"Document '{path}' not found in collection '{collection}'"}

        return {
            "file": doc.filepath,
            "title": doc.title,
            "collection": collection,
            "content_length": doc.body_length,
            "content": doc.body,
        }

    async def list_collections(self) -> dict:
        """List all collections.

        Returns:
            List of collections.
        """
        from ..store.collections import CollectionRepository

        repo = CollectionRepository(self.db)
        collections = repo.list_all()

        return {
            "collections_count": len(collections),
            "collections": [
                {
                    "name": c.name,
                    "path": c.pwd,
                    "glob_pattern": c.glob_pattern,
                }
                for c in collections
            ],
        }

    async def get_status(self) -> dict:
        """Get server and index status.

        Returns:
            Status information.
        """
        llm_available = await self.llm_provider.is_available()

        cursor = self.db.execute("SELECT COUNT(*) as count FROM documents WHERE active = 1")
        doc_count = cursor.fetchone()["count"]

        cursor = self.db.execute("SELECT COUNT(*) as count FROM content_vectors")
        embedding_count = cursor.fetchone()["count"]

        return {
            "status": "ready",
            "database_path": str(self.config.db_path),
            "llm_provider": self.config.llm_provider,
            "llm_available": llm_available,
            "documents_indexed": doc_count,
            "embeddings_stored": embedding_count,
        }
