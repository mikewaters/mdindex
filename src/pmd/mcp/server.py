"""MCP server for PMD."""

from ..core.config import Config
from ..core.exceptions import CollectionNotFoundError
from ..services import ServiceContainer


class PMDMCPServer:
    """MCP server exposing PMD search functionality.

    Uses ServiceContainer for all operations, providing a clean interface
    for MCP clients to search documents and retrieve content.

    Example:

        server = PMDMCPServer(config)
        await server.initialize()
        try:
            results = await server.search("machine learning", limit=5)
            status = await server.get_status()
        finally:
            await server.shutdown()
    """

    def __init__(self, config: Config):
        """Initialize MCP server.

        Args:
            config: Application configuration.
        """
        self.config = config
        self._services: ServiceContainer | None = None

    async def initialize(self) -> None:
        """Initialize the server (connect to database and services)."""
        self._services = ServiceContainer(self.config)
        self._services.connect()

    async def shutdown(self) -> None:
        """Shutdown the server (close connections)."""
        if self._services:
            await self._services.close()
            self._services = None

    @property
    def services(self) -> ServiceContainer:
        """Get the service container.

        Raises:
            RuntimeError: If server not initialized.
        """
        if self._services is None:
            raise RuntimeError("Server not initialized. Call initialize() first.")
        return self._services

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
        # Check if LLM is available for enhanced search
        llm_available = await self.services.is_llm_available()

        results = await self.services.search.hybrid_search(
            query,
            limit=limit,
            collection_name=collection,
            enable_query_expansion=llm_available,
            enable_reranking=llm_available,
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
        coll = self.services.collection_repo.get_by_name(collection)

        if not coll:
            return {"error": f"Collection '{collection}' not found"}

        doc = self.services.document_repo.get(coll.id, path)

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
        collections = self.services.collection_repo.list_all()

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
        return await self.services.status.get_full_status()

    async def index_collection(
        self,
        collection: str,
        force: bool = False,
        embed: bool = False,
    ) -> dict:
        """Index a collection.

        Args:
            collection: Collection name.
            force: Force reindex all documents.

        Returns:
            Indexing result.
        """
        try:
            c = self.services.collection_repo.get_by_name(collection)
            if not c:
                raise CollectionNotFoundError(f"Collection '{collection}' not found")

            from ..sources import get_default_registry

            registry = get_default_registry()
            source = registry.create_source(c)

            result = await self.services.indexing.index_collection(
                collection,
                force=force,
                embed=embed,
                source=source,
            )
            return {
                "success": True,
                "collection": collection,
                "indexed": result.indexed,
                "skipped": result.skipped,
                "errors": len(result.errors),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def embed_collection(
        self,
        collection: str,
        force: bool = False,
    ) -> dict:
        """Generate embeddings for a collection.

        Args:
            collection: Collection name.
            force: Force regenerate all embeddings.

        Returns:
            Embedding result.
        """
        try:
            result = await self.services.indexing.embed_collection(collection, force)
            return {
                "success": True,
                "collection": collection,
                "embedded": result.embedded,
                "skipped": result.skipped,
                "chunks": result.chunks_total,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
