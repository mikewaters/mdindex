"""MCP server for PMD."""

from ..core.config import Config
from ..core.exceptions import SourceCollectionNotFoundError
from ..app import Application, create_application


class PMDMCPServer:
    """MCP server exposing PMD search functionality.

    Uses Application for all operations, providing a clean interface
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
        self._app: Application | None = None

    async def initialize(self) -> None:
        """Initialize the server (connect to database and services)."""
        self._app = await create_application(self.config)

    async def shutdown(self) -> None:
        """Shutdown the server (close connections)."""
        if self._app:
            await self._app.close()
            self._app = None

    @property
    def app(self) -> Application:
        """Get the application instance.

        Raises:
            RuntimeError: If server not initialized.
        """
        if self._app is None:
            raise RuntimeError("Server not initialized. Call initialize() first.")
        return self._app

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
        llm_available = await self.app.is_llm_available()

        results = await self.app.search.hybrid_search(
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
        source_collection = self.app.source_collection_repo.get_by_name(collection)

        if not source_collection:
            return {"error": f"Source collection '{collection}' not found"}

        doc = self.app.document_repo.get(source_collection.id, path)

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
        source_collections = self.app.source_collection_repo.list_all()

        return {
            "source_collections_count": len(source_collections),
            "source_collections": [
                {
                    "name": c.name,
                    "path": c.pwd,
                    "glob_pattern": c.glob_pattern,
                }
                for c in source_collections
            ],
        }

    async def get_status(self) -> dict:
        """Get server and index status.

        Returns:
            Status information.
        """
        return await self.app.status.get_full_status()

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
            embed: Generate embeddings after indexing.

        Returns:
            Indexing result.
        """
        try:
            result = await self.app.indexing.index_collection(
                collection,
                force=force,
                embed=embed,
            )
            return {
                "success": True,
                "collection": collection,
                "indexed": result.indexed,
                "skipped": result.skipped,
                "errors": len(result.errors),
            }
        except SourceCollectionNotFoundError as e:
            return {
                "success": False,
                "error": str(e),
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
            result = await self.app.indexing.embed_collection(collection, force)
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
