# MCP Module Architecture

**Location:** `src/pmd/mcp/`

Model Context Protocol server for LLM integration.

## Files and Key Abstractions

### `server.py`

**`PMDMCPServer`** - MCP protocol implementation

Lifecycle:
- `initialize()` - Connect to database
- `shutdown()` - Release resources

API methods (all async, return dicts):
- `search(query, limit, collection)` - Hybrid search
- `get_document(collection, path)` - Retrieve document
- `list_collections()` - Enumerate collections
- `get_status()` - Index status
- `index_collection(name, force, embed)` - Index collection
- `embed_collection(name, force)` - Generate embeddings

**Invariants:**
- All methods return structured dicts
- LLM features auto-enabled if provider available
