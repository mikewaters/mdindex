# CLI Module Architecture

**Location:** `src/pmd/cli/`

Command-line interface using argparse with hierarchical commands.

## Files and Key Abstractions

### `main.py`

**`create_parser()`** - Build argparse parser with subcommands  
**`main()`** - Entry point with logging, tracing, command routing

Global arguments:
- `-v/--version` - Version display
- `-L/--log-level` - Logging level
- `-c/--config` - Config file path
- `--phoenix-tracing` - Enable tracing

### `commands/__init__.py`

Command handler exports:
- `handle_collection`, `handle_search`, `handle_vsearch`
- `handle_query`, `handle_index_collection`, `handle_embed`
- `handle_update_all`, `handle_cleanup`, `handle_status`

### `commands/collection.py`

Collection management sub-commands:
- `collection add` - Create from filesystem, HTTP, or entity
- `collection list` - List all collections
- `collection show` - Show collection details
- `collection remove` - Delete collection
- `collection rename` - Rename collection

### `commands/search.py`

Three search variants:
- `search` - FTS5 BM25 search
- `vsearch` - Vector semantic search
- `query` - Hybrid search with LLM enhancement

### `commands/index.py`

Indexing commands:
- `index <collection>` - Index documents
- `embed <collection>` - Generate embeddings
- `update-all` - Update all collections
- `cleanup` - Remove orphaned data

### `commands/status.py`

Status reporting:
- `status` - Index status summary
- `status --check-sync` - FTS/vector sync report
