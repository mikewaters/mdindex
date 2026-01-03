"""Command implementations for PMD CLI."""

from .collection import handle_collection
from .index import (
    add_backfill_arguments,
    add_index_arguments,
    handle_backfill_metadata,
    handle_cleanup,
    handle_embed,
    handle_index_collection,
    handle_update_all,
)
from .search import add_search_arguments, handle_query, handle_search, handle_vsearch
from .status import add_status_arguments, handle_status

__all__ = [
    "add_backfill_arguments",
    "add_index_arguments",
    "add_search_arguments",
    "add_status_arguments",
    "handle_backfill_metadata",
    "handle_cleanup",
    "handle_collection",
    "handle_embed",
    "handle_index_collection",
    "handle_query",
    "handle_search",
    "handle_status",
    "handle_update_all",
    "handle_vsearch",
]
