"""Command implementations for PMD CLI."""

from .collection import handle_collection
from .index import (
    add_index_arguments,
    handle_cleanup,
    handle_embed,
    handle_index_collection,
    handle_update_all,
)
from .search import add_search_arguments, handle_query, handle_search, handle_vsearch
from .status import handle_status

__all__ = [
    "handle_collection",
    "handle_search",
    "handle_vsearch",
    "handle_query",
    "add_search_arguments",
    "handle_index_collection",
    "handle_update_all",
    "handle_embed",
    "handle_cleanup",
    "add_index_arguments",
    "handle_status",
]
