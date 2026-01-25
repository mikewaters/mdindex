#!/usr/bin/env python3
# /// script
# dependencies = ["idx"]
# ///
"""Search the idx database.

Executes FTS, vector, and hybrid searches and prints results.

Usage:
    uv run python scripts/search.py "your search query"
"""
import sys

from idx.search import search


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: search.py <query>", file=sys.stderr)
        return 1

    query = " ".join(sys.argv[1:])

    print(f"=== FTS Search: {query!r} ===\n")
    fts_results = search(query, mode="fts")
    _print_results(fts_results)

    print(f"\n=== Vector Search: {query!r} ===\n")
    vector_results = search(query, mode="vector")
    _print_results(vector_results)

    print(f"\n=== Hybrid Search: {query!r} ===\n")
    hybrid_results = search(query, mode="hybrid")
    _print_results(hybrid_results)

    return 0


def _print_results(results) -> None:
    if not results.results:
        print("  No results found.")
        return

    for i, r in enumerate(results.results, 1):
        print(f"{i}. [{r.dataset_name}] {r.path} (score: {r.score:.3f})")
        if r.chunk_text:
            snippet = r.chunk_text[:100].replace("\n", " ")
            print(f"   {snippet}...")

    if results.timing_ms:
        print(f"\n  ({len(results.results)} results in {results.timing_ms:.0f}ms)")


if __name__ == "__main__":
    sys.exit(main())
