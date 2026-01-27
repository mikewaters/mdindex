#!/usr/bin/env python3
# /// script
# dependencies = ["idx"]
# ///
"""Search the idx database.

Executes FTS, vector, and hybrid searches and prints results.

Usage:
    uv run python scripts/search.py "your search query"
    uv run python scripts/search.py -d obsidian "your search query"
"""
import argparse
import sys

from idx.search import search


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the idx database")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument(
        "-d", "--dataset",
        dest="dataset_name",
        help="Filter results to a specific dataset",
    )
    args = parser.parse_args()

    query = " ".join(args.query)
    dataset_name = args.dataset_name

    dataset_info = f" (dataset: {dataset_name})" if dataset_name else ""

    print(f"=== FTS Search: {query!r}{dataset_info} ===\n")
    fts_results = search(query, mode="fts", dataset_name=dataset_name)
    _print_results(fts_results)

    print(f"\n=== Vector Search: {query!r}{dataset_info} ===\n")
    vector_results = search(query, mode="vector", dataset_name=dataset_name)
    _print_results(vector_results)

    print(f"\n=== Hybrid Search: {query!r}{dataset_info} ===\n")
    hybrid_results = search(query, mode="hybrid", dataset_name=dataset_name)
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
