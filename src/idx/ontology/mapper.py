"""
Understands how to map source-specific document metadata to ontology-aware constructs.
"""
from typing import Callable, Dict
from llama_index.core.schema import Document

from idx.ingest.sources import SourceType

from collections.abc import Mapping

def deep_collect(dicts):
    result = {}

    for d in dicts:
        for key, value in d.items():

            # If key not seen yet, initialize
            if key not in result:
                result[key] = value
                continue

            # If both values are dictionaries → recurse
            if isinstance(result[key], Mapping) and isinstance(value, Mapping):
                result[key] = deep_collect([result[key], value])

            # Otherwise → collect into list
            else:
                if not isinstance(result[key], list):
                    result[key] = [result[key]]
                result[key].append(value)

    return result


class ObsidianMetadataMapper:
    """Metadata translator for my Obsidian vaults
    Requirements:
    - "frontmatter" key is superseded by keys added to normal metadata dict,
    in order to support embedding of frontmatter in vector store. The `frontmatter`
    key should be ignored by LLMs and by embeddings.
    - This needs to happen before the persist occurs, so we can write the correct
    structure to the db
    """
    def __init__(self):
        pass

    def __call__(self, metadata: Dict) -> Dict:
        mapped = {}

        mapped = metadata
        return mapped

def get_metadata_mapper(source_type: str) -> Callable:
    match source_type:
        case "obsidian": return ObsidianMetadataMapper

    raise ValueError(f"Unsupported source type: {source_type}")


class OntologyManager:
    def __init__(self, source_type: str, frontmatter_key: str = "frontmatter"):
        self._mapper = get_metadata_mapper(source_type)
        self._key = frontmatter_key

    def map(self, document: Document) -> Dict:
        if self._key is not None:
            mapped = self._mapper(document.metadata[self._key])
        else:
            mapped = self._mapper(document.metadata)
        return mapped