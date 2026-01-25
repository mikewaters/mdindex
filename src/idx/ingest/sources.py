from enum import Enum
from functools import singledispatch

from idx.ingest.schemas import IngestDirectoryConfig, IngestObsidianConfig
from idx.ingest.directory import DirectorySource
from idx.ingest.obsidian import ObsidianVaultSource

__all__ = ["create_source", "SourceType"]

@singledispatch
def create_source(config):
    raise TypeError(f"Unsupported config type: {type(config)}")

@create_source.register
def _(config: IngestObsidianConfig):
    return ObsidianVaultSource(config.source_path)

@create_source.register
def _(config: IngestDirectoryConfig):
    return DirectorySource(
        config.source_path,
        patterns=config.patterns,
        encoding=config.encoding,
    )


class SourceType(str, Enum):
    obsidian = "obsidian"
    directory = "directory"


