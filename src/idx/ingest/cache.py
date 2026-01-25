"""Pipeline caching for ingestion.

Provides persistence and loading of IngestionPipeline state including
docstore (for document hash tracking) and vector store (for embeddings).
"""

from pathlib import Path
from typing import TYPE_CHECKING

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.vector_stores import SimpleVectorStore

from idx.core.settings import get_settings
from idx.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

PIPELINE_DIR = "pipeline_storage"


def _get_cache_path(dataset_name: str) -> Path:
    """Get the cache path for a dataset."""
    settings = get_settings()
    return settings.cache_path / PIPELINE_DIR / dataset_name


def _ensure_directories(dataset_name: str) -> None:
    """Ensure that required cache directories exist."""
    cache_path = _get_cache_path(dataset_name)
    cache_path.mkdir(parents=True, exist_ok=True)


def load_pipeline(dataset_name: str, pipeline: IngestionPipeline) -> IngestionPipeline:
    """Load a persisted pipeline's docstore state.

    Restores the pipeline's internal docstore which tracks document hashes
    for deduplication. The vector_store should be loaded separately and
    passed to the pipeline constructor.

    Args:
        dataset_name: Name of the dataset.
        pipeline: Pipeline instance to load state into.

    Returns:
        The pipeline with restored docstore state.
    """
    cache_path = _get_cache_path(dataset_name)
    if cache_path.exists():
        logger.info(f"Loading persisted pipeline from {cache_path}")
        pipeline.load(persist_dir=str(cache_path))

    return pipeline


def load_vector_store(dataset_name: str) -> SimpleVectorStore | None:
    """Load a persisted vector store for a dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        SimpleVectorStore if found, None otherwise.
    """
    cache_path = _get_cache_path(dataset_name)
    vector_store_path = cache_path / "vector_store.json"

    if vector_store_path.exists():
        logger.info(f"Loading vector store from {vector_store_path}")
        return SimpleVectorStore.from_persist_path(str(vector_store_path))

    return None


def persist_pipeline(dataset_name: str, pipeline: IngestionPipeline) -> None:
    """Persist pipeline's docstore state to disk.

    Saves the pipeline's internal docstore which tracks document hashes.
    The vector_store should be persisted separately via persist_vector_store.

    Args:
        dataset_name: Name of the dataset.
        pipeline: Pipeline to persist.
    """
    _ensure_directories(dataset_name)
    cache_path = _get_cache_path(dataset_name)

    pipeline.persist(persist_dir=str(cache_path))
    logger.info(f"Pipeline persisted to {cache_path}")


def persist_vector_store(dataset_name: str, vector_store: SimpleVectorStore) -> None:
    """Persist a vector store for a dataset.

    Args:
        dataset_name: Name of the dataset.
        vector_store: Vector store to persist.
    """
    _ensure_directories(dataset_name)
    cache_path = _get_cache_path(dataset_name)
    vector_store_path = cache_path / "vector_store.json"

    vector_store.persist(str(vector_store_path))
    logger.info(f"Vector store persisted to {vector_store_path}")


def clear_cache(dataset_name: str) -> None:
    """Clear all cached data for a dataset.

    Removes pipeline docstore and vector store. Used by force=True
    to ensure clean re-indexing.

    Args:
        dataset_name: Name of the dataset to clear.
    """
    import shutil

    cache_path = _get_cache_path(dataset_name)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        logger.info(f"Cleared cache for dataset '{dataset_name}'")