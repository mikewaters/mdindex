
from llama_index.core.ingestion import IngestionPipeline

from idx.core.settings import get_settings
from idx.core.logging import get_logger

logger = get_logger(__name__)

PIPELINE_DIR = "pipeline_storage"       # pipeline cache (+ docstore if you persist pipeline)

def _ensure_directories(dataset_name: str) -> None:
    """Ensure that required cache directories exist."""
    settings = get_settings()
    (settings.cache_path / PIPELINE_DIR).mkdir(parents=True, exist_ok=True)

def load_pipeline(dataset_name: str, pipeline: IngestionPipeline) -> IngestionPipeline:
    """Retrieve a given dataset's persisted ingestion pipeline, if its been cached"""
    settings = get_settings()
    cache_path = settings.cache_path / PIPELINE_DIR / dataset_name
    if cache_path.exists():
        logger.info(f"Loading persisted pipeline from {cache_path}")
        pipeline.load(persist_dir=str(cache_path)) 
    
    return pipeline

def persist_pipeline(dataset_name: str, pipeline: IngestionPipeline) -> None:
    """Persist a given dataset's ingestion pipeline to disk for caching."""
    _ensure_directories(dataset_name)
    settings = get_settings()
    cache_path = settings.cache_path / PIPELINE_DIR / dataset_name

    pipeline.persist(persist_dir=str(cache_path)) 