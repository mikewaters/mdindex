"""idx.core.status - Minimal health/status checks.

Reports on system health: database connectivity, vector store path,
LLM availability, etc.

Example usage:
    from idx.core.status import check_health, HealthStatus

    status = check_health()
    if status.is_healthy:
        print("All systems operational")
    else:
        print(f"Issues: {status.issues}")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from idx.core.logging import get_logger
from idx.core.settings import get_settings

__all__ = [
    "HealthStatus",
    "ComponentStatus",
    "check_health",
    "check_database",
    "check_vector_store",
    "check_stale_documents",
]

logger = get_logger(__name__)


@dataclass
class ComponentStatus:
    """Status of an individual component."""

    name: str
    healthy: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Overall health status of the idx system."""

    is_healthy: bool
    components: list[ComponentStatus] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def add_component(self, status: ComponentStatus) -> None:
        """Add a component status."""
        self.components.append(status)
        if not status.healthy:
            self.issues.append(f"{status.name}: {status.message}")
            self.is_healthy = False


def check_database() -> ComponentStatus:
    """Check database connectivity.

    Attempts to connect to the database and execute a simple query.

    Returns:
        ComponentStatus indicating database health.
    """
    try:
        from sqlalchemy import text

        from idx.store.database import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        settings = get_settings()
        return ComponentStatus(
            name="database",
            healthy=True,
            message="Database connection successful",
            details={"path": str(settings.database_path)},
        )
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return ComponentStatus(
            name="database",
            healthy=False,
            message=f"Database connection failed: {e}",
        )


def check_vector_store() -> ComponentStatus:
    """Check vector store path accessibility.

    Verifies that the vector store directory exists or can be created.

    Returns:
        ComponentStatus indicating vector store health.
    """
    try:
        settings = get_settings()
        path = settings.vector_store_path

        if path.exists():
            if path.is_dir():
                return ComponentStatus(
                    name="vector_store",
                    healthy=True,
                    message="Vector store directory exists",
                    details={"path": str(path), "exists": True},
                )
            else:
                return ComponentStatus(
                    name="vector_store",
                    healthy=False,
                    message=f"Vector store path exists but is not a directory: {path}",
                )
        else:
            # Try to create the directory
            try:
                path.mkdir(parents=True, exist_ok=True)
                return ComponentStatus(
                    name="vector_store",
                    healthy=True,
                    message="Vector store directory created",
                    details={"path": str(path), "exists": False, "created": True},
                )
            except OSError as e:
                return ComponentStatus(
                    name="vector_store",
                    healthy=False,
                    message=f"Cannot create vector store directory: {e}",
                )
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        return ComponentStatus(
            name="vector_store",
            healthy=False,
            message=f"Vector store check failed: {e}",
        )


def check_fts_table() -> ComponentStatus:
    """Check FTS5 table existence.

    Verifies that the FTS5 virtual table exists in the database.

    Returns:
        ComponentStatus indicating FTS table health.
    """
    try:
        from sqlalchemy import text

        from idx.store.database import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'")
            )
            table_name = result.scalar()

        if table_name:
            return ComponentStatus(
                name="fts_table",
                healthy=True,
                message="FTS5 table exists",
            )
        else:
            return ComponentStatus(
                name="fts_table",
                healthy=False,
                message="FTS5 table does not exist - run create_fts_table()",
            )
    except Exception as e:
        logger.error(f"FTS table health check failed: {e}")
        return ComponentStatus(
            name="fts_table",
            healthy=False,
            message=f"FTS table check failed: {e}",
        )


def check_settings() -> ComponentStatus:
    """Check settings configuration.

    Verifies that settings can be loaded without errors.

    Returns:
        ComponentStatus indicating settings health.
    """
    try:
        settings = get_settings()
        return ComponentStatus(
            name="settings",
            healthy=True,
            message="Settings loaded successfully",
            details={
                "database_path": str(settings.database_path),
                "vector_store_path": str(settings.vector_store_path),
                "log_level": settings.log_level,
                "embedding_model": settings.embedding_model,
            },
        )
    except Exception as e:
        logger.error(f"Settings health check failed: {e}")
        return ComponentStatus(
            name="settings",
            healthy=False,
            message=f"Settings configuration error: {e}",
        )


def check_health(
    *,
    check_db: bool = True,
    check_vector: bool = True,
    check_fts: bool = True,
) -> HealthStatus:
    """Perform comprehensive health check.

    Checks all configured components and returns overall health status.

    Args:
        check_db: Include database check.
        check_vector: Include vector store check.
        check_fts: Include FTS table check.

    Returns:
        HealthStatus with component statuses and issues.

    Example:
        status = check_health()
        print(f"Healthy: {status.is_healthy}")
        for component in status.components:
            print(f"  {component.name}: {component.message}")
    """
    status = HealthStatus(is_healthy=True)

    # Always check settings
    status.add_component(check_settings())

    if check_db:
        status.add_component(check_database())

    if check_vector:
        status.add_component(check_vector_store())

    if check_fts:
        status.add_component(check_fts_table())

    if status.is_healthy:
        logger.debug("Health check passed")
    else:
        logger.warning(f"Health check failed: {status.issues}")

    return status


def check_stale_documents(
    dataset_id: int,
    source_path: Path,
    patterns: list[str] | None = None,
) -> ComponentStatus:
    """Check for stale documents in a dataset.

    Compares indexed documents in the database against current source files
    to identify stale paths (documents in DB but no longer in source).

    This is a read-only check that does NOT modify any data.

    Args:
        dataset_id: The dataset ID to check.
        source_path: Path to the source directory.
        patterns: Glob patterns for matching source files.
            Defaults to ["**/*.md"] if None.

    Returns:
        ComponentStatus with stale document details.
        - healthy=True if no stale documents found
        - healthy=False if stale documents exist
        - details contains: stale_count, stale_paths, source_count, indexed_count
    """
    try:
        from idx.ingest.directory import DirectorySource
        from idx.store.database import get_session
        from idx.store.repositories import DocumentRepository

        # Enumerate current source files
        source = DirectorySource(source_path, patterns=patterns)
        source_paths: set[str] = set()

        for doc in source.enumerate():
            source_paths.add(doc.relative_path)

        # Get indexed paths from DB
        with get_session() as session:
            doc_repo = DocumentRepository(session)
            indexed_paths = doc_repo.list_paths_by_dataset(dataset_id, active_only=True)

        # Find stale paths (in DB but not in source)
        stale_paths = indexed_paths - source_paths

        if stale_paths:
            return ComponentStatus(
                name="stale_documents",
                healthy=False,
                message=f"Found {len(stale_paths)} stale document(s) in dataset {dataset_id}",
                details={
                    "stale_count": len(stale_paths),
                    "stale_paths": sorted(stale_paths),
                    "source_count": len(source_paths),
                    "indexed_count": len(indexed_paths),
                },
            )
        else:
            return ComponentStatus(
                name="stale_documents",
                healthy=True,
                message="No stale documents found",
                details={
                    "stale_count": 0,
                    "stale_paths": [],
                    "source_count": len(source_paths),
                    "indexed_count": len(indexed_paths),
                },
            )

    except FileNotFoundError as e:
        logger.error(f"Stale document check failed - source not found: {e}")
        return ComponentStatus(
            name="stale_documents",
            healthy=False,
            message=f"Source directory not found: {source_path}",
            details={"error": str(e)},
        )
    except Exception as e:
        logger.error(f"Stale document check failed: {e}")
        return ComponentStatus(
            name="stale_documents",
            healthy=False,
            message=f"Stale document check failed: {e}",
            details={"error": str(e)},
        )
