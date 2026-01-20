"""idx.core - Core infrastructure (settings, logging, status)."""

from idx.core.logging import configure_logging, get_logger
from idx.core.observability import configure_observability
from idx.core.settings import Settings, get_settings
from idx.core.status import (
    ComponentStatus,
    HealthStatus,
    check_database,
    check_health,
    check_vector_store,
)

__all__ = [
    "Settings",
    "configure_logging",
    "configure_observability",
    "get_logger",
    "get_settings",
    # Status
    "ComponentStatus",
    "HealthStatus",
    "check_database",
    "check_health",
    "check_vector_store",
]
