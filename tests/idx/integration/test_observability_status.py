"""Integration smoke tests for observability and status checks.

Verifies that Langfuse wiring and status checks don't crash
in basic configurations.
"""

from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from idx.store.database import Base, create_engine_for_path


class TestObservabilitySmoke:
    """Smoke tests for observability configuration."""

    def test_observability_module_imports(self) -> None:
        """Observability module imports without error."""
        from idx.core.observability import (
            configure_observability,
            get_callback_manager,
            is_langfuse_available,
            is_observability_configured,
            reset_observability,
        )

        # All functions should be callable
        assert callable(configure_observability)
        assert callable(get_callback_manager)
        assert callable(is_langfuse_available)
        assert callable(is_observability_configured)
        assert callable(reset_observability)

    def test_observability_disabled_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Observability is safe when disabled (default state)."""
        from idx.core.observability import (
            configure_observability,
            get_callback_manager,
            reset_observability,
        )
        from idx.core.settings import get_settings

        # Reset state
        reset_observability()

        # Ensure disabled
        monkeypatch.setenv("IDX_LANGFUSE_ENABLED", "false")
        get_settings.cache_clear()

        # Should not crash
        result = configure_observability()
        assert result is False

        # Callback manager should be None (safe for callers to check)
        manager = get_callback_manager()
        assert manager is None

        # Cleanup
        reset_observability()

    def test_langfuse_availability_check(self) -> None:
        """is_langfuse_available returns boolean without crashing."""
        from idx.core.observability import is_langfuse_available

        result = is_langfuse_available()
        assert isinstance(result, bool)


class TestStatusSmoke:
    """Smoke tests for status/health checks."""

    def test_status_module_imports(self) -> None:
        """Status module imports without error."""
        from idx.core.status import (
            ComponentStatus,
            HealthStatus,
            check_database,
            check_fts_table,
            check_health,
            check_settings,
            check_vector_store,
        )

        # All should be importable
        assert ComponentStatus is not None
        assert HealthStatus is not None
        assert callable(check_health)
        assert callable(check_settings)
        assert callable(check_database)
        assert callable(check_vector_store)
        assert callable(check_fts_table)

    def test_check_settings_runs(self, tmp_path: Path) -> None:
        """check_settings runs without crashing."""
        from idx.core.status import check_settings

        # Should not crash even with default/test settings
        result = check_settings()

        # Should return ComponentStatus
        assert hasattr(result, "name")
        assert hasattr(result, "healthy")
        assert hasattr(result, "message")
        assert result.name == "settings"

    def test_health_status_dataclass(self) -> None:
        """HealthStatus dataclass works correctly."""
        from idx.core.status import ComponentStatus, HealthStatus

        # Create and manipulate
        status = HealthStatus(is_healthy=True)

        status.add_component(
            ComponentStatus(name="test", healthy=True, message="OK")
        )

        assert status.is_healthy is True
        assert len(status.components) == 1

    def test_full_health_check_with_real_db(self, tmp_path: Path) -> None:
        """Full health check runs with a real temporary database."""
        from idx.core.status import check_health
        from idx.store.fts import create_fts_table

        # Create a real test database
        db_path = tmp_path / "test.db"
        vector_path = tmp_path / "vectors"
        vector_path.mkdir()

        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)
        create_fts_table(engine)

        # Patch settings to use our test paths
        from unittest.mock import MagicMock, patch

        with (
            patch("idx.store.database.get_engine") as mock_get_engine,
            patch("idx.core.status.get_settings") as mock_settings,
        ):
            mock_get_engine.return_value = engine
            mock_settings.return_value = MagicMock(
                database_path=db_path,
                vector_store_path=vector_path,
                log_level="INFO",
                embedding_model="default",
            )

            result = check_health()

        # Should complete without error
        assert result.is_healthy is True
        assert len(result.components) == 4  # settings, db, vector, fts
        assert result.issues == []


class TestEndToEndStatusIntegration:
    """End-to-end integration tests for status checks."""

    def test_status_after_ingest(self, tmp_path: Path, patched_embedding) -> None:
        """Health check passes after document ingestion."""
        from contextlib import contextmanager
        from unittest.mock import MagicMock, patch

        from idx.core.status import check_health
        from idx.ingest.pipelines import IngestPipeline
        from idx.ingest.schemas import IngestDirectoryConfig
        from idx.store.fts import create_fts_table

        # Create test documents
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test\n\nSome content.")

        # Setup database
        db_path = tmp_path / "test.db"
        vector_path = tmp_path / "vectors"
        vector_path.mkdir()

        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)
        create_fts_table(engine)

        # Create session factory
        factory = sessionmaker(bind=engine, expire_on_commit=False)

        @contextmanager
        def get_test_session():
            session = factory()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        # Ingest documents using patched session
        with patch("idx.ingest.pipelines.get_session", get_test_session):
            pipeline = IngestPipeline()
            config = IngestDirectoryConfig(
                source_path=docs_dir,
                dataset_name="test",
                patterns=["**/*.md"],
            )
            result = pipeline.ingest(config)

        assert result.documents_created == 1

        # Check health with the same database
        with (
            patch("idx.store.database.get_engine") as mock_get_engine,
            patch("idx.core.status.get_settings") as mock_settings,
        ):
            mock_get_engine.return_value = engine
            mock_settings.return_value = MagicMock(
                database_path=db_path,
                vector_store_path=vector_path,
                log_level="INFO",
                embedding_model="default",
            )

            health = check_health()

        assert health.is_healthy is True

    def test_component_status_serialization(self) -> None:
        """ComponentStatus can be converted to dict for API responses."""
        from dataclasses import asdict

        from idx.core.status import ComponentStatus

        status = ComponentStatus(
            name="database",
            healthy=True,
            message="Connected",
            details={"path": "/tmp/test.db", "size_mb": 1.5},
        )

        # Should serialize to dict
        d = asdict(status)
        assert d["name"] == "database"
        assert d["healthy"] is True
        assert d["message"] == "Connected"
        assert d["details"]["path"] == "/tmp/test.db"

    def test_health_status_summary(self) -> None:
        """HealthStatus provides useful summary information."""
        from idx.core.status import ComponentStatus, HealthStatus

        status = HealthStatus(is_healthy=True)
        status.add_component(
            ComponentStatus(name="a", healthy=True, message="OK")
        )
        status.add_component(
            ComponentStatus(name="b", healthy=False, message="Error")
        )
        status.add_component(
            ComponentStatus(name="c", healthy=True, message="OK")
        )

        # Should track unhealthy state
        assert status.is_healthy is False
        assert len(status.issues) == 1
        assert "b: Error" in status.issues[0]

        # Can get component by name
        names = [c.name for c in status.components]
        assert names == ["a", "b", "c"]
