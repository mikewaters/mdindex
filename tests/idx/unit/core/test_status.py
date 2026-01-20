"""Tests for idx.core.status module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idx.core.status import (
    ComponentStatus,
    HealthStatus,
    check_database,
    check_fts_table,
    check_health,
    check_settings,
    check_vector_store,
)


class TestComponentStatus:
    """Tests for ComponentStatus dataclass."""

    def test_healthy_component(self) -> None:
        """Healthy component has correct attributes."""
        status = ComponentStatus(
            name="test",
            healthy=True,
            message="All good",
        )
        assert status.name == "test"
        assert status.healthy is True
        assert status.message == "All good"
        assert status.details == {}

    def test_unhealthy_component(self) -> None:
        """Unhealthy component has correct attributes."""
        status = ComponentStatus(
            name="test",
            healthy=False,
            message="Something wrong",
            details={"error": "details"},
        )
        assert status.name == "test"
        assert status.healthy is False
        assert status.message == "Something wrong"
        assert status.details == {"error": "details"}


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_initial_healthy(self) -> None:
        """HealthStatus starts healthy by default."""
        status = HealthStatus(is_healthy=True)
        assert status.is_healthy is True
        assert status.components == []
        assert status.issues == []

    def test_add_healthy_component(self) -> None:
        """Adding healthy component keeps status healthy."""
        status = HealthStatus(is_healthy=True)
        component = ComponentStatus(name="db", healthy=True, message="OK")
        status.add_component(component)

        assert status.is_healthy is True
        assert len(status.components) == 1
        assert status.issues == []

    def test_add_unhealthy_component(self) -> None:
        """Adding unhealthy component makes status unhealthy."""
        status = HealthStatus(is_healthy=True)
        component = ComponentStatus(name="db", healthy=False, message="Failed")
        status.add_component(component)

        assert status.is_healthy is False
        assert len(status.components) == 1
        assert "db: Failed" in status.issues

    def test_multiple_components(self) -> None:
        """Multiple components tracked correctly."""
        status = HealthStatus(is_healthy=True)
        status.add_component(ComponentStatus(name="a", healthy=True, message="OK"))
        status.add_component(ComponentStatus(name="b", healthy=False, message="Bad"))
        status.add_component(ComponentStatus(name="c", healthy=True, message="OK"))

        assert status.is_healthy is False
        assert len(status.components) == 3
        assert len(status.issues) == 1


class TestCheckSettings:
    """Tests for check_settings function."""

    def test_settings_load_success(self, tmp_path: Path) -> None:
        """Settings check succeeds when settings load."""
        with patch("idx.core.status.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                database_path=tmp_path / "test.db",
                vector_store_path=tmp_path / "vectors",
                log_level="INFO",
                embedding_model="default",
            )
            result = check_settings()

        assert result.healthy is True
        assert result.name == "settings"
        assert "database_path" in result.details

    def test_settings_load_failure(self) -> None:
        """Settings check fails when settings error."""
        with patch("idx.core.status.get_settings") as mock_settings:
            mock_settings.side_effect = ValueError("Bad config")
            result = check_settings()

        assert result.healthy is False
        assert "Bad config" in result.message


class TestCheckDatabase:
    """Tests for check_database function."""

    def test_database_connection_success(self, tmp_path: Path) -> None:
        """Database check succeeds when connection works."""
        from sqlalchemy import create_engine

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        # Create the database
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))

        with (
            patch("idx.store.database.get_engine") as mock_get_engine,
            patch("idx.core.status.get_settings") as mock_settings,
        ):
            mock_get_engine.return_value = engine
            mock_settings.return_value = MagicMock(database_path=db_path)
            result = check_database()

        assert result.healthy is True
        assert result.name == "database"

    def test_database_connection_failure(self) -> None:
        """Database check fails when connection fails."""
        with patch("idx.store.database.get_engine") as mock_get_engine:
            mock_get_engine.side_effect = Exception("Connection refused")
            result = check_database()

        assert result.healthy is False
        assert "Connection refused" in result.message


class TestCheckVectorStore:
    """Tests for check_vector_store function."""

    def test_vector_store_exists(self, tmp_path: Path) -> None:
        """Vector store check succeeds when directory exists."""
        vector_path = tmp_path / "vectors"
        vector_path.mkdir()

        with patch("idx.core.status.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(vector_store_path=vector_path)
            result = check_vector_store()

        assert result.healthy is True
        assert result.details["exists"] is True

    def test_vector_store_created(self, tmp_path: Path) -> None:
        """Vector store check succeeds when directory can be created."""
        vector_path = tmp_path / "new_vectors"

        with patch("idx.core.status.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(vector_store_path=vector_path)
            result = check_vector_store()

        assert result.healthy is True
        assert result.details.get("created") is True
        assert vector_path.exists()

    def test_vector_store_not_directory(self, tmp_path: Path) -> None:
        """Vector store check fails when path is a file."""
        file_path = tmp_path / "not_a_dir"
        file_path.touch()

        with patch("idx.core.status.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(vector_store_path=file_path)
            result = check_vector_store()

        assert result.healthy is False
        assert "not a directory" in result.message


class TestCheckFtsTable:
    """Tests for check_fts_table function."""

    def test_fts_table_exists(self, tmp_path: Path) -> None:
        """FTS check succeeds when table exists."""
        from sqlalchemy import create_engine, text

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(
                text("CREATE VIRTUAL TABLE documents_fts USING fts5(content)")
            )
            conn.commit()

        with patch("idx.store.database.get_engine") as mock_get_engine:
            mock_get_engine.return_value = engine
            result = check_fts_table()

        assert result.healthy is True
        assert result.name == "fts_table"

    def test_fts_table_missing(self, tmp_path: Path) -> None:
        """FTS check fails when table doesn't exist."""
        from sqlalchemy import create_engine

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")

        with patch("idx.store.database.get_engine") as mock_get_engine:
            mock_get_engine.return_value = engine
            result = check_fts_table()

        assert result.healthy is False
        assert "does not exist" in result.message


class TestCheckHealth:
    """Tests for check_health function."""

    def test_all_healthy(self, tmp_path: Path) -> None:
        """Full health check succeeds when all components healthy."""
        from sqlalchemy import create_engine, text

        db_path = tmp_path / "test.db"
        vector_path = tmp_path / "vectors"
        vector_path.mkdir()

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(
                text("CREATE VIRTUAL TABLE documents_fts USING fts5(content)")
            )
            conn.commit()

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

        assert result.is_healthy is True
        assert len(result.components) == 4  # settings, db, vector, fts
        assert result.issues == []

    def test_selective_checks(self, tmp_path: Path) -> None:
        """Health check can skip components."""
        with patch("idx.core.status.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                database_path=tmp_path / "test.db",
                vector_store_path=tmp_path / "vectors",
                log_level="INFO",
                embedding_model="default",
            )
            result = check_health(check_db=False, check_vector=False, check_fts=False)

        assert result.is_healthy is True
        assert len(result.components) == 1  # Only settings

    def test_partial_failure(self, tmp_path: Path) -> None:
        """Health check reports partial failures."""
        vector_path = tmp_path / "vectors"
        vector_path.mkdir()

        with (
            patch("idx.store.database.get_engine") as mock_get_engine,
            patch("idx.core.status.get_settings") as mock_settings,
        ):
            mock_get_engine.side_effect = Exception("DB error")
            mock_settings.return_value = MagicMock(
                database_path=tmp_path / "test.db",
                vector_store_path=vector_path,
                log_level="INFO",
                embedding_model="default",
            )
            result = check_health(check_fts=False)

        assert result.is_healthy is False
        assert len(result.issues) == 1
        assert "database" in result.issues[0].lower()
