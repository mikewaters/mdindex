"""Tests for idx.store.database module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import Column, Integer, String, text
from sqlalchemy.orm import Session

from idx.store.database import (
    Base,
    create_engine_for_path,
    get_session,
)


class TestModel(Base):
    """Test model for database operations."""

    __tablename__ = "test_models"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)


class TestCreateEngineForPath:
    """Tests for create_engine_for_path function."""

    def test_creates_engine(self, tmp_path: Path) -> None:
        """Engine is created successfully."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)

        assert engine is not None
        assert str(db_path) in str(engine.url)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Parent directory is created if it doesn't exist."""
        db_path = tmp_path / "subdir" / "nested" / "test.db"
        assert not db_path.parent.exists()

        create_engine_for_path(db_path)

        assert db_path.parent.exists()

    def test_echo_parameter(self, tmp_path: Path) -> None:
        """Echo parameter is respected."""
        db_path = tmp_path / "test.db"

        engine_quiet = create_engine_for_path(db_path, echo=False)
        assert engine_quiet.echo is False

        engine_verbose = create_engine_for_path(db_path, echo=True)
        assert engine_verbose.echo is True

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        """WAL journal mode is enabled."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)

        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode"))
            mode = result.scalar()
            assert mode == "wal"

    def test_foreign_keys_enabled(self, tmp_path: Path) -> None:
        """Foreign keys are enabled."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)

        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA foreign_keys"))
            fk_enabled = result.scalar()
            assert fk_enabled == 1


class TestGetSession:
    """Tests for get_session context manager."""

    def test_session_commits_on_success(self, tmp_path: Path) -> None:
        """Session commits automatically on successful exit."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)

        # Patch get_engine to use our test engine
        with patch("idx.store.database.get_engine", return_value=engine):
            # Clear the lru_cache for get_session_factory
            from idx.store.database import get_session_factory

            get_session_factory.cache_clear()

            with get_session() as session:
                model = TestModel(name="test")
                session.add(model)

            # Verify data was committed
            with get_session() as session:
                result = session.query(TestModel).filter_by(name="test").first()
                assert result is not None
                assert result.name == "test"

    def test_session_rollbacks_on_exception(self, tmp_path: Path) -> None:
        """Session rolls back on exception."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)

        with patch("idx.store.database.get_engine", return_value=engine):
            from idx.store.database import get_session_factory

            get_session_factory.cache_clear()

            with pytest.raises(ValueError):
                with get_session() as session:
                    model = TestModel(name="will_be_rolled_back")
                    session.add(model)
                    raise ValueError("Test error")

            # Verify data was not committed
            with get_session() as session:
                result = session.query(TestModel).filter_by(name="will_be_rolled_back").first()
                assert result is None

    def test_session_yields_valid_session(self, tmp_path: Path) -> None:
        """get_session yields a valid Session instance."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)

        with patch("idx.store.database.get_engine", return_value=engine):
            from idx.store.database import get_session_factory

            get_session_factory.cache_clear()

            with get_session() as session:
                assert isinstance(session, Session)
                assert not session.is_active or session.get_bind() is not None


class TestBase:
    """Tests for Base declarative class."""

    def test_base_creates_tables(self, tmp_path: Path) -> None:
        """Base.metadata.create_all creates tables."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)

        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='test_models'")
            )
            table_name = result.scalar()
            assert table_name == "test_models"


class TestSettingsIntegration:
    """Tests for settings integration."""

    def test_get_engine_uses_settings_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_engine uses database_path from settings."""
        from idx.store.database import get_engine

        # Clear cached engine
        get_engine.cache_clear()

        # Set environment variable
        db_path = tmp_path / "settings_test.db"
        monkeypatch.setenv("IDX_DATABASE_PATH", str(db_path))

        # Clear settings cache too
        from idx.core.settings import get_settings

        get_settings.cache_clear()

        engine = get_engine()
        assert str(db_path) in str(engine.url)

        # Cleanup
        get_engine.cache_clear()
        get_settings.cache_clear()
