"""Tests for idx.store.session_context module."""

import pytest
from unittest.mock import MagicMock

from idx.store.session_context import (
    SessionNotSetError,
    clear_session,
    current_session,
    use_session,
)


class TestCurrentSession:
    """Tests for current_session()."""

    def test_raises_when_not_set(self) -> None:
        """Raises SessionNotSetError when no ambient session is set."""
        clear_session()
        with pytest.raises(SessionNotSetError) as exc_info:
            current_session()
        assert "No ambient session set" in str(exc_info.value)

    def test_returns_session_when_set(self) -> None:
        """Returns the ambient session when set."""
        mock_session = MagicMock()
        with use_session(mock_session):
            result = current_session()
        assert result is mock_session

    def test_session_cleared_after_context(self) -> None:
        """Session is cleared when context exits."""
        mock_session = MagicMock()
        with use_session(mock_session):
            assert current_session() is mock_session
        with pytest.raises(SessionNotSetError):
            current_session()


class TestUseSession:
    """Tests for use_session() context manager."""

    def test_sets_session_in_context(self) -> None:
        """Sets the ambient session for the duration of the context."""
        mock_session = MagicMock()
        with use_session(mock_session):
            assert current_session() is mock_session

    def test_restores_previous_session_on_exit(self) -> None:
        """Restores the previous session when context exits."""
        outer_session = MagicMock(name="outer")
        inner_session = MagicMock(name="inner")

        with use_session(outer_session):
            assert current_session() is outer_session
            with use_session(inner_session):
                assert current_session() is inner_session
            assert current_session() is outer_session

    def test_clears_session_on_exception(self) -> None:
        """Clears session even when exception is raised."""
        mock_session = MagicMock()

        with pytest.raises(ValueError):
            with use_session(mock_session):
                assert current_session() is mock_session
                raise ValueError("test error")

        with pytest.raises(SessionNotSetError):
            current_session()

    def test_nested_contexts(self) -> None:
        """Handles deeply nested contexts correctly."""
        sessions = [MagicMock(name=f"session_{i}") for i in range(3)]

        with use_session(sessions[0]):
            assert current_session() is sessions[0]
            with use_session(sessions[1]):
                assert current_session() is sessions[1]
                with use_session(sessions[2]):
                    assert current_session() is sessions[2]
                assert current_session() is sessions[1]
            assert current_session() is sessions[0]


class TestClearSession:
    """Tests for clear_session()."""

    def test_clears_existing_session(self) -> None:
        """Clears the ambient session."""
        mock_session = MagicMock()
        with use_session(mock_session):
            assert current_session() is mock_session
            clear_session()
            with pytest.raises(SessionNotSetError):
                current_session()

    def test_idempotent_when_no_session(self) -> None:
        """Does not raise when no session is set."""
        clear_session()  # Should not raise
        clear_session()  # Should not raise


class TestSessionNotSetError:
    """Tests for SessionNotSetError."""

    def test_error_message(self) -> None:
        """Error message is helpful."""
        error = SessionNotSetError()
        assert "No ambient session set" in str(error)
        assert "use_session" in str(error)
