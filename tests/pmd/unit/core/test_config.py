"""Tests for Config TOML loading and env overrides."""

from pathlib import Path

from pmd.core.config import Config


def test_from_file_applies_toml_then_env(tmp_path: Path, monkeypatch):
    """Environment variables should override TOML values."""
    toml_path = tmp_path / "pmd.toml"
    toml_path.write_text(
        "\n".join(
            [
                'db_path = "/tmp/from_toml.db"',
                'llm_provider = "openrouter"',
                "",
                "[lm_studio]",
                'base_url = "http://toml-lm:1234"',
                "",
                "[openrouter]",
                'api_key = "toml-key"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("INDEX_PATH", "/tmp/from_env.db")
    monkeypatch.setenv("LLM_PROVIDER", "mlx")
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

    config = Config.from_file(toml_path)

    assert str(config.db_path) == "/tmp/from_env.db"
    assert config.llm_provider == "mlx"
    assert config.openrouter.api_key == "env-key"
    assert config.lm_studio.base_url == "http://toml-lm:1234"


def test_from_env_or_file_uses_pmd_config(tmp_path: Path, monkeypatch):
    """PMD_CONFIG should be used when no explicit path is provided."""
    toml_path = tmp_path / "pmd.toml"
    toml_path.write_text('llm_provider = "openrouter"', encoding="utf-8")

    monkeypatch.setenv("PMD_CONFIG", str(toml_path))

    config = Config.from_env_or_file()

    assert config.llm_provider == "openrouter"
