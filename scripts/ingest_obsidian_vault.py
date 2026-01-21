# /// script
# dependencies = [
#   "idx",
# ]
# ///

"""Ingest an Obsidian vault into the idx database.

Run from the project root:
    uv run python scripts/ingest_obsidian_vault.py /path/to/vault

By default, the database path comes from `idx.core.settings` (env var
`IDX_DATABASE_PATH`, or `~/.idx/idx.db`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from idx.core.logging import configure_logging, get_logger
from idx.core.settings import get_settings
from idx.pipelines.ingest import IngestPipeline
from idx.pipelines.schemas import IngestObsidianConfig
from idx.store.database import Base, create_engine_for_path
from idx.store.fts import create_fts_table

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest_obsidian_vault",
        description="Ingest an Obsidian vault using idx.pipelines.ingest.",
    )
    parser.add_argument(
        "vault_path",
        type=Path,
        help="Filesystem path to an Obsidian vault (must contain a .obsidian directory).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name to ingest into (default: vault directory name).",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help="SQLite DB path (default: settings.database_path / IDX_DATABASE_PATH).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all documents even if unchanged.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--echo-sql",
        action="store_true",
        help="Enable SQLAlchemy echo for debugging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argv list (defaults to sys.argv when None).

    Returns:
        Process exit code (0 for success, non-zero for errors).
    """

    args = _build_parser().parse_args(argv)

    configure_logging(level=args.log_level)

    vault_path: Path = args.vault_path.expanduser().resolve()
    dataset_name: str = args.dataset_name or vault_path.name
    database_path: Path = (
        args.database_path.expanduser().resolve()
        if args.database_path is not None
        else get_settings().database_path
    )

    logger.info(f"Vault: {vault_path}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Database: {database_path}")

    engine = create_engine_for_path(database_path, echo=bool(args.echo_sql))
    Base.metadata.create_all(engine)
    create_fts_table(engine)

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        pipeline = IngestPipeline(session_factory=sessionmaker(bind=engine, expire_on_commit=False))
        result = pipeline.ingest(
            IngestObsidianConfig(
                source_path=vault_path,
                dataset_name=dataset_name,
                force=bool(args.force),
            )
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Ingestion failed.")
        return 1
    finally:
        session.close()

    print(f"dataset_id={result.dataset_id} dataset_name={result.dataset_name}")
    print(
        "created={created} updated={updated} skipped={skipped} stale={stale} failed={failed}".format(
            created=result.documents_created,
            updated=result.documents_updated,
            skipped=result.documents_skipped,
            stale=result.documents_stale,
            failed=result.documents_failed,
        )
    )
    if result.errors:
        print("errors:")
        for err in result.errors:
            print(f"- {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

