"""Steps for directory ingestion scenarios."""

from __future__ import annotations

from pathlib import Path

from pytest_bdd import given, then, when
from pytest_bdd.parsers import parse

from idx.ingest.schemas import IngestDirectoryConfig, IngestResult
from idx.store.fts import FTSManager
from idx.store.repositories import DatasetRepository, DocumentRepository
from tests.idx.scenario.pipelines.conftest import ScenarioContext


def _split_csv(value: str) -> list[str]:
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


@given("a temporary database with FTS enabled")
def _given_db(ctx: ScenarioContext, db_session) -> None:
    """Bind the scenario DB session into the context."""

    ctx.session = db_session


@given("an ingest pipeline")
def _given_pipeline(ctx: ScenarioContext, pipeline) -> None:
    """Bind an ingest pipeline into the context."""

    ctx.pipeline = pipeline


@given("a sample directory with markdown files")
def _given_sample_directory(ctx: ScenarioContext, sample_directory: Path) -> None:
    """Use a non-empty directory with a nested markdown file."""

    ctx.directory = sample_directory


@given("an empty directory")
def _given_empty_directory(ctx: ScenarioContext, tmp_path: Path) -> None:
    """Use an empty directory as the ingestion source."""

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ctx.directory = empty_dir


@given("a missing directory path")
def _given_missing_directory(ctx: ScenarioContext, tmp_path: Path) -> None:
    """Use a directory path that does not exist."""

    ctx.directory = tmp_path / "nonexistent"


@given(parse('I have ingested the directory as dataset "{dataset_name}"'))
def _given_initial_ingest(ctx: ScenarioContext, dataset_name: str) -> None:
    """Run a first ingestion so subsequent steps can re-ingest."""

    assert ctx.pipeline is not None
    assert ctx.directory is not None
    config = IngestDirectoryConfig(source_path=ctx.directory, dataset_name=dataset_name)
    ctx.values["initial_result"] = ctx.pipeline.ingest_directory(config)


@given(parse('I change the file "{relative_path}" content'))
def _given_change_file(ctx: ScenarioContext, relative_path: str) -> None:
    """Modify a markdown file on disk to force a re-ingestion update."""

    assert ctx.directory is not None
    path = ctx.directory / relative_path
    path.write_text("# Updated Readme\n\nNew content.")


@given(parse('the file "{relative_path}" is deleted from disk'))
def _given_delete_file(ctx: ScenarioContext, relative_path: str) -> None:
    """Delete a file from disk to trigger stale detection."""

    assert ctx.directory is not None
    (ctx.directory / relative_path).unlink()


@given(parse('the file "{old_path}" is renamed to "{new_path}"'))
def _given_rename_file(ctx: ScenarioContext, old_path: str, new_path: str) -> None:
    """Rename a file by delete+create to mirror unit test behavior."""

    assert ctx.directory is not None
    old_abs = ctx.directory / old_path
    content = old_abs.read_text()
    old_abs.unlink()
    (ctx.directory / new_path).write_text(content)


@when(parse('I ingest the directory as dataset "{dataset_name}" with patterns "{patterns}"'))
def _when_ingest(ctx: ScenarioContext, dataset_name: str, patterns: str) -> None:
    """Ingest a directory into the scenario database."""

    assert ctx.pipeline is not None
    assert ctx.directory is not None
    config = IngestDirectoryConfig(
        source_path=ctx.directory,
        dataset_name=dataset_name,
        patterns=_split_csv(patterns),
    )
    ctx.last_result = ctx.pipeline.ingest_directory(config)


@when(parse('I ingest the directory as dataset "{dataset_name}" with patterns "{patterns}" in force mode'))
def _when_ingest_force(ctx: ScenarioContext, dataset_name: str, patterns: str) -> None:
    """Ingest with `force=True` to update all documents."""

    assert ctx.pipeline is not None
    assert ctx.directory is not None
    config = IngestDirectoryConfig(
        source_path=ctx.directory,
        dataset_name=dataset_name,
        patterns=_split_csv(patterns),
        force=True,
    )
    ctx.last_result = ctx.pipeline.ingest_directory(config)


@when(parse('I ingest the directory as dataset "{dataset_name}" with patterns "{patterns}" again'))
def _when_ingest_again(ctx: ScenarioContext, dataset_name: str, patterns: str) -> None:
    """Ingest a second time and keep both dataset ids for comparison."""

    _when_ingest(ctx, dataset_name, patterns)
    result = _require_result(ctx)
    previous_id = ctx.values.get("previous_dataset_id")
    if previous_id is None:
        ctx.values["previous_dataset_id"] = result.dataset_id
    else:
        ctx.values["second_dataset_id"] = result.dataset_id


@when(parse('I attempt to ingest the directory as dataset "{dataset_name}" with patterns "{patterns}"'))
def _when_attempt_ingest(ctx: ScenarioContext, dataset_name: str, patterns: str) -> None:
    """Attempt ingestion and store the raised exception, if any."""

    assert ctx.pipeline is not None
    assert ctx.directory is not None
    config = IngestDirectoryConfig(
        source_path=ctx.directory,
        dataset_name=dataset_name,
        patterns=_split_csv(patterns),
    )
    ctx.errors.clear()
    try:
        ctx.pipeline.ingest_directory(config)
    except BaseException as exc:  # noqa: BLE001 - testing for a specific exception type
        ctx.errors.append(exc)


@then(parse('the ingest result dataset name is "{expected}"'))
def _then_dataset_name(ctx: ScenarioContext, expected: str) -> None:
    """Assert dataset name on the last ingest result."""

    result = _require_result(ctx)
    assert result.dataset_name == expected


@then("the ingest result dataset id is positive")
def _then_dataset_id_positive(ctx: ScenarioContext) -> None:
    """Assert dataset_id is a positive integer."""

    result = _require_result(ctx)
    assert result.dataset_id > 0


@then(parse('a dataset named "{dataset_name}" exists with source type "{source_type}"'))
def _then_dataset_exists(ctx: ScenarioContext, dataset_name: str, source_type: str) -> None:
    """Verify a dataset record exists with the expected source_type."""

    assert ctx.session is not None
    repo = DatasetRepository(ctx.session)
    dataset = repo.get_by_name(dataset_name)
    assert dataset is not None
    assert dataset.source_type == source_type


@then(parse("the ingest result counts are created={created:d} updated={updated:d} skipped={skipped:d} stale={stale:d} failed={failed:d}"))
def _then_counts(
    ctx: ScenarioContext,
    created: int,
    updated: int,
    skipped: int,
    stale: int,
    failed: int,
) -> None:
    """Assert the ingest result counters match expected values."""

    result = _require_result(ctx)
    assert result.documents_created == created
    assert result.documents_updated == updated
    assert result.documents_skipped == skipped
    assert result.documents_stale == stale
    assert result.documents_failed == failed


@then(parse('the dataset documents are exactly "{paths}"'))
def _then_documents_exact(ctx: ScenarioContext, paths: str) -> None:
    """Assert dataset document paths equal the expected comma-separated list."""

    result = _require_result(ctx)
    assert ctx.session is not None
    repo = DocumentRepository(ctx.session)
    docs = repo.list_by_dataset(result.dataset_id)
    actual = sorted({d.path for d in docs})
    expected = sorted(_split_csv(paths))
    assert actual == expected


@then(parse("the FTS index contains {expected_count:d} entries"))
def _then_fts_count(ctx: ScenarioContext, expected_count: int) -> None:
    """Assert FTS index row count equals expected value."""

    assert ctx.session is not None
    fts = FTSManager(ctx.session)
    assert fts.count() == expected_count


@then(parse('searching the FTS index for "{query}" returns at least {min_count:d} result'))
def _then_fts_search_min(ctx: ScenarioContext, query: str, min_count: int) -> None:
    """Assert FTS search returns at least a minimum number of hits."""

    assert ctx.session is not None
    fts = FTSManager(ctx.session)
    results = fts.search(query)
    assert len(results) >= min_count


@then(parse('searching the FTS index for "{query}" returns 0 results'))
def _then_fts_search_none(ctx: ScenarioContext, query: str) -> None:
    """Assert FTS search returns no results."""

    assert ctx.session is not None
    fts = FTSManager(ctx.session)
    results = fts.search(query)
    assert len(results) == 0


@then(parse('the FTS search results for "{query}" include a path containing "{needle}"'))
def _then_fts_search_includes(ctx: ScenarioContext, query: str, needle: str) -> None:
    """Assert at least one FTS result path contains a substring."""

    assert ctx.session is not None
    fts = FTSManager(ctx.session)
    results = fts.search(query)
    assert any(needle in r.path for r in results)


@then("both ingestions use the same dataset id")
def _then_same_dataset_id(ctx: ScenarioContext) -> None:
    """Assert two ingestions reused the same dataset id."""

    assert ctx.values.get("previous_dataset_id") == ctx.values.get("second_dataset_id")


@then(parse("the ingest result total processed is {expected_total:d}"))
def _then_total_processed(ctx: ScenarioContext, expected_total: int) -> None:
    """Assert total_processed property matches expected."""

    result = _require_result(ctx)
    assert result.total_processed == expected_total


@then("the ingest result is successful")
def _then_success(ctx: ScenarioContext) -> None:
    """Assert the ingest result indicates success."""

    result = _require_result(ctx)
    assert result.success is True


@then("the ingest result has a completed timestamp not earlier than started")
def _then_timestamps(ctx: ScenarioContext) -> None:
    """Assert started_at and completed_at are present and ordered."""

    result = _require_result(ctx)
    assert result.completed_at is not None
    assert result.started_at <= result.completed_at


@then(parse('the active dataset document paths are exactly "{paths}"'))
def _then_active_paths(ctx: ScenarioContext, paths: str) -> None:
    """Assert active-only document paths equal the expected comma-separated list."""

    result = _require_result(ctx)
    assert ctx.session is not None
    repo = DocumentRepository(ctx.session)
    docs = repo.list_by_dataset(result.dataset_id, active_only=True)
    actual = sorted({d.path for d in docs})
    expected = sorted(_split_csv(paths))
    assert actual == expected


@then("a FileNotFoundError is raised")
def _then_filenotfound(ctx: ScenarioContext) -> None:
    """Assert a FileNotFoundError occurred in a prior attempt step."""

    assert ctx.errors, "Expected an error but none was captured."
    assert isinstance(ctx.errors[0], FileNotFoundError)


def _require_result(ctx: ScenarioContext) -> IngestResult:
    if ctx.last_result is None:
        raise AssertionError("No ingest result is available in the scenario context.")
    return ctx.last_result
