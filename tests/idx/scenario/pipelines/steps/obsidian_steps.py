"""Steps for Obsidian vault ingestion scenarios."""

from __future__ import annotations

import json

from pytest_bdd import given, then, when
from pytest_bdd.parsers import parse

from idx.ingest.schemas import IngestObsidianConfig, IngestResult
from idx.store.fts import FTSManager
from idx.store.repositories import DatasetRepository, DocumentRepository
from tests.idx.scenario.pipelines.conftest import ScenarioContext


@given("a valid Obsidian vault")
def _given_vault(ctx: ScenarioContext, obsidian_vault) -> None:
    """Bind a valid Obsidian vault path into the context."""

    ctx.obsidian_vault = obsidian_vault


@given("an invalid Obsidian vault")
def _given_invalid_vault(ctx: ScenarioContext, invalid_obsidian_vault) -> None:
    """Bind an invalid Obsidian vault path into the context."""

    ctx.obsidian_vault = invalid_obsidian_vault


@given(parse('I have ingested the Obsidian vault as dataset "{dataset_name}"'))
def _given_ingested(ctx: ScenarioContext, dataset_name: str) -> None:
    """Run a first Obsidian ingestion for re-ingestion scenarios."""

    assert ctx.pipeline is not None
    assert ctx.obsidian_vault is not None
    config = IngestObsidianConfig(source_path=ctx.obsidian_vault, dataset_name=dataset_name)
    ctx.values["initial_obsidian_result"] = ctx.pipeline.ingest_obsidian(config)


@when(parse('I ingest the Obsidian vault as dataset "{dataset_name}"'))
def _when_ingest_obsidian(ctx: ScenarioContext, dataset_name: str) -> None:
    """Ingest an Obsidian vault."""

    assert ctx.pipeline is not None
    assert ctx.obsidian_vault is not None
    config = IngestObsidianConfig(source_path=ctx.obsidian_vault, dataset_name=dataset_name)
    ctx.last_result = ctx.pipeline.ingest_obsidian(config)


@when(parse('I ingest the Obsidian vault as dataset "{dataset_name}" in force mode'))
def _when_ingest_obsidian_force(ctx: ScenarioContext, dataset_name: str) -> None:
    """Ingest an Obsidian vault with force mode enabled."""

    assert ctx.pipeline is not None
    assert ctx.obsidian_vault is not None
    config = IngestObsidianConfig(
        source_path=ctx.obsidian_vault,
        dataset_name=dataset_name,
        force=True,
    )
    ctx.last_result = ctx.pipeline.ingest_obsidian(config)


@when(parse('I attempt to ingest the Obsidian vault as dataset "{dataset_name}"'))
def _when_attempt_obsidian(ctx: ScenarioContext, dataset_name: str) -> None:
    """Attempt Obsidian ingestion and capture any exception."""

    assert ctx.pipeline is not None
    assert ctx.obsidian_vault is not None
    ctx.errors.clear()
    config = IngestObsidianConfig(source_path=ctx.obsidian_vault, dataset_name=dataset_name)
    try:
        ctx.pipeline.ingest_obsidian(config)
    except BaseException as exc:  # noqa: BLE001 - testing for a specific exception type
        ctx.errors.append(exc)


@then(parse('a dataset named "{dataset_name}" exists with source type "{source_type}"'))
def _then_dataset_exists(ctx: ScenarioContext, dataset_name: str, source_type: str) -> None:
    """Verify a dataset record exists with the expected source_type."""

    assert ctx.session is not None
    repo = DatasetRepository(ctx.session)
    dataset = repo.get_by_name(dataset_name)
    assert dataset is not None
    assert dataset.source_type == source_type


@then(parse("the obsidian ingest result counts are created={created:d} updated={updated:d} skipped={skipped:d} failed={failed:d}"))
def _then_counts(ctx: ScenarioContext, created: int, updated: int, skipped: int, failed: int) -> None:
    """Assert Obsidian ingest counters match expected values."""

    result = _require_result(ctx)
    assert result.documents_created == created
    assert result.documents_updated == updated
    assert result.documents_skipped == skipped
    assert result.documents_failed == failed


@then(parse('the document "{path}" metadata includes tags "{tags}"'))
def _then_metadata_tags(ctx: ScenarioContext, path: str, tags: str) -> None:
    """Assert document metadata contains the expected tags."""

    result = _require_result(ctx)
    assert ctx.session is not None
    repo = DocumentRepository(ctx.session)
    doc = repo.get_by_path(result.dataset_id, path)
    assert doc is not None
    assert doc.metadata_json is not None
    metadata = json.loads(doc.metadata_json)
    expected = [t.strip() for t in tags.split(",") if t.strip()]
    for tag in expected:
        assert tag in metadata.get("tags", [])


@then(parse('the document "{path}" metadata includes aliases "{aliases}"'))
def _then_metadata_aliases(ctx: ScenarioContext, path: str, aliases: str) -> None:
    """Assert document metadata contains the expected aliases."""

    result = _require_result(ctx)
    assert ctx.session is not None
    repo = DocumentRepository(ctx.session)
    doc = repo.get_by_path(result.dataset_id, path)
    assert doc is not None
    assert doc.metadata_json is not None
    metadata = json.loads(doc.metadata_json)
    expected = [a.strip() for a in aliases.split(",") if a.strip()]
    for alias in expected:
        assert alias in metadata.get("aliases", [])


@then(parse('the document "{path}" has no tags or aliases in metadata'))
def _then_plain_no_metadata(ctx: ScenarioContext, path: str) -> None:
    """Assert a document has empty/minimal metadata (no tags/aliases)."""

    result = _require_result(ctx)
    assert ctx.session is not None
    repo = DocumentRepository(ctx.session)
    doc = repo.get_by_path(result.dataset_id, path)
    assert doc is not None
    if doc.metadata_json:
        metadata = json.loads(doc.metadata_json)
        assert not metadata.get("tags")
        assert not metadata.get("aliases")


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


@then(parse('a ValueError is raised containing "{message}"'))
def _then_value_error(ctx: ScenarioContext, message: str) -> None:
    """Assert a captured error is a ValueError containing a message fragment."""

    assert ctx.errors, "Expected an error but none was captured."
    err = ctx.errors[0]
    assert isinstance(err, ValueError)
    assert message in str(err)


def _require_result(ctx: ScenarioContext) -> IngestResult:
    if ctx.last_result is None:
        raise AssertionError("No ingest result is available in the scenario context.")
    return ctx.last_result
