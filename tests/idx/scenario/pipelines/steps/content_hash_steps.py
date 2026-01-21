"""Steps for compute_content_hash scenarios."""

from __future__ import annotations

import re

from pytest_bdd import given, then, when
from pytest_bdd.parsers import re as re_parser

from idx.pipelines.ingest import compute_content_hash
from tests.idx.scenario.pipelines.conftest import ScenarioContext


@given("a content hash calculator")
def _given_hash_calculator(ctx: ScenarioContext) -> None:
    """Ensure scenario context is initialized for hashing."""

    ctx.values.pop("hash_1", None)
    ctx.values.pop("hash_2", None)


@when(re_parser(r'I compute the content hash for "(?P<content>.*)"'))
def _when_compute_hash(ctx: ScenarioContext, content: str) -> None:
    """Compute a content hash and store it as the first hash."""

    ctx.values["hash_1"] = compute_content_hash(content)


@when(re_parser(r'I compute the content hash for "(?P<content>.*)" again'))
def _when_compute_hash_again(ctx: ScenarioContext, content: str) -> None:
    """Compute a content hash and store it as the second hash."""

    ctx.values["hash_2"] = compute_content_hash(content)


@then("the hashes are equal")
def _then_hashes_equal(ctx: ScenarioContext) -> None:
    """Assert that the first and second hashes match."""

    assert ctx.values["hash_1"] == ctx.values["hash_2"]


@then("the hashes are different")
def _then_hashes_different(ctx: ScenarioContext) -> None:
    """Assert that the first and second hashes are different."""

    assert ctx.values["hash_1"] != ctx.values["hash_2"]


@then("the hash has 64 hex characters")
def _then_hash_is_sha256_hex(ctx: ScenarioContext) -> None:
    """Assert the computed hash is a 64-character lowercase hex string."""

    hash_val = ctx.values.get("hash_1") or ctx.values.get("hash_2")
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", hash_val) is not None
