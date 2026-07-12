"""Versioned result-contract invariant tests."""

import math

import pytest
from pydantic import ValidationError

from learning_atlas.core.contracts import CandidateResult, RunResult

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_candidate_rejects_non_finite_metrics(value: float) -> None:
    with pytest.raises(ValidationError, match="metrics must be finite"):
        CandidateResult(name="bad", metrics={"score": value})


def test_result_rejects_non_finite_headline_metric(sample_result: RunResult) -> None:
    with pytest.raises(ValidationError, match="metrics must be finite"):
        sample_result.model_copy(update={"metrics": {"score": math.nan}}).model_validate(
            {**sample_result.model_dump(), "metrics": {"score": math.nan}}
        )


def test_contracts_are_frozen_and_reject_extra_fields(sample_result: RunResult) -> None:
    with pytest.raises(ValidationError, match="frozen_instance"):
        sample_result.seed = 7  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RunResult.model_validate({**sample_result.model_dump(), "surprise": True})


def test_result_round_trips_through_json(sample_result: RunResult) -> None:
    restored = RunResult.model_validate_json(sample_result.model_dump_json())
    assert restored == sample_result
    assert restored.schema_version == "1.0.0"


def test_selection_must_reference_one_unique_candidate(sample_result: RunResult) -> None:
    payload = sample_result.model_dump()
    with pytest.raises(ValidationError, match="selected_model"):
        RunResult.model_validate({**payload, "selected_model": "missing"})
    duplicate = [*payload["candidates"], payload["candidates"][0]]
    with pytest.raises(ValidationError, match="candidate names"):
        RunResult.model_validate({**payload, "candidates": duplicate})
