"""Reproducibility contracts for the supervised comparison harness."""

import pytest

from learning_atlas.supervised.baselines import MeanRegressor
from learning_atlas.supervised.comparison import (
    _candidate_seed_plan,
    _CandidateSpec,
    _named_seed,
)

pytestmark = pytest.mark.unit


def test_named_seed_streams_are_distinct_and_candidate_order_invariant() -> None:
    specs = (
        _CandidateSpec("alpha", lambda _seed: MeanRegressor(), {}),
        _CandidateSpec("beta", lambda _seed: MeanRegressor(), {}),
    )

    forward = _candidate_seed_plan(42, "fixture", specs, fold_count=3)
    reversed_plan = _candidate_seed_plan(42, "fixture", tuple(reversed(specs)), fold_count=3)

    assert forward == reversed_plan
    assert forward["alpha"] != forward["beta"]
    all_model_seeds = {
        seed for candidate in forward.values() for seed in (*candidate.folds, candidate.final)
    }
    component_seeds = {
        _named_seed(42, "fixture", namespace) for namespace in ("data", "split", "folds")
    }
    assert len(all_model_seeds) == 8
    assert len(component_seeds) == 3
    assert all_model_seeds.isdisjoint(component_seeds)
