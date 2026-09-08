"""Numerical, seed isolation and oracle-divergence invariants."""

from dataclasses import asdict

import numpy as np
import pytest

from learning_atlas.reporting.parity import (
    ParityPair,
    measure_pair,
    measure_reference_parity,
    prediction_error,
    reference_pairs,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "left,right", [([], []), ([[1]], [[1]]), ([1], [1, 2]), ([np.nan], [1]), ([1], [np.inf])]
)
def test_prediction_comparison_rejects_nonfinite_or_broadcasting(left, right):
    with pytest.raises(ValueError):
        prediction_error(left, right, 1e-6)


@pytest.mark.parametrize("tolerance", [0, -1, True, np.inf, np.nan])
def test_prediction_comparison_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        prediction_error([1], [1], tolerance)


def test_comparison_retains_disagreement_instead_of_clamping():
    assert prediction_error([1, 2], [1, 2], 1e-6) == (0, True)
    assert prediction_error([1, 2], [1, 4], 1e-6) == (2, False)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.2])
def test_seed_validation(seed):
    with pytest.raises(ValueError, match="seed"):
        measure_pair(reference_pairs()[0], seed)


def test_all_fixed_pairs_retain_train_test_baselines_and_finite_timings():
    rows = measure_reference_parity()
    assert len(rows) == 27
    assert {row.seed for row in rows} == {17, 29, 43}
    for row in rows:
        assert row.max_training_prediction_error <= row.tolerance
        if not row.family.startswith("CART"):
            assert row.passed
        assert all(np.isfinite(value) for value in asdict(row).values() if isinstance(value, float))
        assert row.scratch_fit_seconds > 0
        assert row.oracle_fit_seconds > 0


def test_repeatability_and_candidate_order_isolation_exclude_only_timings():
    def stable(pair):
        return {
            key: value
            for key, value in asdict(measure_pair(pair, 17)).items()
            if not key.endswith("seconds")
        }

    pairs = reference_pairs()[:3]
    forward = {pair.name: stable(pair) for pair in pairs}
    reverse = {pair.name: stable(pair) for pair in reversed(pairs)}
    assert forward == reverse


def test_estimator_failure_retains_seed_context_and_original_cause():
    class Broken:
        def fit(self, X, y):
            raise ValueError("controlled fixture")

        def predict(self, X):
            raise AssertionError("must not predict after failed fit")

    pair = ParityPair("broken", "regression", lambda: (Broken(), Broken()), 1e-8)
    with pytest.raises(RuntimeError, match="broken, seed=17") as raised:
        measure_pair(pair, 17)
    assert isinstance(raised.value.__cause__, ValueError)
