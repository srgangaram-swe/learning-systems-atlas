"""Exhaustive numerical validation-path tests."""

import numpy as np
import pytest

from learning_atlas.core.validation import (
    validate_choices,
    validate_features,
    validate_probability,
    validate_targets,
    validate_X_y,
)

pytestmark = pytest.mark.unit


def test_valid_supervised_pair_is_float64_and_shape_preserving() -> None:
    features, targets = validate_X_y([[1, 2], [3, 4]], [0, 1])
    assert features.dtype == np.float64
    assert targets.dtype == np.float64
    assert features.shape == (2, 2)
    assert targets.shape == (2,)


@pytest.mark.parametrize(
    ("features", "exception", "message"),
    [
        ([[1, 2], [3]], ValueError, "rectangular"),
        ([["one"]], TypeError, "real numeric"),
        ([[1 + 2j]], TypeError, "real numeric"),
        ([[np.nan]], ValueError, "finite"),
        ([[np.inf]], ValueError, "finite"),
        ([1, 2], ValueError, "2D"),
        (np.empty((0, 2)), ValueError, "at least 1 sample"),
        (np.empty((2, 0)), ValueError, "at least one column"),
    ],
)
def test_invalid_feature_inputs_are_actionable(
    features: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        validate_features(features)


def test_feature_count_and_minimum_sample_contracts() -> None:
    with pytest.raises(ValueError, match="min_samples must be positive"):
        validate_features([[1.0]], min_samples=0)
    with pytest.raises(ValueError, match="at least 3 sample"):
        validate_features([[1.0], [2.0]], min_samples=3)
    with pytest.raises(ValueError, match="fitted estimator expects 2"):
        validate_features([[1.0]], expected_features=2)
    assert validate_features([[1.0]], expected_features=1).shape == (1, 1)


@pytest.mark.parametrize(
    ("targets", "n_samples", "exception", "message"),
    [
        ([[1.0]], 1, ValueError, "1D"),
        ([1.0], 2, ValueError, "inconsistent"),
        ([np.nan], 1, ValueError, "finite"),
        (["class"], 1, TypeError, "real numeric"),
    ],
)
def test_invalid_targets_are_rejected(
    targets: object,
    n_samples: int,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        validate_targets(targets, n_samples=n_samples)


@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
def test_probability_validation_accepts_closed_interval(value: float) -> None:
    assert validate_probability(value, name="rate") == value


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_probability_validation_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_probability(value, name="rate")


def test_choice_validation_lists_supported_values() -> None:
    assert validate_choices("svd", name="solver", choices=("svd", "cholesky")) == "svd"
    with pytest.raises(ValueError, match="cholesky, svd"):
        validate_choices("magic", name="solver", choices=("svd", "cholesky"))
