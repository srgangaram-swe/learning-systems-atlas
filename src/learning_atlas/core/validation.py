"""Centralized numerical input validation for supervised estimators."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _numeric_array(values: ArrayLike, *, name: str) -> FloatArray:
    """Convert a rectangular numeric input without silently accepting object/string data."""

    try:
        array = np.asarray(values)
    except ValueError as error:
        msg = f"{name} must be a rectangular numeric array"
        raise ValueError(msg) from error
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        msg = f"{name} must contain real numeric values"
        raise TypeError(msg)
    converted = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(converted)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return converted


def validate_features(
    features: ArrayLike,
    *,
    expected_features: int | None = None,
    min_samples: int = 1,
) -> FloatArray:
    """Return a finite 2D float array with optional fitted-shape enforcement."""

    if min_samples < 1:
        msg = "min_samples must be positive"
        raise ValueError(msg)
    validated = _numeric_array(features, name="features")
    if validated.ndim != 2:
        msg = f"features must be 2D; received {validated.ndim}D"
        raise ValueError(msg)
    if validated.shape[0] < min_samples:
        msg = f"features must contain at least {min_samples} sample(s)"
        raise ValueError(msg)
    if validated.shape[1] == 0:
        msg = "features must contain at least one column"
        raise ValueError(msg)
    if expected_features is not None and validated.shape[1] != expected_features:
        msg = (
            f"features contain {validated.shape[1]} columns; "
            f"fitted estimator expects {expected_features}"
        )
        raise ValueError(msg)
    return validated


def validate_targets(targets: ArrayLike, *, n_samples: int) -> FloatArray:
    """Return a finite 1D numeric target aligned with the feature row count."""

    validated = _numeric_array(targets, name="targets")
    if validated.ndim != 1:
        msg = f"targets must be 1D; received {validated.ndim}D"
        raise ValueError(msg)
    if len(validated) != n_samples:
        msg = f"features and targets have inconsistent samples: {n_samples} != {len(validated)}"
        raise ValueError(msg)
    return validated


def validate_X_y(features: ArrayLike, targets: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Validate a supervised training pair with one shared sample boundary."""

    validated_features = validate_features(features)
    validated_targets = validate_targets(targets, n_samples=len(validated_features))
    return validated_features, validated_targets


def validate_probability(probability: float, *, name: str) -> float:
    """Validate a scalar probability hyperparameter."""

    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        msg = f"{name} must be a finite value in [0, 1]"
        raise ValueError(msg)
    return float(probability)


def validate_choices(value: str, *, name: str, choices: Sequence[str]) -> str:
    """Validate a string option while producing a stable, actionable error."""

    if value not in choices:
        expected = ", ".join(sorted(choices))
        msg = f"{name} must be one of: {expected}"
        raise ValueError(msg)
    return value
