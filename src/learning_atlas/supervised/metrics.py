"""NumPy-only regression and classification metrics.

The functions in this module deliberately keep validation and edge-case behavior
explicit.  They are suitable both for the from-scratch estimators and for
differential tests against independent metric implementations.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.validation import FloatArray, validate_targets

IntArray = NDArray[np.int64]


def _paired_targets(observed: ArrayLike, predicted: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Validate two finite, one-dimensional arrays with a shared sample boundary."""

    observed_array = np.asarray(observed)
    if observed_array.ndim != 1:
        msg = f"observed targets must be 1D; received {observed_array.ndim}D"
        raise ValueError(msg)
    observed_validated = validate_targets(observed, n_samples=len(observed_array))
    predicted_validated = validate_targets(predicted, n_samples=len(observed_validated))
    if len(observed_validated) == 0:
        msg = "metrics require at least one sample"
        raise ValueError(msg)
    return observed_validated, predicted_validated


def mean_squared_error(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Return the arithmetic mean of squared residuals."""

    observed_validated, predicted_validated = _paired_targets(observed, predicted)
    residuals = observed_validated - predicted_validated
    return float(np.mean(residuals * residuals))


def root_mean_squared_error(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Return the non-negative square root of mean squared error."""

    return float(np.sqrt(mean_squared_error(observed, predicted)))


def mean_absolute_error(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Return the arithmetic mean of absolute residuals."""

    observed_validated, predicted_validated = _paired_targets(observed, predicted)
    return float(np.mean(np.abs(observed_validated - predicted_validated)))


def r2_score(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Return R² with a finite convention for constant observed targets.

    A perfect constant prediction receives 1.0 and every imperfect prediction of
    a constant target receives 0.0, matching the project's estimator contract.
    """

    observed_validated, predicted_validated = _paired_targets(observed, predicted)
    residual_sum = float(np.sum((observed_validated - predicted_validated) ** 2))
    centered = observed_validated - np.mean(observed_validated)
    total_sum = float(np.sum(centered * centered))
    if total_sum == 0.0:
        return 1.0 if residual_sum == 0.0 else 0.0
    return 1.0 - residual_sum / total_sum


def accuracy_score(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Return the fraction of exactly matching class labels."""

    observed_validated, predicted_validated = _paired_targets(observed, predicted)
    return float(np.mean(observed_validated == predicted_validated))


def _classification_labels(
    observed: FloatArray,
    predicted: FloatArray,
    labels: ArrayLike | None,
) -> FloatArray:
    if labels is None:
        return np.unique(np.concatenate((observed, predicted)))

    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        msg = f"labels must be 1D; received {labels_array.ndim}D"
        raise ValueError(msg)
    validated = validate_targets(labels, n_samples=len(labels_array))
    if len(validated) == 0:
        msg = "labels must contain at least one class"
        raise ValueError(msg)
    if len(np.unique(validated)) != len(validated):
        msg = "labels must not contain duplicates"
        raise ValueError(msg)
    unknown = np.setdiff1d(np.unique(np.concatenate((observed, predicted))), validated)
    if len(unknown) > 0:
        msg = f"observed or predicted targets contain labels absent from labels: {unknown.tolist()}"
        raise ValueError(msg)
    return validated


def confusion_matrix(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    labels: ArrayLike | None = None,
) -> IntArray:
    """Count observed labels by row and predicted labels by column."""

    observed_validated, predicted_validated = _paired_targets(observed, predicted)
    ordered_labels = _classification_labels(observed_validated, predicted_validated, labels)
    label_to_index = {float(label): index for index, label in enumerate(ordered_labels)}
    matrix = np.zeros((len(ordered_labels), len(ordered_labels)), dtype=np.int64)
    for observed_label, predicted_label in zip(
        observed_validated, predicted_validated, strict=True
    ):
        matrix[label_to_index[float(observed_label)], label_to_index[float(predicted_label)]] += 1
    return matrix


def _macro_statistic(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    statistic: Literal["precision", "recall", "f1"],
    labels: ArrayLike | None,
    zero_division: float,
) -> float:
    if not np.isfinite(zero_division) or not 0.0 <= zero_division <= 1.0:
        msg = "zero_division must be a finite value in [0, 1]"
        raise ValueError(msg)
    observed_validated, predicted_validated = _paired_targets(observed, predicted)
    ordered_labels = _classification_labels(observed_validated, predicted_validated, labels)
    matrix = confusion_matrix(observed_validated, predicted_validated, labels=ordered_labels)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_positive = np.sum(matrix, axis=0, dtype=np.float64)
    actual_positive = np.sum(matrix, axis=1, dtype=np.float64)
    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.full_like(true_positive, zero_division),
        where=predicted_positive != 0.0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.full_like(true_positive, zero_division),
        where=actual_positive != 0.0,
    )
    if statistic == "precision":
        values = precision
    elif statistic == "recall":
        values = recall
    else:
        denominator = precision + recall
        values = np.divide(
            2.0 * precision * recall,
            denominator,
            out=np.full_like(precision, zero_division),
            where=denominator != 0.0,
        )
    return float(np.mean(values))


def precision_score(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    average: Literal["macro"] = "macro",
    labels: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float:
    """Return unweighted mean per-class precision."""

    if average != "macro":
        msg = "average must be 'macro'"
        raise ValueError(msg)
    return _macro_statistic(
        observed,
        predicted,
        statistic="precision",
        labels=labels,
        zero_division=zero_division,
    )


def recall_score(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    average: Literal["macro"] = "macro",
    labels: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float:
    """Return unweighted mean per-class recall."""

    if average != "macro":
        msg = "average must be 'macro'"
        raise ValueError(msg)
    return _macro_statistic(
        observed,
        predicted,
        statistic="recall",
        labels=labels,
        zero_division=zero_division,
    )


def f1_score(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    average: Literal["macro"] = "macro",
    labels: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float:
    """Return unweighted mean per-class harmonic precision/recall."""

    if average != "macro":
        msg = "average must be 'macro'"
        raise ValueError(msg)
    return _macro_statistic(
        observed,
        predicted,
        statistic="f1",
        labels=labels,
        zero_division=zero_division,
    )


def log_loss(
    observed: ArrayLike,
    probabilities: ArrayLike,
    *,
    labels: ArrayLike | None = None,
    epsilon: float = 1e-15,
) -> float:
    """Return clipped binary or multiclass cross-entropy.

    One-dimensional probabilities represent the positive class in a binary
    problem.  Two-dimensional probabilities have one column per class in the
    order supplied by ``labels`` (or sorted observed labels when omitted).
    """

    observed_array = np.asarray(observed)
    if observed_array.ndim != 1:
        msg = f"observed targets must be 1D; received {observed_array.ndim}D"
        raise ValueError(msg)
    observed_validated = validate_targets(observed, n_samples=len(observed_array))
    if len(observed_validated) == 0:
        msg = "metrics require at least one sample"
        raise ValueError(msg)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        msg = "epsilon must be a finite value strictly between 0 and 0.5"
        raise ValueError(msg)

    try:
        raw_probabilities = np.asarray(probabilities)
    except ValueError as error:
        msg = "probabilities must be a rectangular numeric array"
        raise ValueError(msg) from error
    if not np.issubdtype(raw_probabilities.dtype, np.number) or np.issubdtype(
        raw_probabilities.dtype, np.complexfloating
    ):
        msg = "probabilities must contain real numeric values"
        raise TypeError(msg)
    probability_array = np.asarray(raw_probabilities, dtype=np.float64)
    if not np.all(np.isfinite(probability_array)):
        msg = "probabilities must contain only finite values"
        raise ValueError(msg)
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        msg = "probabilities must lie in [0, 1]"
        raise ValueError(msg)

    if probability_array.ndim == 1:
        if len(probability_array) != len(observed_validated):
            msg = "observed targets and probabilities have inconsistent samples"
            raise ValueError(msg)
        if labels is None:
            ordered_labels = np.unique(observed_validated)
        else:
            labels_array = np.asarray(labels)
            if labels_array.ndim != 1:
                msg = f"labels must be 1D; received {labels_array.ndim}D"
                raise ValueError(msg)
            ordered_labels = validate_targets(labels, n_samples=len(labels_array))
        if len(ordered_labels) != 2 or len(np.unique(ordered_labels)) != 2:
            msg = "one-dimensional probabilities require exactly two unique labels"
            raise ValueError(msg)
        if np.any(~np.isin(observed_validated, ordered_labels)):
            msg = "observed targets contain labels absent from labels"
            raise ValueError(msg)
        positive = (observed_validated == ordered_labels[1]).astype(np.float64)
        clipped = np.clip(probability_array, epsilon, 1.0 - epsilon)
        return float(-np.mean(positive * np.log(clipped) + (1.0 - positive) * np.log1p(-clipped)))

    if probability_array.ndim != 2:
        msg = f"probabilities must be 1D or 2D; received {probability_array.ndim}D"
        raise ValueError(msg)
    if probability_array.shape[0] != len(observed_validated):
        msg = "observed targets and probabilities have inconsistent samples"
        raise ValueError(msg)
    if probability_array.shape[1] < 2:
        msg = "probabilities must contain at least two class columns"
        raise ValueError(msg)
    if not np.allclose(np.sum(probability_array, axis=1), 1.0, rtol=1e-7, atol=1e-12):
        msg = "each probability row must sum to 1"
        raise ValueError(msg)

    if labels is None:
        ordered_labels = np.unique(observed_validated)
    else:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 1:
            msg = f"labels must be 1D; received {labels_array.ndim}D"
            raise ValueError(msg)
        ordered_labels = validate_targets(labels, n_samples=len(labels_array))
    if len(ordered_labels) != probability_array.shape[1]:
        msg = "the number of labels must equal the number of probability columns"
        raise ValueError(msg)
    if len(np.unique(ordered_labels)) != len(ordered_labels):
        msg = "labels must not contain duplicates"
        raise ValueError(msg)
    label_to_index = {float(label): index for index, label in enumerate(ordered_labels)}
    try:
        column_indices = np.asarray(
            [label_to_index[float(label)] for label in observed_validated], dtype=np.int64
        )
    except KeyError as error:
        msg = "observed targets contain labels absent from labels"
        raise ValueError(msg) from error
    selected = probability_array[np.arange(len(observed_validated)), column_indices]
    return float(-np.mean(np.log(np.clip(selected, epsilon, 1.0 - epsilon))))
