"""Numerically explicit metrics and partition-stability utilities for clustering."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.validation import FloatArray, validate_features

IntArray = NDArray[np.int64]

DEFAULT_MAX_PAIRWISE_ELEMENTS = 4_000_000


@dataclass(frozen=True, slots=True)
class SilhouetteResult:
    """Finite silhouette evidence, including whether the score is mathematically defined."""

    score: float
    sample_scores: FloatArray
    coverage: float
    n_clusters: int
    n_samples: int
    defined: bool


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return int(value)


def _validate_noise_label(noise_label: int) -> int:
    if isinstance(noise_label, bool) or not isinstance(noise_label, (int, np.integer)):
        msg = "noise_label must be an integer"
        raise TypeError(msg)
    return int(noise_label)


def _validate_labels(labels: ArrayLike, *, name: str) -> IntArray:
    """Return non-empty, finite integer labels without accepting lossy coercions."""

    try:
        raw = np.asarray(labels)
    except ValueError as error:
        msg = f"{name} must be a one-dimensional integer array"
        raise ValueError(msg) from error
    if raw.ndim != 1:
        msg = f"{name} must be 1D; received {raw.ndim}D"
        raise ValueError(msg)
    if len(raw) == 0:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if np.issubdtype(raw.dtype, np.bool_):
        msg = f"{name} must contain integer cluster identifiers, not booleans"
        raise TypeError(msg)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        msg = f"{name} must contain integer cluster identifiers"
        raise TypeError(msg)

    if np.issubdtype(raw.dtype, np.integer):
        minimum = int(np.min(raw))
        maximum = int(np.max(raw))
        bounds = np.iinfo(np.int64)
        if minimum < bounds.min or maximum > bounds.max:
            msg = f"{name} values must fit in signed 64-bit integers"
            raise ValueError(msg)
        return np.asarray(raw, dtype=np.int64)

    numeric = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if not np.all(numeric == np.trunc(numeric)):
        msg = f"{name} must contain integer-valued cluster identifiers"
        raise ValueError(msg)
    bounds = np.iinfo(np.int64)
    if np.any(numeric < bounds.min) or np.any(numeric > bounds.max):
        msg = f"{name} values must fit in signed 64-bit integers"
        raise ValueError(msg)
    return numeric.astype(np.int64)


def _validate_label_pair(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    *,
    ignore_noise: bool,
    noise_label: int,
) -> tuple[IntArray, IntArray]:
    observed = _validate_labels(labels_true, name="labels_true")
    predicted = _validate_labels(labels_pred, name="labels_pred")
    if len(observed) != len(predicted):
        msg = f"label arrays have inconsistent samples: {len(observed)} != {len(predicted)}"
        raise ValueError(msg)
    sentinel = _validate_noise_label(noise_label)
    if ignore_noise:
        retained = (observed != sentinel) & (predicted != sentinel)
        observed = observed[retained]
        predicted = predicted[retained]
        if len(observed) == 0:
            msg = "no samples remain after excluding noise labels"
            raise ValueError(msg)
    return observed, predicted


def _validate_pairwise_budget(n_samples: int, max_pairwise_elements: int) -> None:
    budget = _validate_positive_integer(
        max_pairwise_elements,
        name="max_pairwise_elements",
    )
    required = n_samples * n_samples
    if required > budget:
        msg = (
            f"pairwise distance matrix requires {required} elements for {n_samples} samples; "
            f"max_pairwise_elements={budget}. Reduce the sample count or raise the explicit limit"
        )
        raise MemoryError(msg)


def pairwise_euclidean_distances(
    features: ArrayLike,
    *,
    max_pairwise_elements: int = DEFAULT_MAX_PAIRWISE_ELEMENTS,
) -> FloatArray:
    """Compute a finite Euclidean distance matrix under an explicit quadratic budget."""

    validated = validate_features(features)
    n_samples = len(validated)
    _validate_pairwise_budget(n_samples, max_pairwise_elements)
    distances = np.zeros((n_samples, n_samples), dtype=np.float64)
    for row in range(n_samples - 1):
        with np.errstate(over="ignore", invalid="ignore"):
            differences = validated[row + 1 :] - validated[row]
        if not np.all(np.isfinite(differences)):
            msg = "pairwise differences overflowed; rescale features before clustering"
            raise ValueError(msg)
        absolute = np.abs(differences)
        scale = np.max(absolute, axis=1)
        normalized = np.zeros_like(absolute)
        nonzero = scale > 0.0
        normalized[nonzero] = absolute[nonzero] / scale[nonzero, None]
        with np.errstate(over="ignore", invalid="ignore"):
            values = scale * np.sqrt(np.sum(normalized * normalized, axis=1))
        if not np.all(np.isfinite(values)):
            msg = "pairwise distances overflowed; rescale features before clustering"
            raise ValueError(msg)
        distances[row, row + 1 :] = values
        distances[row + 1 :, row] = values
    return distances


def silhouette_analysis(
    features: ArrayLike,
    labels: ArrayLike,
    *,
    include_noise: bool = False,
    noise_label: int = -1,
    max_pairwise_elements: int = DEFAULT_MAX_PAIRWISE_ELEMENTS,
) -> SilhouetteResult:
    """Return silhouette evidence with a finite sentinel for undefined partitions.

    Noise is excluded by default. A partition with fewer than two non-noise
    clusters, or with no more retained samples than clusters, has score ``-1``
    and ``defined=False``. Singleton clusters contribute a conventional sample
    silhouette of zero when the overall score is otherwise defined.
    """

    validated_features = validate_features(features)
    validated_labels = _validate_labels(labels, name="labels")
    if len(validated_features) != len(validated_labels):
        msg = (
            "features and labels have inconsistent samples: "
            f"{len(validated_features)} != {len(validated_labels)}"
        )
        raise ValueError(msg)
    sentinel = _validate_noise_label(noise_label)
    retained = np.ones(len(validated_labels), dtype=np.bool_)
    if not include_noise:
        retained = validated_labels != sentinel
    retained_count = int(np.sum(retained))
    coverage = retained_count / len(validated_labels)
    retained_labels = validated_labels[retained]
    cluster_values = np.unique(retained_labels)
    n_clusters = len(cluster_values)
    full_scores = np.zeros(len(validated_labels), dtype=np.float64)
    if n_clusters < 2 or retained_count <= n_clusters:
        full_scores.setflags(write=False)
        return SilhouetteResult(
            score=-1.0,
            sample_scores=full_scores,
            coverage=coverage,
            n_clusters=n_clusters,
            n_samples=retained_count,
            defined=False,
        )

    retained_features = validated_features[retained]
    distances = pairwise_euclidean_distances(
        retained_features,
        max_pairwise_elements=max_pairwise_elements,
    )
    scores = np.zeros(retained_count, dtype=np.float64)
    for index, cluster in enumerate(retained_labels):
        same_cluster = retained_labels == cluster
        same_cluster[index] = False
        same_count = int(np.sum(same_cluster))
        if same_count == 0:
            continue
        within = float(np.mean(distances[index, same_cluster]))
        nearest = math.inf
        for other_cluster in cluster_values:
            if other_cluster == cluster:
                continue
            other = retained_labels == other_cluster
            nearest = min(nearest, float(np.mean(distances[index, other])))
        denominator = max(within, nearest)
        scores[index] = 0.0 if denominator == 0.0 else (nearest - within) / denominator

    full_scores[retained] = scores
    full_scores.setflags(write=False)
    score = float(np.mean(scores))
    return SilhouetteResult(
        score=max(-1.0, min(1.0, score)),
        sample_scores=full_scores,
        coverage=coverage,
        n_clusters=n_clusters,
        n_samples=retained_count,
        defined=True,
    )


def silhouette_score(
    features: ArrayLike,
    labels: ArrayLike,
    *,
    include_noise: bool = False,
    noise_label: int = -1,
    max_pairwise_elements: int = DEFAULT_MAX_PAIRWISE_ELEMENTS,
) -> float:
    """Return the mean silhouette or the documented finite ``-1`` sentinel."""

    return silhouette_analysis(
        features,
        labels,
        include_noise=include_noise,
        noise_label=noise_label,
        max_pairwise_elements=max_pairwise_elements,
    ).score


def contingency_matrix(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    *,
    ignore_noise: bool = False,
    noise_label: int = -1,
    max_cells: int = DEFAULT_MAX_PAIRWISE_ELEMENTS,
) -> IntArray:
    """Build a stably ordered contingency table under an explicit allocation limit."""

    observed, predicted = _validate_label_pair(
        labels_true,
        labels_pred,
        ignore_noise=ignore_noise,
        noise_label=noise_label,
    )
    _, observed_inverse = np.unique(observed, return_inverse=True)
    _, predicted_inverse = np.unique(predicted, return_inverse=True)
    rows = int(np.max(observed_inverse)) + 1
    columns = int(np.max(predicted_inverse)) + 1
    cells = rows * columns
    budget = _validate_positive_integer(max_cells, name="max_cells")
    if cells > budget:
        msg = f"contingency matrix requires {cells} cells; max_cells={budget}"
        raise MemoryError(msg)
    table = np.zeros((rows, columns), dtype=np.int64)
    np.add.at(table, (observed_inverse, predicted_inverse), 1)
    return table


def _partition_counts(
    observed: IntArray,
    predicted: IntArray,
) -> tuple[Counter[int], Counter[int], Counter[tuple[int, int]]]:
    rows = Counter(int(value) for value in observed)
    columns = Counter(int(value) for value in predicted)
    cells = Counter(
        (int(observed_value), int(predicted_value))
        for observed_value, predicted_value in zip(observed, predicted, strict=True)
    )
    return rows, columns, cells


def _pairs(count: int) -> int:
    return count * (count - 1) // 2


def adjusted_rand_score(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    *,
    ignore_noise: bool = False,
    noise_label: int = -1,
) -> float:
    """Return the chance-adjusted Rand index using overflow-safe integer counts."""

    observed, predicted = _validate_label_pair(
        labels_true,
        labels_pred,
        ignore_noise=ignore_noise,
        noise_label=noise_label,
    )
    if len(observed) < 2:
        return 1.0
    rows, columns, cells = _partition_counts(observed, predicted)
    total_pairs = _pairs(len(observed))
    cell_pairs = sum(_pairs(count) for count in cells.values())
    row_pairs = sum(_pairs(count) for count in rows.values())
    column_pairs = sum(_pairs(count) for count in columns.values())

    numerator = cell_pairs * total_pairs - row_pairs * column_pairs
    denominator_twice = (row_pairs + column_pairs) * total_pairs - 2 * row_pairs * column_pairs
    if denominator_twice == 0:
        return 1.0
    score = 2.0 * numerator / denominator_twice
    return max(-1.0, min(1.0, float(score)))


def normalized_mutual_info_score(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    *,
    ignore_noise: bool = False,
    noise_label: int = -1,
) -> float:
    """Return arithmetic-mean normalized mutual information in natural-log units."""

    observed, predicted = _validate_label_pair(
        labels_true,
        labels_pred,
        ignore_noise=ignore_noise,
        noise_label=noise_label,
    )
    rows, columns, cells = _partition_counts(observed, predicted)
    if len(cells) == len(rows) == len(columns):
        # Every observed cluster intersects exactly one predicted cluster and
        # vice versa, so the partitions are identical up to label names.
        return 1.0
    sample_count = len(observed)
    mutual_information = 0.0
    for (row, column), count in cells.items():
        probability = count / sample_count
        mutual_information += probability * math.log(
            count * sample_count / (rows[row] * columns[column])
        )
    row_entropy = -sum(
        (count / sample_count) * math.log(count / sample_count) for count in rows.values()
    )
    column_entropy = -sum(
        (count / sample_count) * math.log(count / sample_count) for count in columns.values()
    )
    denominator = 0.5 * (row_entropy + column_entropy)
    if denominator == 0.0:
        return 1.0
    score = mutual_information / denominator
    return max(0.0, min(1.0, float(score)))


def pairwise_partition_stability(
    labelings: Sequence[ArrayLike],
    *,
    ignore_noise: bool = False,
    noise_label: int = -1,
) -> FloatArray:
    """Return the symmetric matrix of ARI agreement across repeated partitions."""

    if not labelings:
        msg = "labelings must contain at least one partition"
        raise ValueError(msg)
    validated = tuple(
        _validate_labels(labels, name=f"labelings[{index}]")
        for index, labels in enumerate(labelings)
    )
    expected = len(validated[0])
    if any(len(labels) != expected for labels in validated[1:]):
        msg = "all stability partitions must contain the same number of samples"
        raise ValueError(msg)
    matrix = np.eye(len(validated), dtype=np.float64)
    for row in range(len(validated)):
        for column in range(row + 1, len(validated)):
            score = adjusted_rand_score(
                validated[row],
                validated[column],
                ignore_noise=ignore_noise,
                noise_label=noise_label,
            )
            matrix[row, column] = score
            matrix[column, row] = score
    return matrix


def mean_partition_stability(
    labelings: Sequence[ArrayLike],
    *,
    ignore_noise: bool = False,
    noise_label: int = -1,
) -> float:
    """Return mean pairwise ARI across at least two repeated partitions."""

    if len(labelings) < 2:
        msg = "mean partition stability requires at least two partitions"
        raise ValueError(msg)
    scores = pairwise_partition_stability(
        labelings,
        ignore_noise=ignore_noise,
        noise_label=noise_label,
    )
    rows, columns = np.triu_indices(len(labelings), k=1)
    return float(np.mean(scores[rows, columns]))
