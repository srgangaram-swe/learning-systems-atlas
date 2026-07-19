"""Label-free selection and representation diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.validation import FloatArray, validate_features

IntArray = NDArray[np.int64]


def pairwise_squared_distances(features: ArrayLike) -> FloatArray:
    """Return a stable non-negative squared Euclidean distance matrix."""

    validated = validate_features(features, min_samples=2)
    norms = np.sum(validated * validated, axis=1, keepdims=True)
    distances = norms + norms.T - 2.0 * validated @ validated.T
    return np.asarray(np.maximum(distances, 0.0), dtype=np.float64)


def neighborhood_preservation(
    original: ArrayLike,
    embedding: ArrayLike,
    *,
    n_neighbors: int = 10,
) -> float:
    """Measure mean k-nearest-neighbor overlap without consulting targets."""

    source = validate_features(original, min_samples=3)
    reduced = validate_features(embedding, min_samples=3)
    if source.shape[0] != reduced.shape[0]:
        msg = "original and embedding must contain the same number of samples"
        raise ValueError(msg)
    if isinstance(n_neighbors, bool) or not 1 <= n_neighbors < len(source):
        msg = "n_neighbors must be an integer in [1, n_samples)"
        raise ValueError(msg)

    source_distances = pairwise_squared_distances(source)
    reduced_distances = pairwise_squared_distances(reduced)
    source_order = np.argsort(source_distances, axis=1, kind="stable")[:, 1 : n_neighbors + 1]
    reduced_order = np.argsort(reduced_distances, axis=1, kind="stable")[:, 1 : n_neighbors + 1]
    overlaps = [
        len(set(source_neighbors).intersection(reduced_neighbors)) / n_neighbors
        for source_neighbors, reduced_neighbors in zip(source_order, reduced_order, strict=True)
    ]
    return float(np.mean(overlaps))


def distance_correlation(original: ArrayLike, embedding: ArrayLike) -> float:
    """Return Pearson correlation between upper-triangle pairwise distances."""

    source = validate_features(original, min_samples=3)
    reduced = validate_features(embedding, min_samples=3)
    if source.shape[0] != reduced.shape[0]:
        msg = "original and embedding must contain the same number of samples"
        raise ValueError(msg)
    indices = np.triu_indices(len(source), k=1)
    source_distances = np.sqrt(pairwise_squared_distances(source)[indices])
    reduced_distances = np.sqrt(pairwise_squared_distances(reduced)[indices])
    source_centered = source_distances - np.mean(source_distances)
    reduced_centered = reduced_distances - np.mean(reduced_distances)
    denominator = float(np.linalg.norm(source_centered) * np.linalg.norm(reduced_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(source_centered, reduced_centered) / denominator, -1.0, 1.0))


def clustering_selection_score(*, silhouette: float, coverage: float, stability: float) -> float:
    """Combine feature-only geometry, assignment coverage, and perturbation stability."""

    values = (silhouette, coverage, stability)
    if not all(np.isfinite(value) for value in values):
        msg = "selection inputs must be finite"
        raise ValueError(msg)
    if not -1.0 <= silhouette <= 1.0:
        msg = "silhouette must be in [-1, 1]"
        raise ValueError(msg)
    if not 0.0 <= coverage <= 1.0:
        msg = "coverage must be in [0, 1]"
        raise ValueError(msg)
    if not -1.0 <= stability <= 1.0:
        msg = "stability must be in [-1, 1]"
        raise ValueError(msg)
    geometric = max(silhouette, 0.0) * coverage
    normalized_stability = (stability + 1.0) / 2.0
    return float(0.65 * geometric + 0.35 * normalized_stability)


def select_highest(scores: Mapping[str, float]) -> str:
    """Select the highest finite score with an explicit alphabetical tie-break."""

    if not scores:
        msg = "selection requires at least one candidate"
        raise ValueError(msg)
    if any(not name or not np.isfinite(score) for name, score in scores.items()):
        msg = "candidate names must be non-empty and scores finite"
        raise ValueError(msg)
    return min(scores, key=lambda name: (-scores[name], name))
