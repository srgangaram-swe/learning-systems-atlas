"""Clarity-first agglomerative clustering with a SciPy-compatible merge history."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from learning_atlas.core.validation import FloatArray, validate_features
from learning_atlas.unsupervised.base import IntArray, UnsupervisedModel
from learning_atlas.unsupervised.metrics import (
    DEFAULT_MAX_PAIRWISE_ELEMENTS,
    pairwise_euclidean_distances,
)

Linkage = Literal["single", "complete", "average", "ward"]


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return int(value)


def _linkage_name(value: str) -> Linkage:
    choices = {"single", "complete", "average", "ward"}
    if value not in choices:
        expected = ", ".join(sorted(choices))
        msg = f"linkage must be one of: {expected}"
        raise ValueError(msg)
    return cast(Linkage, value)


@dataclass(frozen=True, slots=True)
class _Cluster:
    members: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.members)


def _validate_linkage_matrix(linkage_matrix: ArrayLike) -> tuple[FloatArray, int]:
    try:
        raw = np.asarray(linkage_matrix)
    except ValueError as error:
        msg = "linkage_matrix must be a rectangular numeric array"
        raise ValueError(msg) from error
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        msg = "linkage_matrix must contain real numeric values"
        raise TypeError(msg)
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1:] != (4,):
        msg = f"linkage_matrix must have shape (n_samples - 1, 4); received {matrix.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(matrix)):
        msg = "linkage_matrix must contain only finite values"
        raise ValueError(msg)
    if np.any(matrix[:, 2] < 0.0):
        msg = "linkage distances must be non-negative"
        raise ValueError(msg)
    if len(matrix) > 1:
        tolerance = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(matrix[:-1, 2])) * 32.0
        if np.any(matrix[1:, 2] < matrix[:-1, 2] - tolerance):
            msg = "linkage distances must be nondecreasing"
            raise ValueError(msg)

    n_samples = len(matrix) + 1
    active: dict[int, tuple[int, ...]] = {index: (index,) for index in range(n_samples)}
    for step, row in enumerate(matrix):
        left_raw, right_raw, _, size_raw = row
        if left_raw != math.trunc(left_raw) or right_raw != math.trunc(right_raw):
            msg = f"linkage row {step} child identifiers must be integers"
            raise ValueError(msg)
        left = int(left_raw)
        right = int(right_raw)
        if left == right or left not in active or right not in active:
            msg = f"linkage row {step} must reference two distinct active clusters"
            raise ValueError(msg)
        if size_raw != math.trunc(size_raw):
            msg = f"linkage row {step} cluster size must be an integer"
            raise ValueError(msg)
        merged = (*active.pop(left), *active.pop(right))
        if int(size_raw) != len(merged):
            msg = f"linkage row {step} reports an incorrect cluster size"
            raise ValueError(msg)
        active[n_samples + step] = merged
    if len(active) != 1:
        msg = "linkage_matrix does not form one complete hierarchy"
        raise ValueError(msg)
    return matrix, n_samples


def cut_linkage(linkage_matrix: ArrayLike, n_clusters: int) -> IntArray:
    """Cut a validated full linkage tree into exactly ``n_clusters`` groups."""

    matrix, n_samples = _validate_linkage_matrix(linkage_matrix)
    requested = _positive_integer(n_clusters, name="n_clusters")
    if requested > n_samples:
        msg = f"n_clusters must not exceed the fitted sample count: {requested} > {n_samples}"
        raise ValueError(msg)

    active: dict[int, tuple[int, ...]] = {index: (index,) for index in range(n_samples)}
    merges_to_apply = n_samples - requested
    for step, row in enumerate(matrix[:merges_to_apply]):
        left = int(row[0])
        right = int(row[1])
        merged = (*active.pop(left), *active.pop(right))
        active[n_samples + step] = merged

    labels = np.empty(n_samples, dtype=np.int64)
    ordered_clusters = sorted(active.values(), key=lambda members: min(members))
    for label, members in enumerate(ordered_clusters):
        labels[np.asarray(members, dtype=np.int64)] = label
    labels.setflags(write=False)
    return labels


class AgglomerativeClustering(UnsupervisedModel):
    """Bottom-up Euclidean clustering with deterministic merge ties.

    This inspectable implementation recomputes cluster distances directly and
    intentionally favors transparent definitions over production performance.
    It allocates a quadratic sample-distance matrix and can require cubic time.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        *,
        linkage: str = "ward",
        metric: str = "euclidean",
        max_pairwise_elements: int = DEFAULT_MAX_PAIRWISE_ELEMENTS,
    ) -> None:
        super().__init__()
        self.n_clusters = _positive_integer(n_clusters, name="n_clusters")
        self.linkage = _linkage_name(linkage)
        if metric != "euclidean":
            detail = "Ward linkage requires Euclidean geometry" if linkage == "ward" else ""
            msg = f"metric must be 'euclidean'. {detail}".strip()
            raise ValueError(msg)
        self.metric = metric
        self.max_pairwise_elements = _positive_integer(
            max_pairwise_elements,
            name="max_pairwise_elements",
        )
        self._labels: IntArray | None = None
        self._linkage_matrix: FloatArray | None = None
        self._children: IntArray | None = None
        self._distances: FloatArray | None = None
        self._cluster_sizes: IntArray | None = None
        self.n_clusters_: int | None = None

    @property
    def labels_(self) -> IntArray:
        """Assignments from cutting the full tree at configured ``n_clusters``."""

        self._require_fitted()
        assert self._labels is not None
        return self._labels

    @property
    def linkage_matrix_(self) -> FloatArray:
        """Float64 ``(n-1, 4)`` SciPy-compatible chronological merge history."""

        self._require_fitted()
        assert self._linkage_matrix is not None
        return self._linkage_matrix

    @property
    def children_(self) -> IntArray:
        """Integer child identifiers for every chronological merge."""

        self._require_fitted()
        assert self._children is not None
        return self._children

    @property
    def distances_(self) -> FloatArray:
        """Nondecreasing merge distances."""

        self._require_fitted()
        assert self._distances is not None
        return self._distances

    @property
    def cluster_sizes_(self) -> IntArray:
        """Observation count formed by every merge."""

        self._require_fitted()
        assert self._cluster_sizes is not None
        return self._cluster_sizes

    def _updated_distance(
        self,
        left: _Cluster,
        right: _Cluster,
        other: _Cluster,
        *,
        left_distance: float,
        right_distance: float,
        merge_distance: float,
    ) -> float:
        if self.linkage == "single":
            return min(left_distance, right_distance)
        if self.linkage == "complete":
            return max(left_distance, right_distance)
        if self.linkage == "average":
            return (left.size * left_distance + right.size * right_distance) / (
                left.size + right.size
            )

        total = left.size + right.size + other.size
        with np.errstate(over="ignore", invalid="ignore"):
            squared = (
                (other.size + left.size) * left_distance * left_distance
                + (other.size + right.size) * right_distance * right_distance
                - other.size * merge_distance * merge_distance
            ) / total
        scale = max(
            1.0,
            left_distance * left_distance,
            right_distance * right_distance,
            merge_distance * merge_distance,
        )
        tolerance = np.finfo(np.float64).eps * scale * 64.0
        if squared < -tolerance:
            msg = "Ward distance update became negative; rescale features before clustering"
            raise ValueError(msg)
        distance = math.sqrt(max(squared, 0.0))
        if not math.isfinite(distance):
            msg = "Ward distance update overflowed; rescale features before clustering"
            raise ValueError(msg)
        return distance

    def fit(self, features: ArrayLike) -> AgglomerativeClustering:
        """Build the complete bottom-up hierarchy from features only."""

        validated = validate_features(features)
        n_samples = len(validated)
        if self.n_clusters > n_samples:
            msg = (
                "n_clusters must not exceed the fitted sample count: "
                f"{self.n_clusters} > {n_samples}"
            )
            raise ValueError(msg)
        sample_distances = pairwise_euclidean_distances(
            validated,
            max_pairwise_elements=self.max_pairwise_elements,
        )
        active: dict[int, _Cluster] = {
            index: _Cluster(members=(index,)) for index in range(n_samples)
        }
        pair_distances = {
            (left, right): float(sample_distances[left, right])
            for left in range(n_samples - 1)
            for right in range(left + 1, n_samples)
        }
        linkage_matrix = np.empty((n_samples - 1, 4), dtype=np.float64)
        previous_distance = 0.0
        for step in range(n_samples - 1):
            active_ids = sorted(active)
            distance, left_id, right_id = min(
                (pair_distances[(left, right)], left, right)
                for left_position, left in enumerate(active_ids[:-1])
                for right in active_ids[left_position + 1 :]
            )
            tolerance = np.finfo(np.float64).eps * max(1.0, abs(previous_distance)) * 32.0
            if distance < previous_distance:
                if previous_distance - distance <= tolerance:
                    distance = previous_distance
                else:  # defensive: supported reducible linkages should not invert
                    msg = f"{self.linkage} linkage produced a decreasing merge distance"
                    raise RuntimeError(msg)
            left = active[left_id]
            right = active[right_id]
            members = (*left.members, *right.members)
            size = len(members)
            new_id = n_samples + step
            updated_distances = {
                other_id: self._updated_distance(
                    left,
                    right,
                    active[other_id],
                    left_distance=pair_distances[_pair_key(left_id, other_id)],
                    right_distance=pair_distances[_pair_key(right_id, other_id)],
                    merge_distance=distance,
                )
                for other_id in active_ids
                if other_id not in {left_id, right_id}
            }
            active.pop(left_id)
            active.pop(right_id)
            active[new_id] = _Cluster(members=members)
            pair_distances = {
                pair: value
                for pair, value in pair_distances.items()
                if left_id not in pair and right_id not in pair
            }
            pair_distances.update(
                {
                    _pair_key(other_id, new_id): updated_distance
                    for other_id, updated_distance in updated_distances.items()
                }
            )
            linkage_matrix[step] = (left_id, right_id, distance, size)
            previous_distance = distance

        labels = cut_linkage(linkage_matrix, self.n_clusters)
        children = np.asarray(linkage_matrix[:, :2], dtype=np.int64)
        distances = np.asarray(linkage_matrix[:, 2], dtype=np.float64)
        cluster_sizes = np.asarray(linkage_matrix[:, 3], dtype=np.int64)
        for array in (linkage_matrix, children, distances, cluster_sizes):
            array.setflags(write=False)

        self._labels = labels
        self._linkage_matrix = linkage_matrix
        self._children = children
        self._distances = distances
        self._cluster_sizes = cluster_sizes
        self.n_clusters_ = self.n_clusters
        self.n_features_in_ = validated.shape[1]
        return self

    def fit_predict(self, features: ArrayLike) -> IntArray:
        """Fit the hierarchy and return configured cut assignments."""

        return self.fit(features).labels_

    def cut(self, n_clusters: int) -> IntArray:
        """Extract another exact-k partition without refitting the hierarchy."""

        return cut_linkage(self.linkage_matrix_, n_clusters)
