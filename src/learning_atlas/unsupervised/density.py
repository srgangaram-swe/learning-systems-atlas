"""Deterministic, transductive DBSCAN implemented with inspectable NumPy operations."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.validation import FloatArray, validate_features
from learning_atlas.unsupervised.base import IntArray, UnsupervisedModel
from learning_atlas.unsupervised.metrics import (
    DEFAULT_MAX_PAIRWISE_ELEMENTS,
    pairwise_euclidean_distances,
)

StringArray = NDArray[np.str_]


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return int(value)


def _positive_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{name} must be a positive finite number"
        raise ValueError(msg)
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        msg = f"{name} must be a positive finite number"
        raise ValueError(msg)
    return converted


def _row_key(features: FloatArray, index: int) -> tuple[float, ...]:
    return tuple(float(value) for value in features[index])


class DBSCAN(UnsupervisedModel):
    """Density clustering with deterministic core components and border ties.

    A point is core when its closed ``eps`` neighborhood, including itself,
    contains at least ``min_samples`` observations. Core connected components
    define clusters. A border point adjacent to multiple components is assigned
    to the component with the closest core point; exact ties resolve by the
    component's lexicographically smallest feature row. The model is explicitly
    transductive and therefore exposes no invented out-of-sample ``predict``.
    """

    def __init__(
        self,
        *,
        eps: float = 0.5,
        min_samples: int = 5,
        max_pairwise_elements: int = DEFAULT_MAX_PAIRWISE_ELEMENTS,
    ) -> None:
        super().__init__()
        self.eps = _positive_finite(eps, name="eps")
        self.min_samples = _positive_integer(min_samples, name="min_samples")
        self.max_pairwise_elements = _positive_integer(
            max_pairwise_elements,
            name="max_pairwise_elements",
        )
        self._labels: IntArray | None = None
        self._core_sample_indices: IntArray | None = None
        self._core_sample_mask: NDArray[np.bool_] | None = None
        self._sample_roles: StringArray | None = None
        self._neighborhood_counts: IntArray | None = None
        self._k_distances: FloatArray | None = None
        self._components: FloatArray | None = None

    @property
    def labels_(self) -> IntArray:
        """Training assignments, with noise encoded as ``-1``."""

        self._require_fitted()
        assert self._labels is not None
        return self._labels

    @property
    def core_sample_indices_(self) -> IntArray:
        """Sorted row indices satisfying the core-density definition."""

        self._require_fitted()
        assert self._core_sample_indices is not None
        return self._core_sample_indices

    @property
    def core_sample_mask_(self) -> NDArray[np.bool_]:
        """Boolean core-state mask aligned with training rows."""

        self._require_fitted()
        assert self._core_sample_mask is not None
        return self._core_sample_mask

    @property
    def sample_roles_(self) -> StringArray:
        """Fixed string roles: exactly ``core``, ``border``, or ``noise``."""

        self._require_fitted()
        assert self._sample_roles is not None
        return self._sample_roles

    @property
    def neighborhood_counts_(self) -> IntArray:
        """Closed-eps neighborhood sizes, including the sample itself."""

        self._require_fitted()
        assert self._neighborhood_counts is not None
        return self._neighborhood_counts

    @property
    def k_distances_(self) -> FloatArray:
        """Distance to the requested neighbor rank, capped at the sample count."""

        self._require_fitted()
        assert self._k_distances is not None
        return self._k_distances

    @property
    def components_(self) -> FloatArray:
        """Feature rows corresponding to ``core_sample_indices_``."""

        self._require_fitted()
        assert self._components is not None
        return self._components

    def fit(self, features: ArrayLike) -> DBSCAN:
        """Discover density-connected components from features only."""

        validated = validate_features(features)
        distances = pairwise_euclidean_distances(
            validated,
            max_pairwise_elements=self.max_pairwise_elements,
        )
        neighborhoods = distances <= self.eps
        counts: IntArray = np.asarray(
            np.sum(neighborhoods, axis=1, dtype=np.int64),
            dtype=np.int64,
        )
        core_mask: NDArray[np.bool_] = np.asarray(counts >= self.min_samples, dtype=np.bool_)
        core_indices = np.flatnonzero(core_mask).astype(np.int64, copy=False)

        ordered_core = sorted(
            (int(index) for index in core_indices),
            key=lambda index: (_row_key(validated, index), index),
        )
        unseen = set(ordered_core)
        components: list[tuple[int, ...]] = []
        while unseen:
            start = min(unseen, key=lambda index: (_row_key(validated, index), index))
            unseen.remove(start)
            pending = [start]
            discovered: list[int] = []
            while pending:
                current = pending.pop()
                discovered.append(current)
                adjacent = [
                    index
                    for index in np.flatnonzero(neighborhoods[current] & core_mask)
                    if int(index) in unseen
                ]
                adjacent.sort(
                    key=lambda index: (_row_key(validated, int(index)), int(index)),
                    reverse=True,
                )
                for neighbor in adjacent:
                    integer_neighbor = int(neighbor)
                    unseen.remove(integer_neighbor)
                    pending.append(integer_neighbor)
            components.append(tuple(sorted(discovered)))

        def component_key(component: tuple[int, ...]) -> tuple[tuple[float, ...], ...]:
            return tuple(sorted(_row_key(validated, index) for index in component))

        components.sort(key=component_key)
        labels = np.full(len(validated), -1, dtype=np.int64)
        for cluster, component in enumerate(components):
            labels[np.asarray(component, dtype=np.int64)] = cluster

        component_keys = tuple(component_key(component) for component in components)
        for index in np.flatnonzero(~core_mask):
            core_neighbors = np.flatnonzero(neighborhoods[index] & core_mask)
            if len(core_neighbors) == 0:
                continue
            candidates: list[tuple[float, tuple[tuple[float, ...], ...], int]] = []
            for cluster, component in enumerate(components):
                members = np.asarray(component, dtype=np.int64)
                adjacent_members = members[neighborhoods[index, members]]
                if len(adjacent_members) == 0:
                    continue
                nearest = float(np.min(distances[index, adjacent_members]))
                candidates.append((nearest, component_keys[cluster], cluster))
            labels[index] = min(candidates)[2]

        roles = np.full(len(validated), "noise", dtype="<U6")
        roles[labels >= 0] = "border"
        roles[core_mask] = "core"
        neighbor_rank = min(self.min_samples, len(validated)) - 1
        k_distances: FloatArray = np.asarray(
            np.partition(distances, neighbor_rank, axis=1)[:, neighbor_rank],
            dtype=np.float64,
        )

        for array in (labels, core_indices, core_mask, roles, counts, k_distances):
            array.setflags(write=False)
        core_features = np.asarray(validated[core_indices], dtype=np.float64)
        core_features.setflags(write=False)

        self._labels = labels
        self._core_sample_indices = core_indices
        self._core_sample_mask = core_mask
        self._sample_roles = roles
        self._neighborhood_counts = counts
        self._k_distances = k_distances
        self._components = core_features
        self.n_features_in_ = validated.shape[1]
        return self

    def fit_predict(self, features: ArrayLike) -> IntArray:
        """Fit and return the immutable training assignments."""

        return self.fit(features).labels_
