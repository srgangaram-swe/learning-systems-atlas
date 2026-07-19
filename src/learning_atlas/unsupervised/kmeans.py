"""Deterministic, from-scratch Lloyd clustering with k-means++ seeding."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from learning_atlas.core.reproducibility import derive_named_seed, generator_for_seed
from learning_atlas.core.validation import FloatArray
from learning_atlas.unsupervised.base import IntArray, UnsupervisedModel

Initialization = Literal["k-means++", "random"]


class KMeansConvergenceError(RuntimeError):
    """Raised when a restart cannot reach a stable Lloyd solution."""

    def __init__(
        self,
        *,
        restart: int,
        max_iter: int,
        inertia_history: tuple[float, ...],
    ) -> None:
        self.restart = restart
        self.max_iter = max_iter
        self.inertia_history = inertia_history
        final = inertia_history[-1] if inertia_history else float("nan")
        super().__init__(
            "KMeans did not converge for restart "
            f"{restart} within {max_iter} iterations (last inertia={final:.6g}); "
            "increase max_iter, relax tol, reduce n_clusters, or rescale the features"
        )


class KMeansNumericalError(RuntimeError):
    """Raised when finite inputs produce unsafe floating-point intermediates."""


@dataclass(frozen=True, slots=True)
class _KMeansRun:
    centers: FloatArray
    labels: IntArray
    inertia: float
    n_iter: int
    history: tuple[float, ...]


def _integer_parameter(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    converted = int(value)
    if converted < minimum:
        msg = f"{name} must be at least {minimum}"
        raise ValueError(msg)
    return converted


def _finite_real_parameter(value: float, *, name: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{name} must be a real number"
        raise TypeError(msg)
    converted = float(value)
    if not np.isfinite(converted) or converted < minimum:
        msg = f"{name} must be finite and at least {minimum}"
        raise ValueError(msg)
    return converted


def _squared_distances(features: FloatArray, centers: FloatArray) -> FloatArray:
    """Return pairwise squared Euclidean distances or fail before corruption."""

    try:
        with np.errstate(over="raise", invalid="raise"):
            differences = features[:, np.newaxis, :] - centers[np.newaxis, :, :]
            distances = np.einsum(
                "nkd,nkd->nk",
                differences,
                differences,
                optimize=True,
            )
    except FloatingPointError as error:
        msg = "squared distances overflowed; center or rescale the features"
        raise KMeansNumericalError(msg) from error
    if not np.all(np.isfinite(distances)):
        msg = "squared distances became non-finite; center or rescale the features"
        raise KMeansNumericalError(msg)
    return np.maximum(np.asarray(distances, dtype=np.float64), 0.0)


def _stable_farthest_index(
    distances: FloatArray,
    labels: IntArray,
    counts: IntArray,
) -> int:
    assigned_distances = distances[np.arange(len(labels)), labels]
    candidates = np.flatnonzero(counts[labels] > 1)
    if len(candidates) == 0:  # pragma: no cover - protected by n_samples >= n_clusters
        msg = "cannot repair an empty cluster without emptying another cluster"
        raise KMeansNumericalError(msg)
    order = np.lexsort((candidates, -assigned_distances[candidates]))
    return int(candidates[order[0]])


def _assign_with_empty_cluster_repair(
    distances: FloatArray,
    n_clusters: int,
) -> tuple[IntArray, bool]:
    """Assign stable ties to the lowest index, then repair empty clusters."""

    labels = np.asarray(np.argmin(distances, axis=1), dtype=np.int64)
    counts = np.asarray(np.bincount(labels, minlength=n_clusters), dtype=np.int64)
    repaired = False
    for empty_cluster in np.flatnonzero(counts == 0):
        sample = _stable_farthest_index(distances, labels, counts)
        donor = int(labels[sample])
        labels[sample] = int(empty_cluster)
        counts[donor] -= 1
        counts[empty_cluster] += 1
        repaired = True
    return labels, repaired


class KMeans(UnsupervisedModel):
    """Lloyd's algorithm with isolated restart streams and deterministic ties.

    The implementation is intentionally NumPy-only. Each restart receives a
    SHA-256-namespaced child seed, so changing ``n_init`` never perturbs an
    earlier restart. A failed fit publishes no partial learned state.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        init: Initialization = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        self.n_clusters = _integer_parameter(n_clusters, name="n_clusters", minimum=1)
        if init not in ("k-means++", "random"):
            msg = "init must be one of: k-means++, random"
            raise ValueError(msg)
        self.init = init
        self.n_init = _integer_parameter(n_init, name="n_init", minimum=1)
        self.max_iter = _integer_parameter(max_iter, name="max_iter", minimum=1)
        self.tol = _finite_real_parameter(tol, name="tol", minimum=0.0)
        self.random_state = _integer_parameter(random_state, name="random_state", minimum=0)
        if self.random_state > 2**32 - 1:
            msg = "random_state must fit in an unsigned 32-bit integer"
            raise ValueError(msg)

        self._cluster_centers: FloatArray | None = None
        self._labels: IntArray | None = None
        self._inertia: float | None = None
        self._n_iter: int | None = None
        self._inertia_history: tuple[float, ...] | None = None
        self._restart_histories: tuple[tuple[float, ...], ...] | None = None
        self._restart_inertias: tuple[float, ...] | None = None

    @property
    def cluster_centers_(self) -> FloatArray:
        self._require_fitted()
        assert self._cluster_centers is not None
        return self._cluster_centers.copy()

    @property
    def labels_(self) -> IntArray:
        self._require_fitted()
        assert self._labels is not None
        return self._labels.copy()

    @property
    def inertia_(self) -> float:
        self._require_fitted()
        assert self._inertia is not None
        return self._inertia

    @property
    def n_iter_(self) -> int:
        self._require_fitted()
        assert self._n_iter is not None
        return self._n_iter

    @property
    def inertia_history_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._inertia_history is not None
        return self._inertia_history

    @property
    def restart_histories_(self) -> tuple[tuple[float, ...], ...]:
        self._require_fitted()
        assert self._restart_histories is not None
        return self._restart_histories

    @property
    def restart_inertias_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._restart_inertias is not None
        return self._restart_inertias

    @property
    def converged_(self) -> bool:
        self._require_fitted()
        return True

    def fit(self, features: ArrayLike) -> KMeans:
        """Fit all restarts and atomically retain the minimum-inertia solution."""

        previous_feature_count = self.n_features_in_
        try:
            validated = self._fit_features(features, min_samples=self.n_clusters)
            unique_count = len(np.unique(validated, axis=0))
            if unique_count < self.n_clusters:
                msg = (
                    f"n_clusters={self.n_clusters} exceeds the {unique_count} distinct "
                    "observations; reduce n_clusters"
                )
                raise ValueError(msg)

            runs = tuple(self._fit_restart(validated, restart) for restart in range(self.n_init))
            best = min(enumerate(runs), key=lambda item: (item[1].inertia, item[0]))[1]
        except BaseException:
            self.n_features_in_ = previous_feature_count
            raise

        self._cluster_centers = best.centers.copy()
        self._labels = best.labels.copy()
        self._inertia = best.inertia
        self._n_iter = best.n_iter
        self._inertia_history = best.history
        self._restart_histories = tuple(run.history for run in runs)
        self._restart_inertias = tuple(run.inertia for run in runs)
        return self

    def fit_predict(self, features: ArrayLike) -> IntArray:
        """Fit the model and return a defensive copy of training assignments."""

        self.fit(features)
        return self.labels_

    def predict(self, features: ArrayLike) -> IntArray:
        """Assign observations to their closest center with stable index ties."""

        validated = self._inference_features(features)
        assert self._cluster_centers is not None
        distances = _squared_distances(validated, self._cluster_centers)
        return np.asarray(np.argmin(distances, axis=1), dtype=np.int64)

    def transform(self, features: ArrayLike) -> FloatArray:
        """Return Euclidean distance to every fitted cluster center."""

        validated = self._inference_features(features)
        assert self._cluster_centers is not None
        return np.asarray(
            np.sqrt(_squared_distances(validated, self._cluster_centers)), dtype=np.float64
        )

    def _fit_restart(self, features: FloatArray, restart: int) -> _KMeansRun:
        seed = derive_named_seed(
            self.random_state,
            "kmeans",
            self.init,
            "restart",
            str(restart),
        )
        generator = generator_for_seed(seed)
        centers = self._initialize_centers(features, generator)
        history: list[float] = []
        previous_inertia: float | None = None

        for iteration in range(1, self.max_iter + 1):
            distances = _squared_distances(features, centers)
            labels, repaired = _assign_with_empty_cluster_repair(distances, self.n_clusters)
            new_centers = self._centers_from_labels(features, labels)
            assigned_distances = _squared_distances(features, new_centers)
            inertia = float(np.sum(assigned_distances[np.arange(len(features)), labels]))
            if not np.isfinite(inertia):
                msg = "KMeans inertia became non-finite; center or rescale the features"
                raise KMeansNumericalError(msg)
            history.append(inertia)

            center_shift = float(np.max(np.sqrt(np.diag(_squared_distances(centers, new_centers)))))
            final_distances = _squared_distances(features, new_centers)
            final_labels = np.asarray(np.argmin(final_distances, axis=1), dtype=np.int64)
            final_has_empty = bool(
                np.any(np.bincount(final_labels, minlength=self.n_clusters) == 0)
            )
            assignments_stable = not repaired and np.array_equal(final_labels, labels)
            numerical_tolerance = (
                0.0 if previous_inertia is None else 1e-12 * max(1.0, previous_inertia)
            )
            if (
                previous_inertia is not None
                and not repaired
                and inertia > previous_inertia + numerical_tolerance
            ):
                msg = "KMeans inertia increased during an ordinary Lloyd step; rescale the features"
                raise KMeansNumericalError(msg)
            relative_improvement = (
                float("inf")
                if previous_inertia is None or inertia > previous_inertia + numerical_tolerance
                else max(0.0, previous_inertia - inertia) / max(1.0, previous_inertia)
            )
            tolerance_reached = not final_has_empty and (
                center_shift <= self.tol or relative_improvement <= self.tol
            )
            if assignments_stable or tolerance_reached:
                final_inertia = float(
                    np.sum(final_distances[np.arange(len(features)), final_labels])
                )
                history[-1] = final_inertia
                return _KMeansRun(
                    centers=new_centers,
                    labels=final_labels,
                    inertia=final_inertia,
                    n_iter=iteration,
                    history=tuple(history),
                )

            centers = new_centers
            previous_inertia = inertia

        raise KMeansConvergenceError(
            restart=restart,
            max_iter=self.max_iter,
            inertia_history=tuple(history),
        )

    def _initialize_centers(
        self,
        features: FloatArray,
        generator: np.random.Generator,
    ) -> FloatArray:
        _, unique_indices = np.unique(features, axis=0, return_index=True)
        candidates = np.asarray(np.sort(unique_indices), dtype=np.int64)
        if self.init == "random":
            selected = np.asarray(
                generator.choice(candidates, size=self.n_clusters, replace=False),
                dtype=np.int64,
            )
            return features[selected].copy()

        first = int(generator.integers(0, len(features)))
        selected_indices = [first]
        closest = _squared_distances(features, features[[first]])[:, 0]
        while len(selected_indices) < self.n_clusters:
            maximum = float(np.max(closest))
            if maximum <= 0.0:  # pragma: no cover - distinct-count validation protects this
                msg = "k-means++ exhausted distinct candidates"
                raise KMeansNumericalError(msg)
            scaled = closest / maximum
            probabilities = scaled / float(np.sum(scaled))
            chosen = int(generator.choice(len(features), p=probabilities))
            selected_indices.append(chosen)
            candidate_distances = _squared_distances(features, features[[chosen]])[:, 0]
            closest = np.minimum(closest, candidate_distances)
        return features[np.asarray(selected_indices, dtype=np.int64)].copy()

    def _centers_from_labels(self, features: FloatArray, labels: IntArray) -> FloatArray:
        centers = np.empty((self.n_clusters, features.shape[1]), dtype=np.float64)
        try:
            with np.errstate(over="raise", invalid="raise"):
                for cluster in range(self.n_clusters):
                    centers[cluster] = np.mean(features[labels == cluster], axis=0)
        except FloatingPointError as error:
            msg = "cluster means overflowed; center or rescale the features"
            raise KMeansNumericalError(msg) from error
        if not np.all(np.isfinite(centers)):
            msg = "cluster means became non-finite; center or rescale the features"
            raise KMeansNumericalError(msg)
        return centers
