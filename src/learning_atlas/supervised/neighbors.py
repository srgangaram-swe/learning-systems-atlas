"""From-scratch nearest-neighbor classifiers and regressors.

The implementation deliberately stores the training set: k-nearest neighbors is a
lazy learner.  Query-to-training distances are evaluated in vectorized NumPy
blocks, while stable sorting and sorted class labels make every tie deterministic.
"""

from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.estimators import ClassifierMixin, Estimator, RegressorMixin
from learning_atlas.core.validation import FloatArray, validate_choices

DistanceMetric = Literal["euclidean", "manhattan", "minkowski"]
WeightStrategy = Literal["uniform", "distance"]
IntArray = NDArray[np.int64]


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    validated = int(value)
    if validated <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return validated


class _KNeighborsBase(Estimator):
    """Shared validation, storage, and vectorized distance evaluation."""

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: WeightStrategy = "uniform",
        metric: DistanceMetric = "minkowski",
        p: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_neighbors = _positive_integer(n_neighbors, name="n_neighbors")
        self.weights = validate_choices(
            weights,
            name="weights",
            choices=("uniform", "distance"),
        )
        self.metric = validate_choices(
            metric,
            name="metric",
            choices=("euclidean", "manhattan", "minkowski"),
        )
        if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
            msg = "p must be a real scalar"
            raise TypeError(msg)
        validated_p = float(p)
        if not np.isfinite(validated_p) or validated_p < 1.0:
            msg = "p must be a finite value greater than or equal to 1"
            raise ValueError(msg)
        self.p = validated_p
        self._fit_features: FloatArray | None = None
        self._fit_targets: FloatArray | None = None

    def _fit_training_data(self, features: ArrayLike, targets: ArrayLike) -> None:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        if self.n_neighbors > len(validated_features):
            msg = (
                f"n_neighbors={self.n_neighbors} exceeds the "
                f"{len(validated_features)} available training samples"
            )
            raise ValueError(msg)

        # Own immutable snapshots rather than retaining caller-controlled views.
        fit_features = validated_features.copy()
        fit_targets = validated_targets.copy()
        fit_features.flags.writeable = False
        fit_targets.flags.writeable = False
        self._fit_features = fit_features
        self._fit_targets = fit_targets
        self._mark_fitted(fit_features.shape[1])

    def _training_data(self) -> tuple[FloatArray, FloatArray]:
        self._require_fitted()
        assert self._fit_features is not None
        assert self._fit_targets is not None
        return self._fit_features, self._fit_targets

    def _pairwise_distances(self, queries: FloatArray) -> FloatArray:
        fit_features, _ = self._training_data()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            absolute = np.abs(queries[:, np.newaxis, :] - fit_features[np.newaxis, :, :])
            if self.metric == "manhattan":
                distances = np.sum(absolute, axis=2)
            else:
                exponent = 2.0 if self.metric == "euclidean" else self.p
                scale = np.max(absolute, axis=2)
                normalized = np.divide(
                    absolute,
                    scale[:, :, np.newaxis],
                    out=np.zeros_like(absolute),
                    where=scale[:, :, np.newaxis] != 0.0,
                )
                distances = scale * np.sum(normalized**exponent, axis=2) ** (1.0 / exponent)
        converted = np.asarray(distances, dtype=np.float64)
        if not np.all(np.isfinite(converted)):
            msg = "distance computation produced non-finite values; rescale the feature data"
            raise ValueError(msg)
        return converted

    def kneighbors(self, features: ArrayLike) -> tuple[FloatArray, IntArray]:
        """Return sorted neighbor distances and training indices for each query.

        Equal-distance neighbors retain their training-row order.  This stable
        secondary key is intentional and is part of the estimator's deterministic
        tie contract.
        """

        queries = self._validate_predict_data(features)
        distances = self._pairwise_distances(queries)
        order = np.argsort(distances, axis=1, kind="stable")[:, : self.n_neighbors]
        selected = np.take_along_axis(distances, order, axis=1)
        return np.asarray(selected, dtype=np.float64), np.asarray(order, dtype=np.int64)

    def _neighbor_weights(self, distances: FloatArray) -> FloatArray:
        if self.weights == "uniform":
            return np.ones_like(distances)

        zero = distances == 0.0
        rows_with_zero = np.any(zero, axis=1)
        weights = np.zeros_like(distances)
        if np.any(rows_with_zero):
            weights[rows_with_zero] = zero[rows_with_zero].astype(np.float64)
        nonzero_rows = ~rows_with_zero
        if np.any(nonzero_rows):
            positive_distances = distances[nonzero_rows]
            minimum = np.min(positive_distances, axis=1, keepdims=True)
            # Scaling all reciprocal weights in a row by its smallest distance
            # preserves the normalized vote/mean while keeping every weight in
            # [0, 1]. Direct ``1 / distance`` overflows for finite subnormals.
            weights[nonzero_rows] = minimum / positive_distances
        return weights


class KNeighborsClassifier(ClassifierMixin, _KNeighborsBase):
    """k-nearest-neighbor classifier with deterministic weighted voting."""

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: WeightStrategy = "uniform",
        metric: DistanceMetric = "minkowski",
        p: float = 2.0,
    ) -> None:
        super().__init__(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p)
        self._classes: FloatArray | None = None

    @property
    def classes_(self) -> FloatArray:
        """Sorted labels observed during fitting."""

        self._require_fitted()
        assert self._classes is not None
        return self._classes.copy()

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Store a validated training snapshot and its sorted class vocabulary."""

        validated_features, validated_targets = self._validate_fit_data(features, targets)
        if self.n_neighbors > len(validated_features):
            msg = (
                f"n_neighbors={self.n_neighbors} exceeds the "
                f"{len(validated_features)} available training samples"
            )
            raise ValueError(msg)
        classes = np.asarray(np.unique(validated_targets), dtype=np.float64)

        # Commit all state only after every classifier-specific check succeeds.
        self._fit_training_data(validated_features, validated_targets)
        classes.flags.writeable = False
        self._classes = classes
        return self

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        """Return normalized class vote weights in ``classes_`` order."""

        distances, indices = self.kneighbors(features)
        _, fit_targets = self._training_data()
        assert self._classes is not None
        labels = fit_targets[indices]
        weights = self._neighbor_weights(distances)
        votes = np.stack(
            [np.sum(weights * (labels == label), axis=1) for label in self._classes],
            axis=1,
        )
        totals = np.sum(votes, axis=1, keepdims=True)
        return np.asarray(votes / totals, dtype=np.float64)

    def predict(self, features: ArrayLike) -> FloatArray:
        """Predict labels; an exact vote tie resolves to the smallest label."""

        probabilities = self.predict_proba(features)
        assert self._classes is not None
        return np.asarray(self._classes[np.argmax(probabilities, axis=1)], dtype=np.float64)


class KNeighborsRegressor(RegressorMixin, _KNeighborsBase):
    """k-nearest-neighbor regressor with uniform or inverse-distance means."""

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Store an owned, validated training snapshot."""

        self._fit_training_data(features, targets)
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        """Return the weighted mean target among each query's neighbors."""

        distances, indices = self.kneighbors(features)
        _, fit_targets = self._training_data()
        values = fit_targets[indices]
        weights = self._neighbor_weights(distances)
        return np.asarray(
            np.sum(weights * values, axis=1) / np.sum(weights, axis=1),
            dtype=np.float64,
        )
