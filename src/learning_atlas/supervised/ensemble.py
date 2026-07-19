"""NumPy-only bagging and gradient-boosting ensembles.

Random forests use independent bootstrap and tree random streams, align bootstrap
class sets before aggregation, and retain honest out-of-bag coverage.  Gradient
boosting performs stagewise functional gradient descent with bounded shrinkage and a
deterministic validation split when early stopping is requested.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from learning_atlas.core.estimators import ClassifierMixin, Estimator, RegressorMixin
from learning_atlas.core.validation import FloatArray
from learning_atlas.supervised.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    MaxFeatures,
    RandomState,
    _local_generator,
    _validate_integer,
    _validate_max_features,
    _validate_random_state,
)

MaxSamples: TypeAlias = int | float | None
_UINT32_HIGH = int(np.iinfo(np.uint32).max)


def _validate_boolean(value: bool, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        msg = f"{name} must be boolean"
        raise TypeError(msg)
    return bool(value)


def _validate_max_samples(max_samples: MaxSamples) -> MaxSamples:
    if max_samples is None:
        return None
    if isinstance(max_samples, bool):
        msg = "max_samples must not be boolean"
        raise TypeError(msg)
    if isinstance(max_samples, (int, np.integer)):
        if int(max_samples) < 1:
            msg = "integer max_samples must be positive"
            raise ValueError(msg)
        return int(max_samples)
    if isinstance(max_samples, (float, np.floating)):
        value = float(max_samples)
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            msg = "float max_samples must be finite and in (0, 1]"
            raise ValueError(msg)
        return value
    msg = "max_samples must be None, a positive integer, or a float in (0, 1]"
    raise TypeError(msg)


def _resolve_max_samples(max_samples: MaxSamples, n_samples: int) -> int:
    if max_samples is None:
        return n_samples
    if isinstance(max_samples, int):
        if max_samples > n_samples:
            msg = (
                f"integer max_samples={max_samples} exceeds the fitted sample count of {n_samples}"
            )
            raise ValueError(msg)
        return max_samples
    return max(1, int(float(max_samples) * n_samples))


def _validate_forest_parameters(
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: MaxFeatures,
    bootstrap: bool,
    max_samples: MaxSamples,
    oob_score: bool,
    random_state: RandomState,
) -> tuple[int, int | None, int, int, MaxFeatures, bool, MaxSamples, bool, RandomState]:
    n_estimators = _validate_integer(n_estimators, name="n_estimators", minimum=1)
    if max_depth is not None:
        max_depth = _validate_integer(max_depth, name="max_depth", minimum=0)
    min_samples_split = _validate_integer(min_samples_split, name="min_samples_split", minimum=2)
    min_samples_leaf = _validate_integer(min_samples_leaf, name="min_samples_leaf", minimum=1)
    max_features = _validate_max_features(max_features)
    bootstrap = _validate_boolean(bootstrap, name="bootstrap")
    max_samples = _validate_max_samples(max_samples)
    oob_score = _validate_boolean(oob_score, name="oob_score")
    random_state = _validate_random_state(random_state)
    if not bootstrap and max_samples is not None:
        msg = "max_samples is available only when bootstrap=True"
        raise ValueError(msg)
    if not bootstrap and oob_score:
        msg = "oob_score=True requires bootstrap=True"
        raise ValueError(msg)
    return (
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
        max_samples,
        oob_score,
        random_state,
    )


def _bootstrap_indices(
    generator: np.random.Generator,
    *,
    n_samples: int,
    sample_count: int,
    bootstrap: bool,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int], np.dtype[np.bool_]]]:
    if bootstrap:
        sampled = np.asarray(generator.integers(0, n_samples, size=sample_count), dtype=np.int64)
        in_bag = np.zeros(n_samples, dtype=np.bool_)
        in_bag[sampled] = True
        return sampled, ~in_bag
    return np.arange(n_samples, dtype=np.int64), np.zeros(n_samples, dtype=np.bool_)


def _mean_feature_importances(estimators: tuple[DecisionTreeRegressor, ...]) -> FloatArray:
    averaged = np.mean([estimator.feature_importances_ for estimator in estimators], axis=0)
    total = float(np.sum(averaged))
    if total == 0.0:
        return np.zeros_like(averaged, dtype=np.float64)
    return np.asarray(averaged / total, dtype=np.float64)


def _mean_classifier_feature_importances(
    estimators: tuple[DecisionTreeClassifier, ...],
) -> FloatArray:
    averaged = np.mean([estimator.feature_importances_ for estimator in estimators], axis=0)
    total = float(np.sum(averaged))
    if total == 0.0:
        return np.zeros_like(averaged, dtype=np.float64)
    return np.asarray(averaged / total, dtype=np.float64)


def _aligned_probabilities(
    estimator: DecisionTreeClassifier,
    features: FloatArray,
    classes: FloatArray,
) -> FloatArray:
    local_classes = estimator.classes_
    local_probabilities = estimator.predict_proba(features)
    aligned = np.zeros((len(features), len(classes)), dtype=np.float64)
    positions = np.searchsorted(classes, local_classes)
    aligned[:, positions] = local_probabilities
    return aligned


class RandomForestRegressor(RegressorMixin, Estimator):
    """Bootstrap-aggregated regression trees with per-node feature subsampling."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        criterion: Literal["squared_error"] = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = 1.0,
        bootstrap: bool = True,
        max_samples: MaxSamples = None,
        oob_score: bool = True,
        random_state: RandomState = None,
    ) -> None:
        super().__init__()
        if criterion != "squared_error":
            msg = "criterion must be 'squared_error' for RandomForestRegressor"
            raise ValueError(msg)
        (
            self.n_estimators,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_features,
            self.bootstrap,
            self.max_samples,
            self.oob_score,
            self.random_state,
        ) = _validate_forest_parameters(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            max_samples=max_samples,
            oob_score=oob_score,
            random_state=random_state,
        )
        self.criterion = criterion
        self._estimators: tuple[DecisionTreeRegressor, ...] | None = None
        self._feature_importances: FloatArray | None = None
        self._oob_prediction: FloatArray | None = None
        self._oob_score: float | None = None
        self._oob_coverage: float | None = None

    @property
    def estimators_(self) -> tuple[DecisionTreeRegressor, ...]:
        self._require_fitted()
        assert self._estimators is not None
        return self._estimators

    @property
    def feature_importances_(self) -> FloatArray:
        self._require_fitted()
        assert self._feature_importances is not None
        return self._feature_importances.copy()

    @property
    def oob_prediction_(self) -> FloatArray:
        self._require_fitted()
        assert self._oob_prediction is not None
        return self._oob_prediction.copy()

    @property
    def oob_score_(self) -> float:
        self._require_fitted()
        assert self._oob_score is not None
        return self._oob_score

    @property
    def oob_coverage_(self) -> float:
        self._require_fitted()
        assert self._oob_coverage is not None
        return self._oob_coverage

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        sample_count = _resolve_max_samples(self.max_samples, len(validated_features))
        generator = _local_generator(self.random_state)
        estimators: list[DecisionTreeRegressor] = []
        oob_sum = np.zeros(len(validated_features), dtype=np.float64)
        oob_count = np.zeros(len(validated_features), dtype=np.int64)

        for _ in range(self.n_estimators):
            sampled, oob_mask = _bootstrap_indices(
                generator,
                n_samples=len(validated_features),
                sample_count=sample_count,
                bootstrap=self.bootstrap,
            )
            tree_seed = int(generator.integers(0, _UINT32_HIGH))
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            ).fit(validated_features[sampled], validated_targets[sampled])
            estimators.append(tree)
            if self.oob_score and np.any(oob_mask):
                oob_sum[oob_mask] += tree.predict(validated_features[oob_mask])
                oob_count[oob_mask] += 1

        fitted_estimators = tuple(estimators)
        coverage_mask = oob_count > 0
        oob_prediction = np.full(len(validated_features), np.nan, dtype=np.float64)
        oob_prediction[coverage_mask] = oob_sum[coverage_mask] / oob_count[coverage_mask]
        coverage = float(np.mean(coverage_mask)) if self.oob_score else 0.0
        if np.any(coverage_mask):
            observed = validated_targets[coverage_mask]
            predicted = oob_prediction[coverage_mask]
            residual_sum = float(np.sum((observed - predicted) ** 2))
            total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
            score = (
                (1.0 if residual_sum == 0.0 else 0.0)
                if total_sum == 0.0
                else (1.0 - residual_sum / total_sum)
            )
        else:
            score = float("nan")

        self._estimators = fitted_estimators
        self._feature_importances = _mean_feature_importances(fitted_estimators)
        self._oob_prediction = oob_prediction
        self._oob_score = score
        self._oob_coverage = coverage
        self._mark_fitted(validated_features.shape[1])
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        validated = self._validate_predict_data(features)
        predictions = np.asarray(
            [estimator.predict(validated) for estimator in self.estimators_], dtype=np.float64
        )
        return np.asarray(np.mean(predictions, axis=0), dtype=np.float64)


class RandomForestClassifier(ClassifierMixin, Estimator):
    """Bootstrap-aggregated classifier with globally aligned class probabilities."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = "sqrt",
        bootstrap: bool = True,
        max_samples: MaxSamples = None,
        oob_score: bool = True,
        random_state: RandomState = None,
    ) -> None:
        super().__init__()
        if criterion not in {"gini", "entropy"}:
            msg = "criterion must be 'gini' or 'entropy' for RandomForestClassifier"
            raise ValueError(msg)
        (
            self.n_estimators,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_features,
            self.bootstrap,
            self.max_samples,
            self.oob_score,
            self.random_state,
        ) = _validate_forest_parameters(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            max_samples=max_samples,
            oob_score=oob_score,
            random_state=random_state,
        )
        self.criterion = criterion
        self._estimators: tuple[DecisionTreeClassifier, ...] | None = None
        self._classes: FloatArray | None = None
        self._feature_importances: FloatArray | None = None
        self._oob_decision_function: FloatArray | None = None
        self._oob_score: float | None = None
        self._oob_coverage: float | None = None

    @property
    def estimators_(self) -> tuple[DecisionTreeClassifier, ...]:
        self._require_fitted()
        assert self._estimators is not None
        return self._estimators

    @property
    def classes_(self) -> FloatArray:
        self._require_fitted()
        assert self._classes is not None
        return self._classes.copy()

    @property
    def feature_importances_(self) -> FloatArray:
        self._require_fitted()
        assert self._feature_importances is not None
        return self._feature_importances.copy()

    @property
    def oob_decision_function_(self) -> FloatArray:
        self._require_fitted()
        assert self._oob_decision_function is not None
        return self._oob_decision_function.copy()

    @property
    def oob_score_(self) -> float:
        self._require_fitted()
        assert self._oob_score is not None
        return self._oob_score

    @property
    def oob_coverage_(self) -> float:
        self._require_fitted()
        assert self._oob_coverage is not None
        return self._oob_coverage

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        classes = np.unique(validated_targets)
        sample_count = _resolve_max_samples(self.max_samples, len(validated_features))
        generator = _local_generator(self.random_state)
        estimators: list[DecisionTreeClassifier] = []
        oob_sum = np.zeros((len(validated_features), len(classes)), dtype=np.float64)
        oob_count = np.zeros(len(validated_features), dtype=np.int64)

        for _ in range(self.n_estimators):
            sampled, oob_mask = _bootstrap_indices(
                generator,
                n_samples=len(validated_features),
                sample_count=sample_count,
                bootstrap=self.bootstrap,
            )
            tree_seed = int(generator.integers(0, _UINT32_HIGH))
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            ).fit(validated_features[sampled], validated_targets[sampled])
            estimators.append(tree)
            if self.oob_score and np.any(oob_mask):
                oob_sum[oob_mask] += _aligned_probabilities(
                    tree, validated_features[oob_mask], classes
                )
                oob_count[oob_mask] += 1

        fitted_estimators = tuple(estimators)
        coverage_mask = oob_count > 0
        oob_probabilities = np.full(
            (len(validated_features), len(classes)), np.nan, dtype=np.float64
        )
        oob_probabilities[coverage_mask] = (
            oob_sum[coverage_mask] / oob_count[coverage_mask, np.newaxis]
        )
        coverage = float(np.mean(coverage_mask)) if self.oob_score else 0.0
        if np.any(coverage_mask):
            indices = np.argmax(oob_probabilities[coverage_mask], axis=1)
            oob_predictions = classes[indices]
            score = float(np.mean(oob_predictions == validated_targets[coverage_mask]))
        else:
            score = float("nan")

        self._estimators = fitted_estimators
        self._classes = np.asarray(classes, dtype=np.float64)
        self._feature_importances = _mean_classifier_feature_importances(fitted_estimators)
        self._oob_decision_function = oob_probabilities
        self._oob_score = score
        self._oob_coverage = coverage
        self._mark_fitted(validated_features.shape[1])
        return self

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        validated = self._validate_predict_data(features)
        classes = self.classes_
        probabilities = np.zeros((len(validated), len(classes)), dtype=np.float64)
        for estimator in self.estimators_:
            probabilities += _aligned_probabilities(estimator, validated, classes)
        return np.asarray(probabilities / len(self.estimators_), dtype=np.float64)

    def predict(self, features: ArrayLike) -> FloatArray:
        probabilities = self.predict_proba(features)
        return np.asarray(self.classes_[np.argmax(probabilities, axis=1)], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _BoostingParameters:
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    early_stopping_rounds: int | None
    validation_fraction: float
    tol: float
    random_state: RandomState


def _validate_boosting_parameters(
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    early_stopping_rounds: int | None,
    validation_fraction: float,
    tol: float,
    random_state: RandomState,
) -> _BoostingParameters:
    n_estimators = _validate_integer(n_estimators, name="n_estimators", minimum=1)
    max_depth = _validate_integer(max_depth, name="max_depth", minimum=1)
    min_samples_split = _validate_integer(min_samples_split, name="min_samples_split", minimum=2)
    min_samples_leaf = _validate_integer(min_samples_leaf, name="min_samples_leaf", minimum=1)
    if not np.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
        msg = "learning_rate must be finite and in (0, 1]"
        raise ValueError(msg)
    if early_stopping_rounds is not None:
        early_stopping_rounds = _validate_integer(
            early_stopping_rounds, name="early_stopping_rounds", minimum=1
        )
    if not np.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        msg = "validation_fraction must be finite and in (0, 1)"
        raise ValueError(msg)
    if not np.isfinite(tol) or tol < 0.0:
        msg = "tol must be finite and non-negative"
        raise ValueError(msg)
    return _BoostingParameters(
        n_estimators=n_estimators,
        learning_rate=float(learning_rate),
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        early_stopping_rounds=early_stopping_rounds,
        validation_fraction=float(validation_fraction),
        tol=float(tol),
        random_state=_validate_random_state(random_state),
    )


def _random_train_validation_indices(
    n_samples: int,
    *,
    validation_fraction: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int], np.dtype[np.int64]]]:
    if n_samples < 2:
        msg = "early stopping requires at least two samples"
        raise ValueError(msg)
    validation_count = min(n_samples - 1, max(1, int(np.ceil(n_samples * validation_fraction))))
    order = np.asarray(generator.permutation(n_samples), dtype=np.int64)
    return order[validation_count:], order[:validation_count]


def _stratified_train_validation_indices(
    targets: FloatArray,
    *,
    validation_fraction: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int], np.dtype[np.int64]]]:
    training: list[np.ndarray[tuple[int], np.dtype[np.int64]]] = []
    validation: list[np.ndarray[tuple[int], np.dtype[np.int64]]] = []
    for class_label in np.unique(targets):
        indices = np.flatnonzero(targets == class_label).astype(np.int64)
        if len(indices) < 2:
            msg = "early stopping requires at least two samples from each class"
            raise ValueError(msg)
        shuffled = indices[np.asarray(generator.permutation(len(indices)), dtype=np.int64)]
        validation_count = min(
            len(indices) - 1,
            max(1, int(np.ceil(len(indices) * validation_fraction))),
        )
        validation.append(shuffled[:validation_count])
        training.append(shuffled[validation_count:])
    return np.concatenate(training), np.concatenate(validation)


def _squared_loss(observed: FloatArray, predicted: FloatArray) -> float:
    return 0.5 * float(np.mean((observed - predicted) ** 2))


def _sigmoid(raw_predictions: FloatArray) -> FloatArray:
    probabilities = np.empty_like(raw_predictions, dtype=np.float64)
    positive = raw_predictions >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-raw_predictions[positive]))
    exponential = np.exp(raw_predictions[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    return probabilities


def _binary_log_loss(encoded: FloatArray, raw_predictions: FloatArray) -> float:
    return float(np.mean(np.logaddexp(0.0, raw_predictions) - encoded * raw_predictions))


def _monotonic_step(
    current_prediction: FloatArray,
    direction: FloatArray,
    *,
    learning_rate: float,
    loss: Callable[[FloatArray, FloatArray], float],
    observed: FloatArray,
) -> tuple[float, FloatArray, float]:
    """Backtrack only for roundoff/adverse stages, guaranteeing honest loss history."""

    current_loss = loss(observed, current_prediction)
    stage_weight = learning_rate
    for _ in range(60):
        candidate = np.asarray(current_prediction + stage_weight * direction, dtype=np.float64)
        candidate_loss = loss(observed, candidate)
        if candidate_loss <= current_loss:
            return stage_weight, candidate, candidate_loss
        stage_weight *= 0.5
    msg = "boosting stage failed to find a finite non-increasing loss update"
    raise RuntimeError(msg)


def _boosting_feature_importances(
    estimators: tuple[DecisionTreeRegressor, ...], stage_weights: tuple[float, ...]
) -> FloatArray:
    weights = np.asarray(stage_weights, dtype=np.float64)
    stacked = np.asarray([tree.feature_importances_ for tree in estimators], dtype=np.float64)
    importances = np.average(stacked, axis=0, weights=weights)
    total = float(np.sum(importances))
    if total == 0.0:
        return np.zeros_like(importances)
    return np.asarray(importances / total, dtype=np.float64)


class GradientBoostingRegressor(RegressorMixin, Estimator):
    """Least-squares gradient boosting over shallow regression trees."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.1,
        tol: float = 1e-4,
        random_state: RandomState = None,
    ) -> None:
        super().__init__()
        parameters = _validate_boosting_parameters(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            tol=tol,
            random_state=random_state,
        )
        self.n_estimators = parameters.n_estimators
        self.learning_rate = parameters.learning_rate
        self.max_depth = parameters.max_depth
        self.min_samples_split = parameters.min_samples_split
        self.min_samples_leaf = parameters.min_samples_leaf
        self.early_stopping_rounds = parameters.early_stopping_rounds
        self.validation_fraction = parameters.validation_fraction
        self.tol = parameters.tol
        self.random_state = parameters.random_state
        self._estimators: tuple[DecisionTreeRegressor, ...] | None = None
        self._stage_weights: tuple[float, ...] | None = None
        self._train_loss: tuple[float, ...] | None = None
        self._validation_loss: tuple[float, ...] | None = None
        self._initial_prediction: float | None = None
        self._feature_importances: FloatArray | None = None

    @property
    def estimators_(self) -> tuple[DecisionTreeRegressor, ...]:
        self._require_fitted()
        assert self._estimators is not None
        return self._estimators

    @property
    def stage_weights_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._stage_weights is not None
        return self._stage_weights

    @property
    def train_loss_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._train_loss is not None
        return self._train_loss

    @property
    def loss_history_(self) -> tuple[float, ...]:
        return self.train_loss_

    @property
    def validation_loss_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._validation_loss is not None
        return self._validation_loss

    @property
    def n_estimators_(self) -> int:
        return len(self.estimators_)

    @property
    def initial_prediction_(self) -> float:
        self._require_fitted()
        assert self._initial_prediction is not None
        return self._initial_prediction

    @property
    def feature_importances_(self) -> FloatArray:
        self._require_fitted()
        assert self._feature_importances is not None
        return self._feature_importances.copy()

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        generator = _local_generator(self.random_state)
        if self.early_stopping_rounds is None:
            train_features = validated_features
            train_targets = validated_targets
            validation_features: FloatArray | None = None
            validation_targets: FloatArray | None = None
        else:
            train_indices, validation_indices = _random_train_validation_indices(
                len(validated_features),
                validation_fraction=self.validation_fraction,
                generator=generator,
            )
            train_features = validated_features[train_indices]
            train_targets = validated_targets[train_indices]
            validation_features = validated_features[validation_indices]
            validation_targets = validated_targets[validation_indices]

        initial_prediction = float(np.mean(train_targets))
        train_prediction = np.full(len(train_targets), initial_prediction, dtype=np.float64)
        validation_prediction = (
            None
            if validation_features is None
            else np.full(len(validation_features), initial_prediction, dtype=np.float64)
        )
        estimators: list[DecisionTreeRegressor] = []
        stage_weights: list[float] = []
        train_losses: list[float] = []
        validation_losses: list[float] = []
        best_validation = np.inf
        rounds_without_improvement = 0

        for _ in range(self.n_estimators):
            residual = train_targets - train_prediction
            tree_seed = int(generator.integers(0, _UINT32_HIGH))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed,
            ).fit(train_features, residual)
            direction = tree.predict(train_features)
            stage_weight, train_prediction, train_loss = _monotonic_step(
                train_prediction,
                direction,
                learning_rate=self.learning_rate,
                loss=_squared_loss,
                observed=train_targets,
            )
            estimators.append(tree)
            stage_weights.append(stage_weight)
            train_losses.append(train_loss)

            if validation_features is not None:
                assert validation_targets is not None
                assert validation_prediction is not None
                validation_prediction += stage_weight * tree.predict(validation_features)
                validation_loss = _squared_loss(validation_targets, validation_prediction)
                validation_losses.append(validation_loss)
                if validation_loss < best_validation - self.tol:
                    best_validation = validation_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                assert self.early_stopping_rounds is not None
                if rounds_without_improvement >= self.early_stopping_rounds:
                    break

        fitted_estimators = tuple(estimators)
        fitted_weights = tuple(stage_weights)
        self._estimators = fitted_estimators
        self._stage_weights = fitted_weights
        self._train_loss = tuple(train_losses)
        self._validation_loss = tuple(validation_losses)
        self._initial_prediction = initial_prediction
        self._feature_importances = _boosting_feature_importances(fitted_estimators, fitted_weights)
        self._mark_fitted(validated_features.shape[1])
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        validated = self._validate_predict_data(features)
        prediction = np.full(len(validated), self.initial_prediction_, dtype=np.float64)
        for stage_weight, estimator in zip(self.stage_weights_, self.estimators_, strict=True):
            prediction += stage_weight * estimator.predict(validated)
        return prediction


class GradientBoostingClassifier(ClassifierMixin, Estimator):
    """Binary log-loss gradient boosting over shallow regression trees."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.1,
        tol: float = 1e-4,
        random_state: RandomState = None,
    ) -> None:
        super().__init__()
        parameters = _validate_boosting_parameters(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            tol=tol,
            random_state=random_state,
        )
        self.n_estimators = parameters.n_estimators
        self.learning_rate = parameters.learning_rate
        self.max_depth = parameters.max_depth
        self.min_samples_split = parameters.min_samples_split
        self.min_samples_leaf = parameters.min_samples_leaf
        self.early_stopping_rounds = parameters.early_stopping_rounds
        self.validation_fraction = parameters.validation_fraction
        self.tol = parameters.tol
        self.random_state = parameters.random_state
        self._estimators: tuple[DecisionTreeRegressor, ...] | None = None
        self._stage_weights: tuple[float, ...] | None = None
        self._train_loss: tuple[float, ...] | None = None
        self._validation_loss: tuple[float, ...] | None = None
        self._classes: FloatArray | None = None
        self._initial_log_odds: float | None = None
        self._feature_importances: FloatArray | None = None

    @property
    def estimators_(self) -> tuple[DecisionTreeRegressor, ...]:
        self._require_fitted()
        assert self._estimators is not None
        return self._estimators

    @property
    def stage_weights_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._stage_weights is not None
        return self._stage_weights

    @property
    def train_loss_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._train_loss is not None
        return self._train_loss

    @property
    def loss_history_(self) -> tuple[float, ...]:
        return self.train_loss_

    @property
    def validation_loss_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._validation_loss is not None
        return self._validation_loss

    @property
    def n_estimators_(self) -> int:
        return len(self.estimators_)

    @property
    def classes_(self) -> FloatArray:
        self._require_fitted()
        assert self._classes is not None
        return self._classes.copy()

    @property
    def initial_log_odds_(self) -> float:
        self._require_fitted()
        assert self._initial_log_odds is not None
        return self._initial_log_odds

    @property
    def feature_importances_(self) -> FloatArray:
        self._require_fitted()
        assert self._feature_importances is not None
        return self._feature_importances.copy()

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        classes = np.unique(validated_targets)
        if len(classes) != 2:
            msg = "GradientBoostingClassifier requires exactly two target classes"
            raise ValueError(msg)
        encoded = np.asarray(validated_targets == classes[1], dtype=np.float64)
        generator = _local_generator(self.random_state)
        if self.early_stopping_rounds is None:
            train_features = validated_features
            train_targets = encoded
            validation_features: FloatArray | None = None
            validation_targets: FloatArray | None = None
        else:
            train_indices, validation_indices = _stratified_train_validation_indices(
                validated_targets,
                validation_fraction=self.validation_fraction,
                generator=generator,
            )
            train_features = validated_features[train_indices]
            train_targets = encoded[train_indices]
            validation_features = validated_features[validation_indices]
            validation_targets = encoded[validation_indices]

        prevalence = float(np.mean(train_targets))
        epsilon = np.finfo(np.float64).eps
        clipped = float(np.clip(prevalence, epsilon, 1.0 - epsilon))
        initial_log_odds = float(np.log(clipped / (1.0 - clipped)))
        train_raw = np.full(len(train_targets), initial_log_odds, dtype=np.float64)
        validation_raw = (
            None
            if validation_features is None
            else np.full(len(validation_features), initial_log_odds, dtype=np.float64)
        )
        estimators: list[DecisionTreeRegressor] = []
        stage_weights: list[float] = []
        train_losses: list[float] = []
        validation_losses: list[float] = []
        best_validation = np.inf
        rounds_without_improvement = 0

        for _ in range(self.n_estimators):
            negative_gradient = train_targets - _sigmoid(train_raw)
            tree_seed = int(generator.integers(0, _UINT32_HIGH))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed,
            ).fit(train_features, negative_gradient)
            direction = tree.predict(train_features)
            stage_weight, train_raw, train_loss = _monotonic_step(
                train_raw,
                direction,
                learning_rate=self.learning_rate,
                loss=_binary_log_loss,
                observed=train_targets,
            )
            estimators.append(tree)
            stage_weights.append(stage_weight)
            train_losses.append(train_loss)

            if validation_features is not None:
                assert validation_targets is not None
                assert validation_raw is not None
                validation_raw += stage_weight * tree.predict(validation_features)
                validation_loss = _binary_log_loss(validation_targets, validation_raw)
                validation_losses.append(validation_loss)
                if validation_loss < best_validation - self.tol:
                    best_validation = validation_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                assert self.early_stopping_rounds is not None
                if rounds_without_improvement >= self.early_stopping_rounds:
                    break

        fitted_estimators = tuple(estimators)
        fitted_weights = tuple(stage_weights)
        self._estimators = fitted_estimators
        self._stage_weights = fitted_weights
        self._train_loss = tuple(train_losses)
        self._validation_loss = tuple(validation_losses)
        self._classes = np.asarray(classes, dtype=np.float64)
        self._initial_log_odds = initial_log_odds
        self._feature_importances = _boosting_feature_importances(fitted_estimators, fitted_weights)
        self._mark_fitted(validated_features.shape[1])
        return self

    def decision_function(self, features: ArrayLike) -> FloatArray:
        """Return additive raw log odds for the positive class."""

        validated = self._validate_predict_data(features)
        raw = np.full(len(validated), self.initial_log_odds_, dtype=np.float64)
        for stage_weight, estimator in zip(self.stage_weights_, self.estimators_, strict=True):
            raw += stage_weight * estimator.predict(validated)
        return raw

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        positive = _sigmoid(self.decision_function(features))
        return np.column_stack((1.0 - positive, positive)).astype(np.float64, copy=False)

    def predict(self, features: ArrayLike) -> FloatArray:
        positive = self.predict_proba(features)[:, 1] >= 0.5
        return np.asarray(self.classes_[positive.astype(np.int64)], dtype=np.float64)


# Explicit alias documents the supported classification scope without fragmenting the API.
GradientBoostingBinaryClassifier = GradientBoostingClassifier
