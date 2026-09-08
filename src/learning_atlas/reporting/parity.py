"""Fixed-hypothesis numerical qualification against independent sklearn oracles.

This reporting harness does not select models. All three seeds and all nine
predeclared pairs are reported, including failures. Fit and prediction timings
are machine-dependent observations, never correctness thresholds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn import linear_model, naive_bayes, neighbors, tree

from learning_atlas.supervised import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GaussianNB,
    KNeighborsClassifier,
    KNeighborsRegressor,
    Lasso,
    LinearRegression,
    MultinomialNB,
    Ridge,
)

FloatArray = NDArray[np.float64]


class Predictor(Protocol):
    """Minimal fit/predict boundary shared with independent reference estimators."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> object: ...

    def predict(self, X: ArrayLike) -> ArrayLike: ...


@dataclass(frozen=True, slots=True)
class ParityPair:
    """A predeclared oracle pair, absolute prediction tolerance and task units."""

    name: str
    task: Literal["regression", "classification", "counts"]
    factory: Callable[[], tuple[Predictor, Predictor]]
    tolerance: float


@dataclass(frozen=True, slots=True)
class ParityObservation:
    """One immutable seed/pair measurement; loss is MSE or misclassification rate."""

    family: str
    seed: int
    metric: str
    tolerance: float
    max_prediction_error: float
    max_training_prediction_error: float
    passed: bool
    scratch_train_loss: float
    scratch_test_loss: float
    oracle_test_loss: float
    naive_test_loss: float
    scratch_fit_seconds: float
    oracle_fit_seconds: float
    scratch_predict_seconds: float
    oracle_predict_seconds: float


def reference_pairs() -> tuple[ParityPair, ...]:
    """Return fresh factories; no fitted state or mutable RNG is shared."""
    return (
        ParityPair(
            "OLS",
            "regression",
            lambda: (LinearRegression(), linear_model.LinearRegression()),
            1e-10,
        ),
        ParityPair(
            "Ridge", "regression", lambda: (Ridge(alpha=0.5), linear_model.Ridge(alpha=0.5)), 1e-10
        ),
        ParityPair(
            "Lasso",
            "regression",
            lambda: (
                Lasso(alpha=0.05, tol=1e-10),
                linear_model.Lasso(alpha=0.05, tol=1e-10, max_iter=10000),
            ),
            1e-7,
        ),
        ParityPair(
            "Gaussian NB", "classification", lambda: (GaussianNB(), naive_bayes.GaussianNB()), 1e-12
        ),
        ParityPair(
            "Multinomial NB",
            "counts",
            lambda: (MultinomialNB(), naive_bayes.MultinomialNB()),
            1e-12,
        ),
        ParityPair(
            "kNN regression",
            "regression",
            lambda: (
                KNeighborsRegressor(n_neighbors=5),
                neighbors.KNeighborsRegressor(n_neighbors=5),
            ),
            1e-10,
        ),
        ParityPair(
            "kNN classification",
            "classification",
            lambda: (
                KNeighborsClassifier(n_neighbors=5),
                neighbors.KNeighborsClassifier(n_neighbors=5),
            ),
            1e-12,
        ),
        ParityPair(
            "CART regression",
            "regression",
            lambda: (
                DecisionTreeRegressor(max_depth=3, random_state=0),
                tree.DecisionTreeRegressor(max_depth=3, random_state=0),
            ),
            1e-8,
        ),
        ParityPair(
            "CART classification",
            "classification",
            lambda: (
                DecisionTreeClassifier(max_depth=3, random_state=0),
                tree.DecisionTreeClassifier(max_depth=3, random_state=0),
            ),
            1e-12,
        ),
    )


def prediction_error(
    actual: ArrayLike, expected: ArrayLike, tolerance: float
) -> tuple[float, bool]:
    """Compare non-empty finite vectors; reject broadcasting and invalid tolerances."""
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or not left.size:
        raise ValueError("parity predictions require aligned non-empty vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("parity predictions must be finite")
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("parity tolerance must be finite and positive")
    error = float(np.max(np.abs(left - right)))
    return error, error <= tolerance


def _data(seed: int, task: str) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(160, 5))
    target = features @ np.array([2.0, -1.0, 0.5, 0.0, 1.5]) + rng.normal(scale=0.1, size=160)
    if task != "regression":
        target = (target > 0).astype(float)
    if task == "counts":
        features = np.rint(np.abs(features) * 4)
    return features, target


def _loss(target: FloatArray, prediction: FloatArray, regression: bool) -> float:
    return float(
        np.mean((target - prediction) ** 2) if regression else np.mean(target != prediction)
    )


def measure_pair(pair: ParityPair, seed: int) -> ParityObservation:
    """Fit on 120 synthetic rows, evaluate 40 untouched rows; no learned preprocessing.

    Complexity follows the paired model; inputs are fixed at 160 by 5. Fitting or
    numerical failures propagate with pair/seed context, never become pass rows.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    features, target = _data(seed, pair.task)
    scratch, oracle = pair.factory()
    fit_times: list[float] = []
    predict_times: list[float] = []
    predictions: list[FloatArray] = []
    try:
        for model in (scratch, oracle):
            started = time.perf_counter()
            model.fit(features[:120], target[:120])
            fit_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            predictions.append(np.asarray(model.predict(features[120:]), dtype=float))
            predict_times.append(time.perf_counter() - started)
        train_prediction = np.asarray(scratch.predict(features[:120]), dtype=float)
        error, passed = prediction_error(predictions[0], predictions[1], pair.tolerance)
        train_error, _ = prediction_error(
            train_prediction, oracle.predict(features[:120]), pair.tolerance
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise RuntimeError(f"reference parity failed for {pair.name}, seed={seed}") from exc
    regression = pair.task == "regression"
    naive = float(np.mean(target[:120])) if regression else float(np.mean(target[:120]) >= 0.5)
    return ParityObservation(
        pair.name,
        seed,
        "MSE" if regression else "misclassification fraction",
        pair.tolerance,
        error,
        train_error,
        passed,
        _loss(target[:120], train_prediction, regression),
        _loss(target[120:], predictions[0], regression),
        _loss(target[120:], predictions[1], regression),
        _loss(target[120:], np.full(40, naive), regression),
        fit_times[0],
        fit_times[1],
        predict_times[0],
        predict_times[1],
    )


def measure_reference_parity() -> tuple[ParityObservation, ...]:
    """Evaluate all fixed seeds/pairs in stable order without test-set selection."""
    return tuple(measure_pair(pair, seed) for pair in reference_pairs() for seed in (17, 29, 43))
