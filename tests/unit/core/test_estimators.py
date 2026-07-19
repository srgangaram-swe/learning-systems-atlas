"""Estimator fitted-state, prediction-shape, and mixin-score tests."""

from typing import Self

import numpy as np
import pytest
from numpy.typing import ArrayLike

from learning_atlas.core.estimators import (
    ClassifierMixin,
    Estimator,
    NotFittedError,
    RegressorMixin,
)
from learning_atlas.core.validation import FloatArray

pytestmark = pytest.mark.unit


class MeanRegressor(RegressorMixin, Estimator):
    def __init__(self) -> None:
        super().__init__()
        self._mean: float | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        self._mean = float(np.mean(validated_targets))
        self._mark_fitted(validated_features.shape[1])
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        validated = self._validate_predict_data(features)
        assert self._mean is not None
        return np.full(len(validated), self._mean, dtype=np.float64)

    def mark_invalid_fit_for_test(self) -> None:
        self._mark_fitted(0)


class MajorityClassifier(ClassifierMixin, Estimator):
    def __init__(self) -> None:
        super().__init__()
        self._majority: float | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        labels, counts = np.unique(validated_targets, return_counts=True)
        self._majority = float(labels[np.argmax(counts)])
        self._mark_fitted(validated_features.shape[1])
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        validated = self._validate_predict_data(features)
        assert self._majority is not None
        return np.full(len(validated), self._majority, dtype=np.float64)


def test_not_fitted_state_has_clear_error() -> None:
    estimator = MeanRegressor()
    assert estimator.is_fitted is False
    with pytest.raises(NotFittedError, match="MeanRegressor is not fitted"):
        _ = estimator.n_features_in_
    with pytest.raises(NotFittedError, match="call fit"):
        estimator.predict([[1.0]])


def test_fit_records_feature_contract_and_returns_self() -> None:
    estimator = MeanRegressor()
    assert estimator.fit([[1.0, 2.0], [3.0, 4.0]], [1.0, 3.0]) is estimator
    assert estimator.is_fitted is True
    assert estimator.n_features_in_ == 2
    np.testing.assert_array_equal(estimator.predict([[5.0, 6.0]]), [2.0])
    with pytest.raises(ValueError, match="expects 2"):
        estimator.predict([[5.0]])
    with pytest.raises(ValueError, match="positive"):
        estimator.mark_invalid_fit_for_test()


def test_regressor_score_handles_regular_and_constant_targets() -> None:
    estimator = MeanRegressor().fit([[0.0], [1.0], [2.0]], [1.0, 2.0, 3.0])
    assert estimator.score([[0.0], [1.0], [2.0]], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    perfect_constant = MeanRegressor().fit([[0.0], [1.0]], [4.0, 4.0])
    assert perfect_constant.score([[2.0], [3.0]], [4.0, 4.0]) == 1.0
    assert perfect_constant.score([[2.0], [3.0]], [3.0, 3.0]) == 0.0


def test_classifier_score_returns_accuracy() -> None:
    estimator = MajorityClassifier().fit([[0.0], [1.0], [2.0]], [1.0, 1.0, 0.0])
    assert estimator.score([[3.0], [4.0], [5.0]], [1.0, 0.0, 1.0]) == pytest.approx(2 / 3)
