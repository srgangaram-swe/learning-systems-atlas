"""Transparent naive baselines for controlled from-scratch comparisons."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from learning_atlas.core.estimators import ClassifierMixin, Estimator, RegressorMixin
from learning_atlas.core.validation import FloatArray


class MeanRegressor(RegressorMixin, Estimator):
    """Predict the training-target mean for every observation."""

    def __init__(self) -> None:
        super().__init__()
        self._mean: float | None = None

    @property
    def mean_(self) -> float:
        self._require_fitted()
        assert self._mean is not None
        return self._mean

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        mean = float(np.mean(validated_targets))
        self._mean = mean
        self._mark_fitted(validated_features.shape[1])
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        validated = self._validate_predict_data(features)
        return np.full(len(validated), self.mean_, dtype=np.float64)


class PriorClassifier(ClassifierMixin, Estimator):
    """Predict the most frequent class and expose empirical class probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self._classes: FloatArray | None = None
        self._class_prior: FloatArray | None = None

    @property
    def classes_(self) -> FloatArray:
        self._require_fitted()
        assert self._classes is not None
        return self._classes.copy()

    @property
    def class_prior_(self) -> FloatArray:
        self._require_fitted()
        assert self._class_prior is not None
        return self._class_prior.copy()

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        classes, counts = np.unique(validated_targets, return_counts=True)
        if len(classes) < 2:
            msg = "PriorClassifier requires at least two target classes"
            raise ValueError(msg)
        priors = np.asarray(counts / len(validated_targets), dtype=np.float64)
        self._classes = np.asarray(classes, dtype=np.float64)
        self._class_prior = priors
        self._mark_fitted(validated_features.shape[1])
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        validated = self._validate_predict_data(features)
        classes = self.classes_
        prior = self.class_prior_
        return np.full(len(validated), classes[int(np.argmax(prior))], dtype=np.float64)

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        validated = self._validate_predict_data(features)
        return np.tile(self.class_prior_, (len(validated), 1))
