"""Small, typed estimator contracts for from-scratch supervised models."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from learning_atlas.core.validation import (
    FloatArray,
    validate_features,
    validate_targets,
    validate_X_y,
)


class NotFittedError(RuntimeError):
    """Raised when inference is requested before estimator fitting."""


class Estimator(ABC):
    """Base fitted-state and feature-shape contract for supervised estimators."""

    def __init__(self) -> None:
        self._n_features_in: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether a successful fit has established model state."""

        return self._n_features_in is not None

    @property
    def n_features_in_(self) -> int:
        """Number of fitted input features, following the scikit-learn naming convention."""

        self._require_fitted()
        assert self._n_features_in is not None
        return self._n_features_in

    def _validate_fit_data(
        self,
        features: ArrayLike,
        targets: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        validated_features, validated_targets = validate_X_y(features, targets)
        return validated_features, validated_targets

    def _mark_fitted(self, n_features: int) -> None:
        if n_features <= 0:
            msg = "fitted feature count must be positive"
            raise ValueError(msg)
        self._n_features_in = n_features

    def _validate_predict_data(self, features: ArrayLike) -> FloatArray:
        self._require_fitted()
        assert self._n_features_in is not None
        return validate_features(features, expected_features=self._n_features_in)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            msg = f"{type(self).__name__} is not fitted; call fit before inference"
            raise NotFittedError(msg)

    @abstractmethod
    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Fit estimator state and return self."""

    @abstractmethod
    def predict(self, features: ArrayLike) -> FloatArray:
        """Predict one value or label per sample."""


class RegressorMixin(ABC):
    """Coefficient-of-determination scoring for numerical regressors."""

    @abstractmethod
    def predict(self, features: ArrayLike) -> FloatArray:
        """Predict continuous outcomes."""

    def score(self, features: ArrayLike, targets: ArrayLike) -> float:
        """Return finite R², matching scikit-learn's constant-target convention."""

        predictions = self.predict(features)
        observed = validate_targets(targets, n_samples=len(predictions))
        residual_sum = float(np.sum((observed - predictions) ** 2))
        total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
        if total_sum == 0.0:
            return 1.0 if residual_sum == 0.0 else 0.0
        return 1.0 - residual_sum / total_sum


class ClassifierMixin(ABC):
    """Accuracy scoring for numerical classifiers."""

    @abstractmethod
    def predict(self, features: ArrayLike) -> FloatArray:
        """Predict class labels."""

    def score(self, features: ArrayLike, targets: ArrayLike) -> float:
        """Return mean classification accuracy."""

        predictions = self.predict(features)
        observed = validate_targets(targets, n_samples=len(predictions))
        return float(np.mean(predictions == observed))
