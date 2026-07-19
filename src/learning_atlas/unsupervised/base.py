"""Paradigm-native contracts shared by from-scratch unsupervised models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.validation import FloatArray, validate_features

IntArray = NDArray[np.int64]


class NotFittedError(RuntimeError):
    """Raised when learned unsupervised state is requested before fitting."""


class UnsupervisedModel:
    """Own fitted feature-shape state without imposing a supervised API."""

    def __init__(self) -> None:
        self.n_features_in_: int | None = None

    def _fit_features(self, features: ArrayLike, *, min_samples: int = 1) -> FloatArray:
        validated = validate_features(features, min_samples=min_samples)
        self.n_features_in_ = validated.shape[1]
        return validated

    def _inference_features(self, features: ArrayLike) -> FloatArray:
        if self.n_features_in_ is None:
            msg = f"{type(self).__name__} is not fitted"
            raise NotFittedError(msg)
        return validate_features(features, expected_features=self.n_features_in_)

    def _require_fitted(self) -> None:
        if self.n_features_in_ is None:
            msg = f"{type(self).__name__} is not fitted"
            raise NotFittedError(msg)


@runtime_checkable
class Clusterer(Protocol):
    """Native clustering contract; transductive models need not invent predict."""

    def fit(self, features: ArrayLike) -> Clusterer:
        """Discover structure from features only."""

    def fit_predict(self, features: ArrayLike) -> IntArray:
        """Fit and return one assignment per training observation."""


@runtime_checkable
class Transformer(Protocol):
    """Native representation-learning contract for inductive transforms."""

    def fit(self, features: ArrayLike) -> Transformer:
        """Learn a feature-only representation."""

    def transform(self, features: ArrayLike) -> FloatArray:
        """Apply the learned representation to observations."""
