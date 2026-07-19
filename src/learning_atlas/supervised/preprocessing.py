"""Leakage-safe NumPy preprocessing and deterministic data splitting."""

from typing import NamedTuple, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.estimators import NotFittedError
from learning_atlas.core.validation import (
    FloatArray,
    validate_features,
    validate_targets,
    validate_X_y,
)
from learning_atlas.supervised.datasets import RandomState, _rng

IntArray = NDArray[np.int64]


class SplitData(NamedTuple):
    """Named, tuple-compatible train/test arrays."""

    x_train: FloatArray
    x_test: FloatArray
    y_train: FloatArray
    y_test: FloatArray


class StandardScaler:
    """Standardize columns using training-set population moments.

    Zero-variance columns receive a scale of one, remain finite, and round-trip
    exactly through :meth:`inverse_transform`.
    """

    def __init__(self, *, with_mean: bool = True, with_std: bool = True) -> None:
        if not isinstance(with_mean, bool) or not isinstance(with_std, bool):
            msg = "with_mean and with_std must be booleans"
            raise TypeError(msg)
        self.with_mean = with_mean
        self.with_std = with_std
        self._mean: FloatArray | None = None
        self._var: FloatArray | None = None
        self._scale: FloatArray | None = None
        self._n_features_in: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether training moments have been established."""

        return self._n_features_in is not None

    @property
    def n_features_in_(self) -> int:
        """Number of fitted feature columns."""

        self._require_fitted()
        assert self._n_features_in is not None
        return self._n_features_in

    @property
    def mean_(self) -> FloatArray:
        """Population mean learned independently for each feature."""

        self._require_fitted()
        assert self._mean is not None
        return self._mean.copy()

    @property
    def var_(self) -> FloatArray:
        """Population variance learned independently for each feature."""

        self._require_fitted()
        assert self._var is not None
        return self._var.copy()

    @property
    def scale_(self) -> FloatArray:
        """Effective per-feature scale, with constant columns mapped to one."""

        self._require_fitted()
        assert self._scale is not None
        return self._scale.copy()

    def fit(self, features: ArrayLike) -> Self:
        """Learn moments from one finite two-dimensional training array."""

        validated = validate_features(features)
        mean = np.asarray(np.mean(validated, axis=0), dtype=np.float64)
        variance = np.asarray(np.var(validated, axis=0), dtype=np.float64)
        scale = np.asarray(np.sqrt(variance), dtype=np.float64)
        scale[scale == 0.0] = 1.0
        self._mean = mean
        self._var = variance
        self._scale = scale
        self._n_features_in = validated.shape[1]
        return self

    def transform(self, features: ArrayLike) -> FloatArray:
        """Apply fitted centering and scaling without mutating the input."""

        validated = self._validate_transform_data(features)
        assert self._mean is not None
        assert self._scale is not None
        transformed = validated.copy()
        if self.with_mean:
            transformed -= self._mean
        if self.with_std:
            transformed /= self._scale
        return np.asarray(transformed, dtype=np.float64)

    def inverse_transform(self, features: ArrayLike) -> FloatArray:
        """Undo this instance's fitted transformation."""

        validated = self._validate_transform_data(features)
        assert self._mean is not None
        assert self._scale is not None
        restored = validated.copy()
        if self.with_std:
            restored *= self._scale
        if self.with_mean:
            restored += self._mean
        return np.asarray(restored, dtype=np.float64)

    def fit_transform(self, features: ArrayLike) -> FloatArray:
        """Fit on and transform the same training array."""

        return self.fit(features).transform(features)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            msg = "StandardScaler is not fitted; call fit before transformation"
            raise NotFittedError(msg)

    def _validate_transform_data(self, features: ArrayLike) -> FloatArray:
        self._require_fitted()
        assert self._n_features_in is not None
        return validate_features(features, expected_features=self._n_features_in)


def _test_count(test_size: float | int, n_samples: int) -> int:
    if isinstance(test_size, bool):
        msg = "test_size must be a float proportion or integer sample count"
        raise TypeError(msg)
    if isinstance(test_size, (int, np.integer)):
        count = int(test_size)
        if not 1 <= count < n_samples:
            msg = "integer test_size must be in [1, n_samples)"
            raise ValueError(msg)
        return count
    if not isinstance(test_size, (float, np.floating)):
        msg = "test_size must be a float proportion or integer sample count"
        raise TypeError(msg)
    proportion = float(test_size)
    if not np.isfinite(proportion) or not 0.0 < proportion < 1.0:
        msg = "float test_size must be a finite value strictly between 0 and 1"
        raise ValueError(msg)
    count = int(np.ceil(proportion * n_samples))
    if count >= n_samples:
        msg = "test_size leaves no training samples"
        raise ValueError(msg)
    return count


def _stratified_indices(
    stratification: FloatArray,
    *,
    n_test: int,
    generator: np.random.Generator,
    shuffle: bool,
) -> tuple[IntArray, IntArray]:
    labels, inverse, counts = np.unique(stratification, return_inverse=True, return_counts=True)
    if len(labels) < 2:
        msg = "stratification requires at least two classes"
        raise ValueError(msg)
    if np.any(counts < 2):
        msg = "every stratification class must contain at least two samples"
        raise ValueError(msg)
    n_train = len(stratification) - n_test
    if n_test < len(labels) or n_train < len(labels):
        msg = "train and test partitions must each have at least one sample per class"
        raise ValueError(msg)

    ideal = counts.astype(np.float64) * n_test / len(stratification)
    allocation = np.floor(ideal).astype(np.int64)
    allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts - 1)
    fractions = ideal - np.floor(ideal)
    while int(np.sum(allocation)) < n_test:
        candidates = np.flatnonzero(allocation < counts - 1)
        if len(candidates) == 0:
            msg = "unable to allocate the requested stratified test partition"
            raise ValueError(msg)
        chosen = int(candidates[np.argmax(fractions[candidates])])
        allocation[chosen] += 1
        fractions[chosen] = -1.0
    while int(np.sum(allocation)) > n_test:
        candidates = np.flatnonzero(allocation > 1)
        if len(candidates) == 0:
            msg = "unable to allocate the requested stratified test partition"
            raise ValueError(msg)
        chosen = int(candidates[np.argmin(fractions[candidates])])
        allocation[chosen] -= 1
        fractions[chosen] = 2.0

    train_parts: list[IntArray] = []
    test_parts: list[IntArray] = []
    for class_index, class_test_count in enumerate(allocation):
        indices = np.flatnonzero(inverse == class_index).astype(np.int64)
        if shuffle:
            indices = generator.permutation(indices)
        test_parts.append(indices[: int(class_test_count)])
        train_parts.append(indices[int(class_test_count) :])
    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    if shuffle:
        train_indices = generator.permutation(train_indices)
        test_indices = generator.permutation(test_indices)
    return train_indices, test_indices


def train_test_split(
    features: ArrayLike,
    targets: ArrayLike,
    *,
    test_size: float | int = 0.25,
    seed: RandomState = 0,
    shuffle: bool = True,
    stratify: ArrayLike | None = None,
) -> SplitData:
    """Create a deterministic holdout boundary before learned preprocessing.

    Passing a ``numpy.random.Generator`` intentionally advances that generator;
    passing the same integer seed produces identical partitions on every call.
    """

    if not isinstance(shuffle, bool):
        msg = "shuffle must be a boolean"
        raise TypeError(msg)
    validated_features, validated_targets = validate_X_y(features, targets)
    if len(validated_features) < 2:
        msg = "train_test_split requires at least two samples"
        raise ValueError(msg)
    n_test = _test_count(test_size, len(validated_features))
    generator = _rng(seed)

    if stratify is not None:
        stratification = validate_targets(stratify, n_samples=len(validated_features))
        train_indices, test_indices = _stratified_indices(
            stratification,
            n_test=n_test,
            generator=generator,
            shuffle=shuffle,
        )
    else:
        indices = np.arange(len(validated_features), dtype=np.int64)
        if shuffle:
            indices = generator.permutation(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    return SplitData(
        x_train=np.asarray(validated_features[train_indices], dtype=np.float64),
        x_test=np.asarray(validated_features[test_indices], dtype=np.float64),
        y_train=np.asarray(validated_targets[train_indices], dtype=np.float64),
        y_test=np.asarray(validated_targets[test_indices], dtype=np.float64),
    )
