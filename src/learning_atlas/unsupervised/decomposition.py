"""From-scratch principal component analysis via a thin NumPy SVD."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from learning_atlas.core.validation import FloatArray, validate_features
from learning_atlas.unsupervised.base import UnsupervisedModel


class DecompositionConvergenceError(RuntimeError):
    """Raised when the numerical SVD cannot produce a valid decomposition."""


def _component_count(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = "n_components must be a positive integer or None"
        raise TypeError(msg)
    converted = int(value)
    if converted < 1:
        msg = "n_components must be positive"
        raise ValueError(msg)
    return converted


def _orient_components(components: FloatArray) -> FloatArray:
    """Resolve SVD sign ambiguity by making each dominant loading nonnegative."""

    oriented = components.copy()
    dominant_columns = np.argmax(np.abs(oriented), axis=1)
    dominant_loadings = oriented[np.arange(len(oriented)), dominant_columns]
    signs = np.where(dominant_loadings < 0.0, -1.0, 1.0)
    oriented *= signs[:, np.newaxis]
    return np.asarray(oriented, dtype=np.float64)


class PCA(UnsupervisedModel):
    """Principal component analysis on centered data using a thin SVD.

    Explained variances use the sample-covariance convention (``ddof=1``).
    When whitening is enabled, projected coordinates are divided by the
    retained sample standard deviations, so their sample covariance is the
    identity up to floating-point tolerance. Whitening is rejected when any
    requested direction is numerically rank deficient.
    """

    def __init__(self, n_components: int | None = None, *, whiten: bool = False) -> None:
        super().__init__()
        self.n_components = _component_count(n_components)
        if not isinstance(whiten, bool):
            msg = "whiten must be a boolean"
            raise TypeError(msg)
        self.whiten = whiten
        self._n_components: int | None = None
        self._mean: FloatArray | None = None
        self._components: FloatArray | None = None
        self._singular_values: FloatArray | None = None
        self._explained_variance: FloatArray | None = None
        self._explained_variance_ratio: FloatArray | None = None
        self._rank: int | None = None

    @property
    def n_components_(self) -> int:
        """Number of retained principal directions."""

        self._require_fitted()
        assert self._n_components is not None
        return self._n_components

    @property
    def mean_(self) -> FloatArray:
        """Per-feature training mean used for centering."""

        self._require_fitted()
        assert self._mean is not None
        return self._mean.copy()

    @property
    def components_(self) -> FloatArray:
        """Orthonormal principal axes in decreasing-variance order."""

        self._require_fitted()
        assert self._components is not None
        return self._components.copy()

    @property
    def singular_values_(self) -> FloatArray:
        """Singular values associated with the retained axes."""

        self._require_fitted()
        assert self._singular_values is not None
        return self._singular_values.copy()

    @property
    def explained_variance_(self) -> FloatArray:
        """Sample variance explained by each retained component."""

        self._require_fitted()
        assert self._explained_variance is not None
        return self._explained_variance.copy()

    @property
    def explained_variance_ratio_(self) -> FloatArray:
        """Fraction of total sample variance explained by each component."""

        self._require_fitted()
        assert self._explained_variance_ratio is not None
        return self._explained_variance_ratio.copy()

    @property
    def rank_(self) -> int:
        """Numerical rank of the centered training matrix."""

        self._require_fitted()
        assert self._rank is not None
        return self._rank

    def fit(self, features: ArrayLike) -> Self:
        """Fit principal axes without explicitly forming a covariance matrix."""

        validated = validate_features(features, min_samples=2)
        n_samples, n_features = validated.shape
        maximum_components = min(n_samples, n_features)
        retained = maximum_components if self.n_components is None else self.n_components
        assert retained is not None
        if retained > maximum_components:
            msg = (
                f"n_components={retained} exceeds the thin-SVD limit "
                f"min(n_samples, n_features)={maximum_components}"
            )
            raise ValueError(msg)

        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.asarray(np.mean(validated, axis=0), dtype=np.float64)
            centered = np.asarray(validated - mean, dtype=np.float64)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(centered)):
            msg = "PCA centering overflowed; rescale the input features"
            raise DecompositionConvergenceError(msg)
        try:
            _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError as error:
            msg = "PCA thin SVD did not converge; inspect feature scaling and conditioning"
            raise DecompositionConvergenceError(msg) from error

        singular_values = np.asarray(singular_values, dtype=np.float64)
        right_vectors = np.asarray(right_vectors, dtype=np.float64)
        if not np.all(np.isfinite(singular_values)) or not np.all(np.isfinite(right_vectors)):
            msg = "PCA thin SVD produced non-finite state; rescale the input features"
            raise DecompositionConvergenceError(msg)

        largest = float(singular_values[0]) if len(singular_values) else 0.0
        rank_tolerance = np.finfo(np.float64).eps * max(centered.shape) * largest
        numerical_rank = int(np.count_nonzero(singular_values > rank_tolerance))
        selected_singular_values = singular_values[:retained].copy()
        if self.whiten and np.any(selected_singular_values <= rank_tolerance):
            msg = (
                "whitening requires every retained component to have positive numerical "
                f"variance; requested {retained} component(s) from rank {numerical_rank} data"
            )
            raise ValueError(msg)

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            all_variances = np.asarray(
                np.square(singular_values / np.sqrt(n_samples - 1)), dtype=np.float64
            )
        if not np.all(np.isfinite(all_variances)):
            msg = "PCA explained variance overflowed; rescale the input features"
            raise DecompositionConvergenceError(msg)
        selected_variances = all_variances[:retained].copy()
        if self.whiten and np.any(selected_variances <= 0.0):
            msg = (
                "whitening requires every retained component to have representable "
                "positive variance; rescale the input features"
            )
            raise ValueError(msg)
        if largest == 0.0:
            variance_ratios = np.zeros(retained, dtype=np.float64)
        else:
            scaled_singular_values = singular_values / largest
            scaled_sum_of_squares = float(np.sum(np.square(scaled_singular_values)))
            variance_ratios = np.asarray(
                np.square(scaled_singular_values[:retained]) / scaled_sum_of_squares,
                dtype=np.float64,
            )
            variance_ratios = np.clip(variance_ratios, 0.0, 1.0)

        components = _orient_components(right_vectors[:retained])
        self._n_components = retained
        self._mean = mean
        self._components = components
        self._singular_values = selected_singular_values
        self._explained_variance = selected_variances
        self._explained_variance_ratio = variance_ratios
        self._rank = numerical_rank
        self.n_features_in_ = n_features
        return self

    def transform(self, features: ArrayLike) -> FloatArray:
        """Project observations into the fitted principal-coordinate system."""

        validated = self._inference_features(features)
        assert self._mean is not None
        assert self._components is not None
        projected = np.asarray((validated - self._mean) @ self._components.T, dtype=np.float64)
        if self.whiten:
            assert self._explained_variance is not None
            projected /= np.sqrt(self._explained_variance)
        return projected

    def inverse_transform(self, representation: ArrayLike) -> FloatArray:
        """Map retained coordinates back to the original feature space."""

        self._require_fitted()
        assert self._n_components is not None
        validated = validate_features(representation, expected_features=self._n_components)
        restored_coordinates = validated.copy()
        if self.whiten:
            assert self._explained_variance is not None
            restored_coordinates *= np.sqrt(self._explained_variance)
        assert self._components is not None
        assert self._mean is not None
        return np.asarray(restored_coordinates @ self._components + self._mean, dtype=np.float64)

    def fit_transform(self, features: ArrayLike) -> FloatArray:
        """Fit on and project the same feature matrix."""

        return self.fit(features).transform(features)


__all__ = ["PCA", "DecompositionConvergenceError"]
