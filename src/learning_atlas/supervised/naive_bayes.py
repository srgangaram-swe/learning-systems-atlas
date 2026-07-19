"""Numerically stable, from-scratch Naive Bayes classifiers."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.estimators import ClassifierMixin, Estimator
from learning_atlas.core.validation import FloatArray

IntArray = NDArray[np.int64]


def _validate_positive(value: object, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"{name} must be a real scalar"
        raise TypeError(msg)
    converted = float(value)
    lower_bound_satisfied = converted >= 0.0 if allow_zero else converted > 0.0
    if not np.isfinite(converted) or not lower_bound_satisfied:
        qualifier = "non-negative" if allow_zero else "positive"
        msg = f"{name} must be a finite {qualifier} value"
        raise ValueError(msg)
    return converted


def _validate_class_prior(class_prior: ArrayLike | None, n_classes: int) -> FloatArray | None:
    if class_prior is None:
        return None
    try:
        prior = np.asarray(class_prior)
    except ValueError as error:
        msg = "class_prior must be a one-dimensional numeric array"
        raise ValueError(msg) from error
    if not np.issubdtype(prior.dtype, np.number) or np.issubdtype(prior.dtype, np.complexfloating):
        msg = "class_prior must contain real numeric values"
        raise TypeError(msg)
    probabilities = np.asarray(prior, dtype=np.float64)
    if probabilities.ndim != 1 or len(probabilities) != n_classes:
        msg = f"class_prior must contain exactly {n_classes} probabilities"
        raise ValueError(msg)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        msg = "class_prior probabilities must be finite and strictly positive"
        raise ValueError(msg)
    if not np.isclose(np.sum(probabilities), 1.0, rtol=1e-10, atol=1e-12):
        msg = "class_prior probabilities must sum to 1"
        raise ValueError(msg)
    return probabilities.copy()


def _normalized_log_probabilities(joint_log_likelihood: FloatArray) -> FloatArray:
    """Normalize row-wise log scores using the log-sum-exp identity."""

    if not np.all(np.isfinite(joint_log_likelihood)):
        msg = (
            "Naive Bayes inference produced non-finite joint log likelihoods; "
            "rescale feature magnitudes"
        )
        raise ValueError(msg)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        maxima = np.max(joint_log_likelihood, axis=1, keepdims=True)
        centered = joint_log_likelihood - maxima
        # Subtract in centered coordinates.  Adding a small log-normalizer to an
        # enormous negative maximum can round the normalizer away entirely.
        normalized = centered - np.log(np.sum(np.exp(centered), axis=1, keepdims=True))
    if not np.all(np.isfinite(normalized)):
        msg = (
            "Naive Bayes inference exceeded the stable log-probability range; "
            "rescale feature magnitudes"
        )
        raise ValueError(msg)
    return np.asarray(normalized, dtype=np.float64)


class _NaiveBayesBase(ClassifierMixin, Estimator):
    """Common class state and stable probability inference."""

    def __init__(self, *, class_prior: ArrayLike | None = None) -> None:
        super().__init__()
        self.class_prior = class_prior
        self._classes: FloatArray | None = None
        self._class_log_prior: FloatArray | None = None

    @property
    def classes_(self) -> FloatArray:
        """Sorted class labels observed during fitting."""

        self._require_fitted()
        assert self._classes is not None
        return self._classes.copy()

    @property
    def class_log_prior_(self) -> FloatArray:
        """Natural logarithm of the fitted class probabilities."""

        self._require_fitted()
        assert self._class_log_prior is not None
        return self._class_log_prior.copy()

    def _joint_log_likelihood(self, features: FloatArray) -> FloatArray:
        raise NotImplementedError

    def predict_log_proba(self, features: ArrayLike) -> FloatArray:
        """Return normalized log posterior probabilities in ``classes_`` order."""

        validated = self._validate_predict_data(features)
        # Finite inputs can still overflow a squared Gaussian residual or a
        # multinomial dot product. Convert that numerical loss of information
        # into the shared actionable inference error instead of leaking a NumPy
        # warning followed by NaN probabilities.
        with np.errstate(over="ignore", invalid="ignore"):
            joint_log_likelihood = self._joint_log_likelihood(validated)
        return _normalized_log_probabilities(joint_log_likelihood)

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        """Return finite posterior probabilities whose rows sum to one."""

        return np.asarray(np.exp(self.predict_log_proba(features)), dtype=np.float64)

    def predict(self, features: ArrayLike) -> FloatArray:
        """Predict the maximum-posterior class with smallest-label tie breaking."""

        probabilities = self.predict_proba(features)
        assert self._classes is not None
        return np.asarray(self._classes[np.argmax(probabilities, axis=1)], dtype=np.float64)

    def _class_state(self, targets: FloatArray) -> tuple[FloatArray, IntArray, FloatArray]:
        classes, encoded, counts = np.unique(targets, return_inverse=True, return_counts=True)
        classes_array = np.asarray(classes, dtype=np.float64)
        encoded_array = np.asarray(encoded, dtype=np.int64)
        counts_array = np.asarray(counts, dtype=np.float64)
        supplied_prior = _validate_class_prior(self.class_prior, len(classes_array))
        probabilities = (
            counts_array / np.sum(counts_array) if supplied_prior is None else supplied_prior
        )
        return classes_array, encoded_array, np.log(probabilities)


class GaussianNB(_NaiveBayesBase):
    """Gaussian Naive Bayes with per-class moments and variance smoothing."""

    def __init__(
        self,
        *,
        var_smoothing: float = 1e-9,
        class_prior: ArrayLike | None = None,
    ) -> None:
        super().__init__(class_prior=class_prior)
        self.var_smoothing = _validate_positive(
            var_smoothing,
            name="var_smoothing",
            allow_zero=True,
        )
        self._theta: FloatArray | None = None
        self._var: FloatArray | None = None
        self._class_count: FloatArray | None = None
        self._epsilon: float | None = None

    @property
    def theta_(self) -> FloatArray:
        """Per-class feature means."""

        self._require_fitted()
        assert self._theta is not None
        return self._theta.copy()

    @property
    def var_(self) -> FloatArray:
        """Smoothed per-class feature variances."""

        self._require_fitted()
        assert self._var is not None
        return self._var.copy()

    @property
    def class_count_(self) -> FloatArray:
        """Training sample count for each class."""

        self._require_fitted()
        assert self._class_count is not None
        return self._class_count.copy()

    @property
    def epsilon_(self) -> float:
        """Additive variance floor used by the fitted model."""

        self._require_fitted()
        assert self._epsilon is not None
        return self._epsilon

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Estimate class priors and diagonal Gaussian likelihoods."""

        validated_features, validated_targets = self._validate_fit_data(features, targets)
        classes, encoded, class_log_prior = self._class_state(validated_targets)
        n_classes = len(classes)
        n_features = validated_features.shape[1]
        theta = np.empty((n_classes, n_features), dtype=np.float64)
        variances = np.empty_like(theta)
        counts = np.bincount(encoded, minlength=n_classes).astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for class_index in range(n_classes):
                class_features = validated_features[encoded == class_index]
                theta[class_index] = np.mean(class_features, axis=0)
                variances[class_index] = np.var(class_features, axis=0)

            scale = float(np.max(np.var(validated_features, axis=0)))
            epsilon = max(self.var_smoothing * scale, np.finfo(np.float64).eps)
            variances += epsilon
        if (
            not np.all(np.isfinite(theta))
            or not np.all(np.isfinite(variances))
            or not np.isfinite(epsilon)
        ):
            msg = "GaussianNB fitting produced non-finite moments; rescale feature magnitudes"
            raise ValueError(msg)

        self._classes = classes
        self._class_log_prior = class_log_prior
        self._theta = theta
        self._var = variances
        self._class_count = counts
        self._epsilon = epsilon
        self._mark_fitted(n_features)
        return self

    def _joint_log_likelihood(self, features: FloatArray) -> FloatArray:
        assert self._theta is not None
        assert self._var is not None
        assert self._class_log_prior is not None
        differences = features[:, np.newaxis, :] - self._theta[np.newaxis, :, :]
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * np.pi * self._var)[np.newaxis, :, :]
            + differences * differences / self._var[np.newaxis, :, :],
            axis=2,
        )
        return np.asarray(log_likelihood + self._class_log_prior, dtype=np.float64)


class MultinomialNB(_NaiveBayesBase):
    """Multinomial Naive Bayes for non-negative count or frequency features."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: ArrayLike | None = None,
    ) -> None:
        super().__init__(class_prior=class_prior)
        self.alpha = _validate_positive(alpha, name="alpha")
        if not isinstance(fit_prior, bool):
            msg = "fit_prior must be a boolean"
            raise TypeError(msg)
        self.fit_prior = fit_prior
        self._class_count: FloatArray | None = None
        self._feature_count: FloatArray | None = None
        self._feature_log_prob: FloatArray | None = None

    @property
    def class_count_(self) -> FloatArray:
        """Training sample count for each class."""

        self._require_fitted()
        assert self._class_count is not None
        return self._class_count.copy()

    @property
    def feature_count_(self) -> FloatArray:
        """Observed per-class feature totals before smoothing."""

        self._require_fitted()
        assert self._feature_count is not None
        return self._feature_count.copy()

    @property
    def feature_log_prob_(self) -> FloatArray:
        """Log conditional probability of every feature given each class."""

        self._require_fitted()
        assert self._feature_log_prob is not None
        return self._feature_log_prob.copy()

    @staticmethod
    def _validate_nonnegative(features: FloatArray) -> None:
        if np.any(features < 0.0):
            msg = "MultinomialNB requires non-negative feature counts"
            raise ValueError(msg)

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Estimate smoothed multinomial likelihoods entirely in log space."""

        validated_features, validated_targets = self._validate_fit_data(features, targets)
        self._validate_nonnegative(validated_features)
        classes, encoded, empirical_log_prior = self._class_state(validated_targets)
        n_classes = len(classes)
        n_features = validated_features.shape[1]
        counts = np.bincount(encoded, minlength=n_classes).astype(np.float64)
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            np.add.at(feature_count, encoded, validated_features)
            smoothed = feature_count + self.alpha
            feature_log_prob = np.log(smoothed) - np.log(np.sum(smoothed, axis=1, keepdims=True))
        if not np.all(np.isfinite(feature_count)) or not np.all(np.isfinite(feature_log_prob)):
            msg = "MultinomialNB fitting produced non-finite feature counts; rescale magnitudes"
            raise ValueError(msg)
        if self.class_prior is not None or self.fit_prior:
            class_log_prior = empirical_log_prior
        else:
            class_log_prior = np.full(n_classes, -np.log(float(n_classes)), dtype=np.float64)

        self._classes = classes
        self._class_log_prior = class_log_prior
        self._class_count = counts
        self._feature_count = feature_count
        self._feature_log_prob = np.asarray(feature_log_prob, dtype=np.float64)
        self._mark_fitted(n_features)
        return self

    def _joint_log_likelihood(self, features: FloatArray) -> FloatArray:
        self._validate_nonnegative(features)
        assert self._feature_log_prob is not None
        assert self._class_log_prior is not None
        return np.asarray(
            features @ self._feature_log_prob.T + self._class_log_prior,
            dtype=np.float64,
        )
