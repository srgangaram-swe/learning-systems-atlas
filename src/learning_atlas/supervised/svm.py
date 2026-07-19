"""A deterministic, NumPy-only binary kernel support-vector machine."""

from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.estimators import ClassifierMixin, Estimator
from learning_atlas.core.validation import FloatArray, validate_choices

KernelName = Literal["linear", "rbf", "poly"]
GammaOption = Literal["scale", "auto"] | float
IntArray = NDArray[np.int64]

_ALPHA_EPSILON = 1e-8
_STEP_EPSILON = 1e-12


class SVMConvergenceError(RuntimeError):
    """Raised when strict convergence is requested but SMO exhausts its budget."""

    def __init__(self, *, iterations: int, max_kkt_violation: float) -> None:
        self.iterations = iterations
        self.max_kkt_violation = max_kkt_violation
        message = (
            "KernelSVMClassifier did not satisfy its KKT tolerance after "
            f"{iterations} iterations (maximum violation={max_kkt_violation:.6g})"
        )
        super().__init__(message)


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"{name} must be a real scalar"
        raise TypeError(msg)
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        msg = f"{name} must be a finite positive value"
        raise ValueError(msg)
    return converted


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    converted = int(value)
    if converted <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return converted


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"{name} must be a real scalar"
        raise TypeError(msg)
    converted = float(value)
    if not np.isfinite(converted):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return converted


class KernelSVMClassifier(ClassifierMixin, Estimator):
    """Binary C-SVM trained by deterministic sequential minimal optimization.

    The solver scans samples and secondary coordinates in stable index order after
    ranking them by error separation.  When ``raise_on_nonconvergence`` is false,
    the best finite iterate remains usable and ``converged_`` plus the fitted KKT
    diagnostics make the incomplete optimization explicit.  Strict callers can
    request an atomic ``SVMConvergenceError`` instead.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        kernel: KernelName = "rbf",
        gamma: GammaOption = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1_000,
        max_passes: int = 10,
        raise_on_nonconvergence: bool = False,
    ) -> None:
        super().__init__()
        self.C = _positive_float(C, name="C")
        self.kernel = validate_choices(
            kernel,
            name="kernel",
            choices=("linear", "poly", "rbf"),
        )
        if isinstance(gamma, str):
            self.gamma: str | float = validate_choices(
                gamma,
                name="gamma",
                choices=("auto", "scale"),
            )
        elif isinstance(gamma, bool):
            msg = "gamma must be 'scale', 'auto', or a finite positive value"
            raise ValueError(msg)
        else:
            self.gamma = _positive_float(gamma, name="gamma")
        self.degree = _positive_integer(degree, name="degree")
        self.coef0 = _finite_float(coef0, name="coef0")
        self.tol = _positive_float(tol, name="tol")
        self.max_iter = _positive_integer(max_iter, name="max_iter")
        self.max_passes = _positive_integer(max_passes, name="max_passes")
        if not isinstance(raise_on_nonconvergence, bool):
            msg = "raise_on_nonconvergence must be a boolean"
            raise TypeError(msg)
        self.raise_on_nonconvergence = raise_on_nonconvergence

        self._classes: FloatArray | None = None
        self._support: IntArray | None = None
        self._support_vectors: FloatArray | None = None
        self._support_labels: FloatArray | None = None
        self._support_alphas: FloatArray | None = None
        self._alphas: FloatArray | None = None
        self._dual_coef: FloatArray | None = None
        self._intercept: float | None = None
        self._gamma: float | None = None
        self._n_iter: int | None = None
        self._converged: bool | None = None
        self._kkt_violation: FloatArray | None = None
        self._dual_objective: float | None = None
        self._n_support: IntArray | None = None

    @property
    def classes_(self) -> FloatArray:
        """The two sorted labels used for negative and positive margins."""

        self._require_fitted()
        assert self._classes is not None
        return self._classes.copy()

    @property
    def support_(self) -> IntArray:
        """Indices of training rows with non-zero dual coefficients."""

        self._require_fitted()
        assert self._support is not None
        return self._support.copy()

    @property
    def support_vectors_(self) -> FloatArray:
        """Owned support-vector matrix."""

        self._require_fitted()
        assert self._support_vectors is not None
        return self._support_vectors.copy()

    @property
    def dual_coef_(self) -> FloatArray:
        """Signed support-vector coefficients with shape ``(1, n_support)``."""

        self._require_fitted()
        assert self._dual_coef is not None
        return self._dual_coef.copy()

    @property
    def alphas_(self) -> FloatArray:
        """Full non-negative dual solution in training-row order."""

        self._require_fitted()
        assert self._alphas is not None
        return self._alphas.copy()

    @property
    def intercept_(self) -> FloatArray:
        """Fitted bias as a one-element array for estimator interoperability."""

        self._require_fitted()
        assert self._intercept is not None
        return np.asarray([self._intercept], dtype=np.float64)

    @property
    def gamma_(self) -> float:
        """Numerical kernel scale resolved during fitting."""

        self._require_fitted()
        assert self._gamma is not None
        return self._gamma

    @property
    def n_iter_(self) -> int:
        """Number of complete SMO scans performed."""

        self._require_fitted()
        assert self._n_iter is not None
        return self._n_iter

    @property
    def converged_(self) -> bool:
        """Whether the final dual solution satisfies the configured KKT tolerance."""

        self._require_fitted()
        assert self._converged is not None
        return self._converged

    @property
    def kkt_violation_(self) -> FloatArray:
        """Per-training-sample KKT residual at the final iterate."""

        self._require_fitted()
        assert self._kkt_violation is not None
        return self._kkt_violation.copy()

    @property
    def max_kkt_violation_(self) -> float:
        """Largest fitted KKT residual."""

        return float(np.max(self.kkt_violation_))

    @property
    def dual_objective_(self) -> float:
        """Final maximized dual objective value."""

        self._require_fitted()
        assert self._dual_objective is not None
        return self._dual_objective

    @property
    def n_support_(self) -> IntArray:
        """Support-vector counts for the negative and positive classes."""

        self._require_fitted()
        assert self._n_support is not None
        return self._n_support.copy()

    def _resolve_gamma(self, features: FloatArray) -> float:
        if isinstance(self.gamma, float):
            return self.gamma
        if self.gamma == "auto":
            return float(1.0 / features.shape[1])
        variance = float(np.var(features))
        if variance <= np.finfo(np.float64).eps:
            return 1.0
        return float(1.0 / (features.shape[1] * variance))

    def _kernel_matrix(
        self,
        left: FloatArray,
        right: FloatArray,
        *,
        gamma: float,
    ) -> FloatArray:
        dot_products = left @ right.T
        if self.kernel == "linear":
            matrix = dot_products
        elif self.kernel == "poly":
            with np.errstate(over="ignore", invalid="ignore"):
                matrix = (gamma * dot_products + self.coef0) ** self.degree
        else:
            left_norm = np.sum(left * left, axis=1)[:, np.newaxis]
            right_norm = np.sum(right * right, axis=1)[np.newaxis, :]
            squared_distances = np.maximum(left_norm + right_norm - 2.0 * dot_products, 0.0)
            matrix = np.exp(-gamma * squared_distances)
        converted = np.asarray(matrix, dtype=np.float64)
        if not np.all(np.isfinite(converted)):
            msg = "kernel evaluation produced non-finite values; rescale data or kernel parameters"
            raise ValueError(msg)
        return converted

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Fit the binary dual problem with deterministic SMO updates."""

        validated_features, validated_targets = self._validate_fit_data(features, targets)
        classes = np.asarray(np.unique(validated_targets), dtype=np.float64)
        if len(classes) != 2:
            msg = f"KernelSVMClassifier requires exactly two classes; received {len(classes)}"
            raise ValueError(msg)
        signed_targets = np.where(validated_targets == classes[0], -1.0, 1.0)
        signed_targets = np.asarray(signed_targets, dtype=np.float64)
        gamma = self._resolve_gamma(validated_features)
        gram = self._kernel_matrix(validated_features, validated_features, gamma=gamma)

        alphas = np.zeros(len(validated_features), dtype=np.float64)
        alpha_epsilon = min(_ALPHA_EPSILON, self.C * 1e-6)
        step_epsilon = min(_STEP_EPSILON, self.C * 1e-8)
        bias = 0.0
        iterations = 0
        passes_without_change = 0
        while iterations < self.max_iter and passes_without_change < self.max_passes:
            changed = 0
            for first in range(len(alphas)):
                signed_dual = alphas * signed_targets
                first_error = float(signed_dual @ gram[:, first] + bias - signed_targets[first])
                first_margin_error = signed_targets[first] * first_error
                violates = (
                    first_margin_error < -self.tol and alphas[first] < self.C - alpha_epsilon
                ) or (first_margin_error > self.tol and alphas[first] > alpha_epsilon)
                if not violates:
                    continue

                errors = gram.T @ signed_dual + bias - signed_targets
                separation = np.abs(first_error - errors)
                candidates = np.argsort(-separation, kind="stable")
                for second_value in candidates:
                    second = int(second_value)
                    if second == first:
                        continue
                    first_old = alphas[first]
                    second_old = alphas[second]
                    if signed_targets[first] == signed_targets[second]:
                        lower = max(0.0, first_old + second_old - self.C)
                        upper = min(self.C, first_old + second_old)
                    else:
                        lower = max(0.0, second_old - first_old)
                        upper = min(self.C, self.C + second_old - first_old)
                    if upper - lower <= step_epsilon:
                        continue

                    curvature = (
                        gram[first, first] + gram[second, second] - 2.0 * gram[first, second]
                    )
                    if curvature <= _STEP_EPSILON:
                        continue
                    second_new = (
                        second_old
                        + signed_targets[second] * (first_error - float(errors[second])) / curvature
                    )
                    second_new = float(np.clip(second_new, lower, upper))
                    if abs(second_new - second_old) <= step_epsilon * (
                        second_new + second_old + step_epsilon
                    ):
                        continue
                    first_new = first_old + (
                        signed_targets[first] * signed_targets[second] * (second_old - second_new)
                    )

                    first_delta = first_new - first_old
                    second_delta = second_new - second_old
                    first_bias = (
                        bias
                        - first_error
                        - signed_targets[first] * first_delta * gram[first, first]
                        - signed_targets[second] * second_delta * gram[first, second]
                    )
                    second_bias = (
                        bias
                        - float(errors[second])
                        - signed_targets[first] * first_delta * gram[first, second]
                        - signed_targets[second] * second_delta * gram[second, second]
                    )
                    alphas[first] = first_new
                    alphas[second] = second_new
                    if alpha_epsilon < first_new < self.C - alpha_epsilon:
                        bias = first_bias
                    elif alpha_epsilon < second_new < self.C - alpha_epsilon:
                        bias = second_bias
                    else:
                        bias = 0.5 * (first_bias + second_bias)
                    changed += 1
                    break

            iterations += 1
            passes_without_change = passes_without_change + 1 if changed == 0 else 0
            decision = gram @ (alphas * signed_targets) + bias
            violations = self._kkt_residuals(alphas, signed_targets * decision)
            if float(np.max(violations)) <= self.tol:
                break

        # Re-estimate the intercept from free support vectors only when doing so
        # improves the fitted KKT residual. Averaging is numerically helpful in
        # many problems, but must not turn an already-converged iterate into an
        # unconverged published model.
        final_decision = gram @ (alphas * signed_targets) + bias
        kkt_violation = self._kkt_residuals(alphas, signed_targets * final_decision)
        max_violation = float(np.max(kkt_violation))
        free = (alphas > alpha_epsilon) & (alphas < self.C - alpha_epsilon)
        if np.any(free):
            no_bias_decision = gram @ (alphas * signed_targets)
            candidate_bias = float(np.mean(signed_targets[free] - no_bias_decision[free]))
            candidate_decision = no_bias_decision + candidate_bias
            candidate_violation = self._kkt_residuals(
                alphas,
                signed_targets * candidate_decision,
            )
            candidate_max = float(np.max(candidate_violation))
            if candidate_max <= max_violation:
                bias = candidate_bias
                kkt_violation = candidate_violation
                max_violation = candidate_max
        converged = max_violation <= self.tol
        if not converged and self.raise_on_nonconvergence:
            raise SVMConvergenceError(
                iterations=iterations,
                max_kkt_violation=max_violation,
            )

        support = np.flatnonzero(alphas > alpha_epsilon).astype(np.int64)
        support_vectors = validated_features[support].copy()
        support_labels = signed_targets[support].copy()
        support_alphas = alphas[support].copy()
        dual_coef = (support_alphas * support_labels)[np.newaxis, :]
        n_support = np.asarray(
            [
                np.sum(validated_targets[support] == classes[0]),
                np.sum(validated_targets[support] == classes[1]),
            ],
            dtype=np.int64,
        )
        signed_dual = alphas * signed_targets
        dual_objective = float(np.sum(alphas) - 0.5 * signed_dual @ gram @ signed_dual)

        self._classes = classes
        self._support = support
        self._support_vectors = support_vectors
        self._support_labels = support_labels
        self._support_alphas = support_alphas
        self._alphas = alphas
        self._dual_coef = np.asarray(dual_coef, dtype=np.float64)
        self._intercept = bias
        self._gamma = gamma
        self._n_iter = iterations
        self._converged = converged
        self._kkt_violation = kkt_violation
        self._dual_objective = dual_objective
        self._n_support = n_support
        self._mark_fitted(validated_features.shape[1])
        return self

    def _kkt_residuals(self, alphas: FloatArray, margins: FloatArray) -> FloatArray:
        residuals = np.empty_like(alphas)
        alpha_epsilon = min(_ALPHA_EPSILON, self.C * 1e-6)
        at_lower = alphas <= alpha_epsilon
        at_upper = alphas >= self.C - alpha_epsilon
        free = ~(at_lower | at_upper)
        residuals[at_lower] = np.maximum(0.0, 1.0 - margins[at_lower])
        residuals[at_upper] = np.maximum(0.0, margins[at_upper] - 1.0)
        residuals[free] = np.abs(margins[free] - 1.0)
        return np.asarray(residuals, dtype=np.float64)

    def decision_function(self, features: ArrayLike) -> FloatArray:
        """Return signed distance-like kernel scores before label thresholding."""

        validated = self._validate_predict_data(features)
        assert self._support_vectors is not None
        assert self._support_labels is not None
        assert self._support_alphas is not None
        assert self._intercept is not None
        assert self._gamma is not None
        kernel = self._kernel_matrix(validated, self._support_vectors, gamma=self._gamma)
        return np.asarray(
            kernel @ (self._support_alphas * self._support_labels) + self._intercept,
            dtype=np.float64,
        )

    def predict(self, features: ArrayLike) -> FloatArray:
        """Predict binary labels, resolving an exact zero margin to the smaller label."""

        decision = self.decision_function(features)
        assert self._classes is not None
        return np.asarray(np.where(decision > 0.0, self._classes[1], self._classes[0]))
