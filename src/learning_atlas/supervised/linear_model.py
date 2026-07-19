"""From-scratch linear and logistic estimators implemented with NumPy only."""

from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike

from learning_atlas.core.estimators import ClassifierMixin, Estimator, RegressorMixin
from learning_atlas.core.validation import FloatArray, validate_choices
from learning_atlas.supervised.datasets import RandomState, _rng


class ConvergenceError(RuntimeError):
    """Raised when an iterative solver cannot satisfy its convergence contract."""

    def __init__(self, estimator: str, iterations: int, loss_history: tuple[float, ...]) -> None:
        self.estimator = estimator
        self.iterations = iterations
        self.loss_history = loss_history
        final_loss = loss_history[-1] if loss_history else float("nan")
        super().__init__(
            f"{estimator} did not converge after {iterations} iteration(s); "
            f"final objective={final_loss:.8g}"
        )


def _finite_non_negative(value: float, *, name: str, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"{name} must be a real scalar"
        raise TypeError(msg)
    converted = float(value)
    lower_valid = converted > 0.0 if strictly_positive else converted >= 0.0
    qualifier = "positive" if strictly_positive else "non-negative"
    if not np.isfinite(converted) or not lower_valid:
        msg = f"{name} must be a finite {qualifier} value"
        raise ValueError(msg)
    return converted


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if int(value) < 1:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return int(value)


def _coefficient_problem(
    features: FloatArray,
    targets: FloatArray,
    *,
    fit_intercept: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    if fit_intercept:
        feature_mean = np.asarray(np.mean(features, axis=0), dtype=np.float64)
        target_mean = float(np.mean(targets))
        centered_features = np.asarray(features - feature_mean, dtype=np.float64)
        centered_targets = np.asarray(targets - target_mean, dtype=np.float64)
    else:
        feature_mean = np.zeros(features.shape[1], dtype=np.float64)
        target_mean = 0.0
        centered_features = features
        centered_targets = targets
    return centered_features, centered_targets, feature_mean, target_mean


def _intercept(
    coefficients: FloatArray,
    feature_mean: FloatArray,
    target_mean: float,
    *,
    fit_intercept: bool,
) -> float:
    if not fit_intercept:
        return 0.0
    return target_mean - float(feature_mean @ coefficients)


def _least_squares(
    features: FloatArray,
    targets: FloatArray,
    *,
    fit_intercept: bool,
) -> tuple[FloatArray, float, float]:
    centered_features, centered_targets, feature_mean, target_mean = _coefficient_problem(
        features, targets, fit_intercept=fit_intercept
    )
    try:
        coefficients = np.asarray(
            np.linalg.lstsq(centered_features, centered_targets, rcond=None)[0], dtype=np.float64
        )
    except np.linalg.LinAlgError as error:
        raise ConvergenceError("LinearRegression", 1, ()) from error
    fitted_intercept = _intercept(
        coefficients,
        feature_mean,
        target_mean,
        fit_intercept=fit_intercept,
    )
    residuals = targets - (features @ coefficients + fitted_intercept)
    loss = 0.5 * float(np.mean(residuals * residuals))
    return coefficients, fitted_intercept, loss


class _LinearState(Estimator):
    """Shared immutable fitted-state access for linear predictors."""

    def __init__(self) -> None:
        super().__init__()
        self._coef: FloatArray | None = None
        self._intercept: float | None = None
        self._loss_history: tuple[float, ...] | None = None
        self._n_iter: int | None = None
        self._converged: bool | None = None

    @property
    def coef_(self) -> FloatArray:
        """A defensive copy of fitted feature coefficients."""

        self._require_fitted()
        assert self._coef is not None
        return self._coef.copy()

    @property
    def intercept_(self) -> float:
        """Fitted, unpenalized intercept."""

        self._require_fitted()
        assert self._intercept is not None
        return self._intercept

    @property
    def loss_history_(self) -> tuple[float, ...]:
        """Objective values recorded by the successful solver run."""

        self._require_fitted()
        assert self._loss_history is not None
        return self._loss_history

    @property
    def n_iter_(self) -> int:
        """Number of optimizer updates, or one for a direct solve."""

        self._require_fitted()
        assert self._n_iter is not None
        return self._n_iter

    @property
    def converged_(self) -> bool:
        """Whether the fitted solver satisfied its stopping condition."""

        self._require_fitted()
        assert self._converged is not None
        return self._converged

    def _publish_state(
        self,
        *,
        coefficients: FloatArray,
        intercept: float,
        loss_history: tuple[float, ...],
        n_iter: int,
        n_features: int,
    ) -> None:
        self._coef = coefficients.copy()
        self._intercept = float(intercept)
        self._loss_history = loss_history
        self._n_iter = n_iter
        self._converged = True
        self._mark_fitted(n_features)

    def predict(self, features: ArrayLike) -> FloatArray:
        """Return the fitted affine response."""

        validated = self._validate_predict_data(features)
        assert self._coef is not None
        assert self._intercept is not None
        return np.asarray(validated @ self._coef + self._intercept, dtype=np.float64)


class LinearRegression(RegressorMixin, _LinearState):
    """Ordinary least squares via ``lstsq`` or full-batch gradient descent."""

    def __init__(
        self,
        *,
        solver: Literal["lstsq", "gradient_descent"] = "lstsq",
        fit_intercept: bool = True,
        learning_rate: float | None = None,
        max_iter: int = 10_000,
        tol: float = 1e-8,
        seed: RandomState = 0,
    ) -> None:
        super().__init__()
        self.solver = validate_choices(
            solver,
            name="solver",
            choices=("gradient_descent", "lstsq"),
        )
        if not isinstance(fit_intercept, bool):
            msg = "fit_intercept must be a boolean"
            raise TypeError(msg)
        self.fit_intercept = fit_intercept
        self.learning_rate = (
            None
            if learning_rate is None
            else _finite_non_negative(
                learning_rate,
                name="learning_rate",
                strictly_positive=True,
            )
        )
        self.max_iter = _positive_integer(max_iter, name="max_iter")
        self.tol = _finite_non_negative(tol, name="tol", strictly_positive=True)
        self.seed = seed

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Fit ordinary least squares without explicitly forming an inverse."""

        validated_features, validated_targets = self._validate_fit_data(features, targets)
        if self.solver == "lstsq":
            coefficients, intercept, loss = _least_squares(
                validated_features,
                validated_targets,
                fit_intercept=self.fit_intercept,
            )
            self._publish_state(
                coefficients=coefficients,
                intercept=intercept,
                loss_history=(loss,),
                n_iter=1,
                n_features=validated_features.shape[1],
            )
            return self

        coefficients, intercept, history, n_iter = self._fit_gradient_descent(
            validated_features, validated_targets
        )
        self._publish_state(
            coefficients=coefficients,
            intercept=intercept,
            loss_history=history,
            n_iter=n_iter,
            n_features=validated_features.shape[1],
        )
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        """Return the fitted affine response."""

        return _LinearState.predict(self, features)

    def _fit_gradient_descent(
        self,
        features: FloatArray,
        targets: FloatArray,
    ) -> tuple[FloatArray, float, tuple[float, ...], int]:
        centered_features, centered_targets, feature_mean, target_mean = _coefficient_problem(
            features,
            targets,
            fit_intercept=self.fit_intercept,
        )
        generator = _rng(self.seed)
        coefficients = np.asarray(
            generator.normal(scale=1e-3, size=features.shape[1]), dtype=np.float64
        )
        lipschitz = float(np.linalg.norm(centered_features, ord=2) ** 2 / len(features))
        if lipschitz == 0.0:
            coefficients.fill(0.0)
            intercept = _intercept(
                coefficients,
                feature_mean,
                target_mean,
                fit_intercept=self.fit_intercept,
            )
            residuals = targets - (features @ coefficients + intercept)
            return coefficients, intercept, (0.5 * float(np.mean(residuals**2)),), 0
        step_size = self.learning_rate if self.learning_rate is not None else 1.0 / lipschitz
        history: list[float] = []
        for iteration in range(self.max_iter + 1):
            residuals = centered_features @ coefficients - centered_targets
            loss = 0.5 * float(np.mean(residuals * residuals))
            if not np.isfinite(loss):
                raise ConvergenceError("LinearRegression", iteration, tuple(history))
            history.append(loss)
            gradient = np.asarray(centered_features.T @ residuals / len(features), dtype=np.float64)
            if float(np.max(np.abs(gradient))) <= self.tol:
                intercept = _intercept(
                    coefficients,
                    feature_mean,
                    target_mean,
                    fit_intercept=self.fit_intercept,
                )
                return coefficients, intercept, tuple(history), iteration
            if iteration == self.max_iter:
                break
            coefficients = np.asarray(coefficients - step_size * gradient, dtype=np.float64)
        raise ConvergenceError("LinearRegression", self.max_iter, tuple(history))


class Ridge(RegressorMixin, _LinearState):
    """L2-regularized least squares with an unpenalized intercept."""

    def __init__(self, *, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        super().__init__()
        self.alpha = _finite_non_negative(alpha, name="alpha")
        if not isinstance(fit_intercept, bool):
            msg = "fit_intercept must be a boolean"
            raise TypeError(msg)
        self.fit_intercept = fit_intercept

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Fit through a stable solve, falling back to ``lstsq`` at alpha zero."""

        validated_features, validated_targets = self._validate_fit_data(features, targets)
        centered_features, centered_targets, feature_mean, target_mean = _coefficient_problem(
            validated_features,
            validated_targets,
            fit_intercept=self.fit_intercept,
        )
        if self.alpha == 0.0:
            coefficients = np.asarray(
                np.linalg.lstsq(centered_features, centered_targets, rcond=None)[0],
                dtype=np.float64,
            )
        else:
            gram = centered_features.T @ centered_features
            regularized = gram + self.alpha * np.eye(gram.shape[0], dtype=np.float64)
            try:
                coefficients = np.asarray(
                    np.linalg.solve(regularized, centered_features.T @ centered_targets),
                    dtype=np.float64,
                )
            except np.linalg.LinAlgError as error:
                raise ConvergenceError("Ridge", 1, ()) from error
        intercept = _intercept(
            coefficients,
            feature_mean,
            target_mean,
            fit_intercept=self.fit_intercept,
        )
        residuals = validated_targets - (validated_features @ coefficients + intercept)
        objective = 0.5 * float(np.sum(residuals * residuals)) + 0.5 * self.alpha * float(
            coefficients @ coefficients
        )
        self._publish_state(
            coefficients=coefficients,
            intercept=intercept,
            loss_history=(objective,),
            n_iter=1,
            n_features=validated_features.shape[1],
        )
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        """Return the fitted affine response."""

        return _LinearState.predict(self, features)


def _soft_threshold(value: float, threshold: float) -> float:
    return float(np.sign(value) * max(abs(value) - threshold, 0.0))


class Lasso(RegressorMixin, _LinearState):
    """L1-regularized regression using deterministic cyclic coordinate descent."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 10_000,
        tol: float = 1e-8,
    ) -> None:
        super().__init__()
        self.alpha = _finite_non_negative(alpha, name="alpha")
        if not isinstance(fit_intercept, bool):
            msg = "fit_intercept must be a boolean"
            raise TypeError(msg)
        self.fit_intercept = fit_intercept
        self.max_iter = _positive_integer(max_iter, name="max_iter")
        self.tol = _finite_non_negative(tol, name="tol", strictly_positive=True)

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Fit Lasso and raise rather than publish unconverged coefficients."""

        validated_features, validated_targets = self._validate_fit_data(features, targets)
        if self.alpha == 0.0:
            coefficients, intercept, loss = _least_squares(
                validated_features,
                validated_targets,
                fit_intercept=self.fit_intercept,
            )
            self._publish_state(
                coefficients=coefficients,
                intercept=intercept,
                loss_history=(loss,),
                n_iter=1,
                n_features=validated_features.shape[1],
            )
            return self

        centered_features, centered_targets, feature_mean, target_mean = _coefficient_problem(
            validated_features,
            validated_targets,
            fit_intercept=self.fit_intercept,
        )
        coefficients = np.zeros(validated_features.shape[1], dtype=np.float64)
        residuals = centered_targets.copy()
        squared_norm = np.mean(centered_features * centered_features, axis=0)
        history: list[float] = []
        for iteration in range(1, self.max_iter + 1):
            maximum_change = 0.0
            for feature_index in range(validated_features.shape[1]):
                old_coefficient = float(coefficients[feature_index])
                column = centered_features[:, feature_index]
                if squared_norm[feature_index] == 0.0:
                    new_coefficient = 0.0
                else:
                    residuals += column * old_coefficient
                    correlation = float(column @ residuals / len(centered_features))
                    new_coefficient = _soft_threshold(correlation, self.alpha) / float(
                        squared_norm[feature_index]
                    )
                    residuals -= column * new_coefficient
                coefficients[feature_index] = new_coefficient
                maximum_change = max(maximum_change, abs(new_coefficient - old_coefficient))
            objective = 0.5 * float(np.mean(residuals * residuals)) + self.alpha * float(
                np.sum(np.abs(coefficients))
            )
            if not np.isfinite(objective):
                raise ConvergenceError("Lasso", iteration, tuple(history))
            history.append(objective)
            if maximum_change <= self.tol:
                intercept = _intercept(
                    coefficients,
                    feature_mean,
                    target_mean,
                    fit_intercept=self.fit_intercept,
                )
                self._publish_state(
                    coefficients=coefficients,
                    intercept=intercept,
                    loss_history=tuple(history),
                    n_iter=iteration,
                    n_features=validated_features.shape[1],
                )
                return self
        raise ConvergenceError("Lasso", self.max_iter, tuple(history))

    def predict(self, features: ArrayLike) -> FloatArray:
        """Return the fitted affine response."""

        return _LinearState.predict(self, features)


def _stable_sigmoid(logits: FloatArray) -> FloatArray:
    probabilities = np.empty_like(logits, dtype=np.float64)
    non_negative = logits >= 0.0
    probabilities[non_negative] = 1.0 / (1.0 + np.exp(-logits[non_negative]))
    exponentiated = np.exp(logits[~non_negative])
    probabilities[~non_negative] = exponentiated / (1.0 + exponentiated)
    return probabilities


class LogisticRegression(ClassifierMixin, _LinearState):
    """Binary L2-regularized logistic regression using full-batch gradient descent."""

    def __init__(
        self,
        *,
        alpha: float = 0.01,
        fit_intercept: bool = True,
        learning_rate: float | None = None,
        max_iter: int = 10_000,
        tol: float = 1e-7,
        seed: RandomState = 0,
    ) -> None:
        super().__init__()
        self.alpha = _finite_non_negative(alpha, name="alpha")
        if not isinstance(fit_intercept, bool):
            msg = "fit_intercept must be a boolean"
            raise TypeError(msg)
        self.fit_intercept = fit_intercept
        self.learning_rate = (
            None
            if learning_rate is None
            else _finite_non_negative(
                learning_rate,
                name="learning_rate",
                strictly_positive=True,
            )
        )
        self.max_iter = _positive_integer(max_iter, name="max_iter")
        self.tol = _finite_non_negative(tol, name="tol", strictly_positive=True)
        self.seed = seed
        self._classes: FloatArray | None = None

    @property
    def classes_(self) -> FloatArray:
        """Sorted binary labels corresponding to probability columns."""

        self._require_fitted()
        assert self._classes is not None
        return self._classes.copy()

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Minimize regularized mean negative log likelihood."""

        validated_features, validated_targets = self._validate_fit_data(features, targets)
        classes = np.unique(validated_targets)
        if len(classes) != 2:
            msg = f"LogisticRegression requires exactly two classes; received {len(classes)}"
            raise ValueError(msg)
        encoded = np.asarray(validated_targets == classes[1], dtype=np.float64)
        if self.fit_intercept:
            design = np.column_stack(
                (validated_features, np.ones(len(validated_features), dtype=np.float64))
            )
        else:
            design = validated_features
        generator = _rng(self.seed)
        parameters = np.asarray(
            generator.normal(scale=1e-3, size=design.shape[1]), dtype=np.float64
        )
        if self.fit_intercept:
            parameters[-1] = 0.0
        spectral_term = 0.25 * float(np.linalg.norm(design, ord=2) ** 2 / len(design))
        lipschitz = spectral_term + self.alpha
        if lipschitz == 0.0:
            parameters.fill(0.0)
        if self.learning_rate is not None:
            step_size = self.learning_rate
        else:
            step_size = 1.0 if lipschitz == 0.0 else 1.0 / lipschitz
        history: list[float] = []
        for iteration in range(self.max_iter + 1):
            logits = np.asarray(design @ parameters, dtype=np.float64)
            penalty_parameters = parameters[:-1] if self.fit_intercept else parameters
            objective = float(np.mean(np.logaddexp(0.0, logits) - encoded * logits))
            objective += 0.5 * self.alpha * float(penalty_parameters @ penalty_parameters)
            if not np.isfinite(objective):
                raise ConvergenceError("LogisticRegression", iteration, tuple(history))
            history.append(objective)
            probability = _stable_sigmoid(logits)
            gradient = np.asarray(
                design.T @ (probability - encoded) / len(design), dtype=np.float64
            )
            if self.fit_intercept:
                gradient[:-1] += self.alpha * parameters[:-1]
            else:
                gradient += self.alpha * parameters
            if float(np.max(np.abs(gradient))) <= self.tol:
                if self.fit_intercept:
                    coefficients = np.asarray(parameters[:-1], dtype=np.float64)
                    intercept = float(parameters[-1])
                else:
                    coefficients = parameters.copy()
                    intercept = 0.0
                self._classes = np.asarray(classes, dtype=np.float64)
                self._publish_state(
                    coefficients=coefficients,
                    intercept=intercept,
                    loss_history=tuple(history),
                    n_iter=iteration,
                    n_features=validated_features.shape[1],
                )
                return self
            if iteration == self.max_iter:
                break
            parameters = np.asarray(parameters - step_size * gradient, dtype=np.float64)
        raise ConvergenceError("LogisticRegression", self.max_iter, tuple(history))

    def decision_function(self, features: ArrayLike) -> FloatArray:
        """Return signed logits for the fitted positive class."""

        return _LinearState.predict(self, features)

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        """Return negative- and positive-class probabilities by column."""

        positive = _stable_sigmoid(self.decision_function(features))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, features: ArrayLike) -> FloatArray:
        """Return original labels using a deterministic 0.5 threshold."""

        probabilities = self.predict_proba(features)[:, 1]
        assert self._classes is not None
        indices = (probabilities >= 0.5).astype(np.int64)
        return np.asarray(self._classes[indices], dtype=np.float64)
