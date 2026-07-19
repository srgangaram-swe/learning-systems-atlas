"""Full-covariance Gaussian mixtures fit by stable expectation-maximization."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from learning_atlas.core.reproducibility import derive_named_seed, generator_for_seed
from learning_atlas.core.validation import FloatArray
from learning_atlas.unsupervised.base import IntArray, UnsupervisedModel
from learning_atlas.unsupervised.kmeans import (
    KMeans,
    KMeansConvergenceError,
    KMeansNumericalError,
)

MixtureInitialization = Literal["kmeans", "random"]


class GaussianMixtureConvergenceError(RuntimeError):
    """Raised when one EM restart exhausts its iteration budget."""

    def __init__(
        self,
        *,
        restart: int,
        max_iter: int,
        log_likelihood_history: tuple[float, ...],
    ) -> None:
        self.restart = restart
        self.max_iter = max_iter
        self.log_likelihood_history = log_likelihood_history
        final = log_likelihood_history[-1] if log_likelihood_history else float("nan")
        super().__init__(
            "GaussianMixture EM did not converge for restart "
            f"{restart} within {max_iter} iterations "
            f"(last total log likelihood={final:.6g}); increase max_iter, relax tol, "
            "increase reg_covar, reduce n_components, or rescale the features"
        )


class GaussianMixtureNumericalError(RuntimeError):
    """Raised instead of publishing invalid mixture parameters."""


@dataclass(frozen=True, slots=True)
class _MixtureParameters:
    weights: FloatArray
    means: FloatArray
    covariances: FloatArray


@dataclass(frozen=True, slots=True)
class _MixtureRun:
    parameters: _MixtureParameters
    responsibilities: FloatArray
    labels: IntArray
    n_iter: int
    log_likelihood_history: tuple[float, ...]

    @property
    def lower_bound(self) -> float:
        return self.log_likelihood_history[-1] / len(self.labels)


def _integer_parameter(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    converted = int(value)
    if converted < minimum:
        msg = f"{name} must be at least {minimum}"
        raise ValueError(msg)
    return converted


def _finite_real_parameter(value: float, *, name: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{name} must be a real number"
        raise TypeError(msg)
    converted = float(value)
    if not np.isfinite(converted) or converted < minimum:
        msg = f"{name} must be finite and at least {minimum}"
        raise ValueError(msg)
    return converted


def _logsumexp(values: FloatArray, *, axis: int) -> FloatArray:
    maxima = np.max(values, axis=axis, keepdims=True)
    shifted = values - maxima
    totals = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = maxima + np.log(totals)
    squeezed = np.squeeze(result, axis=axis)
    if not np.all(np.isfinite(squeezed)):
        msg = "mixture log-density normalization became non-finite; rescale the features"
        raise GaussianMixtureNumericalError(msg)
    return np.asarray(squeezed, dtype=np.float64)


class GaussianMixture(UnsupervisedModel):
    """Finite Gaussian mixture with full covariance matrices and stable EM.

    Component likelihoods are evaluated through Cholesky solves in log space;
    neither covariance inverses nor raw probability products are formed.
    Restarts use order-independent named seed streams and learned state is
    published only after every requested restart converges.
    """

    def __init__(
        self,
        n_components: int = 1,
        *,
        init: MixtureInitialization = "kmeans",
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        self.n_components = _integer_parameter(n_components, name="n_components", minimum=1)
        if init not in ("kmeans", "random"):
            msg = "init must be one of: kmeans, random"
            raise ValueError(msg)
        self.init = init
        self.n_init = _integer_parameter(n_init, name="n_init", minimum=1)
        self.max_iter = _integer_parameter(max_iter, name="max_iter", minimum=1)
        self.tol = _finite_real_parameter(tol, name="tol", minimum=0.0)
        self.reg_covar = _finite_real_parameter(
            reg_covar,
            name="reg_covar",
            minimum=0.0,
        )
        self.random_state = _integer_parameter(random_state, name="random_state", minimum=0)
        if self.random_state > 2**32 - 1:
            msg = "random_state must fit in an unsigned 32-bit integer"
            raise ValueError(msg)

        self._weights: FloatArray | None = None
        self._means: FloatArray | None = None
        self._covariances: FloatArray | None = None
        self._responsibilities: FloatArray | None = None
        self._labels: IntArray | None = None
        self._n_iter: int | None = None
        self._log_likelihood_history: tuple[float, ...] | None = None
        self._restart_histories: tuple[tuple[float, ...], ...] | None = None
        self._restart_lower_bounds: tuple[float, ...] | None = None

    @property
    def weights_(self) -> FloatArray:
        self._require_fitted()
        assert self._weights is not None
        return self._weights.copy()

    @property
    def means_(self) -> FloatArray:
        self._require_fitted()
        assert self._means is not None
        return self._means.copy()

    @property
    def covariances_(self) -> FloatArray:
        self._require_fitted()
        assert self._covariances is not None
        return self._covariances.copy()

    @property
    def responsibilities_(self) -> FloatArray:
        self._require_fitted()
        assert self._responsibilities is not None
        return self._responsibilities.copy()

    @property
    def labels_(self) -> IntArray:
        self._require_fitted()
        assert self._labels is not None
        return self._labels.copy()

    @property
    def n_iter_(self) -> int:
        self._require_fitted()
        assert self._n_iter is not None
        return self._n_iter

    @property
    def converged_(self) -> bool:
        self._require_fitted()
        return True

    @property
    def lower_bound_(self) -> float:
        self._require_fitted()
        assert self._log_likelihood_history is not None
        assert self._labels is not None
        return self._log_likelihood_history[-1] / len(self._labels)

    @property
    def n_parameters_(self) -> int:
        """Number of free parameters in the fitted full-covariance mixture."""

        self._require_fitted()
        assert self.n_features_in_ is not None
        feature_count = self.n_features_in_
        covariance_parameters = feature_count * (feature_count + 1) // 2
        return (
            self.n_components
            - 1
            + self.n_components * feature_count
            + self.n_components * covariance_parameters
        )

    @property
    def log_likelihood_history_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._log_likelihood_history is not None
        return self._log_likelihood_history

    @property
    def lower_bound_history_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._log_likelihood_history is not None
        assert self._labels is not None
        sample_count = len(self._labels)
        return tuple(value / sample_count for value in self._log_likelihood_history)

    @property
    def restart_histories_(self) -> tuple[tuple[float, ...], ...]:
        self._require_fitted()
        assert self._restart_histories is not None
        return self._restart_histories

    @property
    def restart_lower_bounds_(self) -> tuple[float, ...]:
        self._require_fitted()
        assert self._restart_lower_bounds is not None
        return self._restart_lower_bounds

    def fit(self, features: ArrayLike) -> GaussianMixture:
        """Fit every EM restart and retain the maximum-likelihood solution."""

        previous_feature_count = self.n_features_in_
        try:
            validated = self._fit_features(features, min_samples=self.n_components)
            unique_count = len(np.unique(validated, axis=0))
            if unique_count < self.n_components:
                msg = (
                    f"n_components={self.n_components} exceeds the {unique_count} distinct "
                    "observations; reduce n_components"
                )
                raise ValueError(msg)
            runs = tuple(self._fit_restart(validated, restart) for restart in range(self.n_init))
            best = max(
                enumerate(runs),
                key=lambda item: (item[1].lower_bound, -item[0]),
            )[1]
        except BaseException:
            self.n_features_in_ = previous_feature_count
            raise

        self._weights = best.parameters.weights.copy()
        self._means = best.parameters.means.copy()
        self._covariances = best.parameters.covariances.copy()
        self._responsibilities = best.responsibilities.copy()
        self._labels = best.labels.copy()
        self._n_iter = best.n_iter
        self._log_likelihood_history = best.log_likelihood_history
        self._restart_histories = tuple(run.log_likelihood_history for run in runs)
        self._restart_lower_bounds = tuple(run.lower_bound for run in runs)
        return self

    def fit_predict(self, features: ArrayLike) -> IntArray:
        """Fit the mixture and return maximum-responsibility assignments."""

        self.fit(features)
        return self.labels_

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        """Return normalized component responsibilities for each observation."""

        validated = self._inference_features(features)
        parameters = self._fitted_parameters()
        _, responsibilities = self._expectation(validated, parameters)
        return responsibilities

    def predict(self, features: ArrayLike) -> IntArray:
        """Assign each observation to its highest-responsibility component."""

        return np.asarray(np.argmax(self.predict_proba(features), axis=1), dtype=np.int64)

    def score_samples(self, features: ArrayLike) -> FloatArray:
        """Return the marginal log density of every observation."""

        validated = self._inference_features(features)
        log_probabilities, _ = self._expectation(validated, self._fitted_parameters())
        return log_probabilities

    def score(self, features: ArrayLike) -> float:
        """Return mean marginal log likelihood; larger values are better."""

        return float(np.mean(self.score_samples(features)))

    def aic(self, features: ArrayLike) -> float:
        """Return Akaike's information criterion; lower values are preferred."""

        log_likelihood = float(np.sum(self.score_samples(features)))
        return float(2.0 * self.n_parameters_ - 2.0 * log_likelihood)

    def bic(self, features: ArrayLike) -> float:
        """Return Schwarz's Bayesian information criterion; lower is preferred."""

        log_probabilities = self.score_samples(features)
        return float(
            np.log(len(log_probabilities)) * self.n_parameters_ - 2.0 * np.sum(log_probabilities)
        )

    def _fitted_parameters(self) -> _MixtureParameters:
        self._require_fitted()
        assert self._weights is not None
        assert self._means is not None
        assert self._covariances is not None
        return _MixtureParameters(self._weights, self._means, self._covariances)

    def _fit_restart(self, features: FloatArray, restart: int) -> _MixtureRun:
        seed = derive_named_seed(
            self.random_state,
            "gaussian-mixture",
            self.init,
            "restart",
            str(restart),
        )
        parameters = self._initialize_parameters(features, seed)
        log_probabilities, responsibilities = self._expectation(features, parameters)
        previous_log_likelihood = float(np.sum(log_probabilities))
        if not np.isfinite(previous_log_likelihood):
            msg = "initial GaussianMixture log likelihood is non-finite; rescale the features"
            raise GaussianMixtureNumericalError(msg)
        history = [previous_log_likelihood]

        for iteration in range(1, self.max_iter + 1):
            parameters = self._maximization(features, responsibilities)
            log_probabilities, responsibilities = self._expectation(features, parameters)
            log_likelihood = float(np.sum(log_probabilities))
            if not np.isfinite(log_likelihood):
                msg = "GaussianMixture log likelihood became non-finite; rescale the features"
                raise GaussianMixtureNumericalError(msg)

            numerical_tolerance = 1e-10 * max(1.0, abs(previous_log_likelihood))
            if log_likelihood < previous_log_likelihood - numerical_tolerance:
                msg = (
                    "GaussianMixture EM decreased total log likelihood beyond numerical "
                    f"tolerance: {previous_log_likelihood:.12g} -> {log_likelihood:.12g}"
                )
                raise GaussianMixtureNumericalError(msg)
            history.append(log_likelihood)
            improvement = max(0.0, log_likelihood - previous_log_likelihood) / len(features)
            if improvement <= self.tol:
                labels = np.asarray(np.argmax(responsibilities, axis=1), dtype=np.int64)
                return _MixtureRun(
                    parameters=parameters,
                    responsibilities=responsibilities,
                    labels=labels,
                    n_iter=iteration,
                    log_likelihood_history=tuple(history),
                )
            previous_log_likelihood = log_likelihood

        raise GaussianMixtureConvergenceError(
            restart=restart,
            max_iter=self.max_iter,
            log_likelihood_history=tuple(history),
        )

    def _initialize_parameters(self, features: FloatArray, seed: int) -> _MixtureParameters:
        if self.init == "kmeans":
            try:
                labels = KMeans(
                    n_clusters=self.n_components,
                    init="k-means++",
                    n_init=1,
                    max_iter=max(100, self.max_iter),
                    tol=min(self.tol, 1e-4),
                    random_state=seed,
                ).fit_predict(features)
            except (KMeansConvergenceError, KMeansNumericalError) as error:
                msg = f"GaussianMixture k-means initialization failed: {error}"
                raise GaussianMixtureNumericalError(msg) from error
            responsibilities = np.zeros((len(features), self.n_components), dtype=np.float64)
            responsibilities[np.arange(len(features)), labels] = 1.0
            return self._maximization(features, responsibilities)

        generator = generator_for_seed(seed)
        _, unique_indices = np.unique(features, axis=0, return_index=True)
        candidates = np.asarray(np.sort(unique_indices), dtype=np.int64)
        selected = np.asarray(
            generator.choice(candidates, size=self.n_components, replace=False),
            dtype=np.int64,
        )
        means = features[selected].copy()
        centered = features - np.mean(features, axis=0)
        try:
            with np.errstate(over="raise", invalid="raise"):
                covariance = centered.T @ centered / len(features)
        except FloatingPointError as error:
            msg = "initial covariance overflowed; center or rescale the features"
            raise GaussianMixtureNumericalError(msg) from error
        covariance = np.asarray(covariance, dtype=np.float64)
        covariance.flat[:: features.shape[1] + 1] += self.reg_covar
        covariances = np.repeat(covariance[np.newaxis, :, :], self.n_components, axis=0)
        weights = np.full(self.n_components, 1.0 / self.n_components, dtype=np.float64)
        parameters = _MixtureParameters(weights, means, covariances)
        self._validate_parameters(parameters)
        return parameters

    def _maximization(
        self,
        features: FloatArray,
        responsibilities: FloatArray,
    ) -> _MixtureParameters:
        component_mass = np.asarray(np.sum(responsibilities, axis=0), dtype=np.float64)
        minimum_mass = np.finfo(np.float64).eps * len(features)
        if np.any(component_mass <= minimum_mass):
            collapsed = np.flatnonzero(component_mass <= minimum_mass).tolist()
            msg = (
                f"GaussianMixture component(s) {collapsed} collapsed during EM; increase "
                "reg_covar, reduce n_components, or choose a different seed"
            )
            raise GaussianMixtureNumericalError(msg)

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                weights = component_mass / len(features)
                means = (responsibilities.T @ features) / component_mass[:, np.newaxis]
                covariances = np.empty(
                    (self.n_components, features.shape[1], features.shape[1]),
                    dtype=np.float64,
                )
                for component in range(self.n_components):
                    difference = features - means[component]
                    weighted = difference * responsibilities[:, component, np.newaxis]
                    covariance = weighted.T @ difference / component_mass[component]
                    covariance = 0.5 * (covariance + covariance.T)
                    covariance.flat[:: features.shape[1] + 1] += self.reg_covar
                    covariances[component] = covariance
        except FloatingPointError as error:
            msg = "GaussianMixture parameter update overflowed; center or rescale the features"
            raise GaussianMixtureNumericalError(msg) from error

        parameters = _MixtureParameters(
            np.asarray(weights, dtype=np.float64),
            np.asarray(means, dtype=np.float64),
            covariances,
        )
        self._validate_parameters(parameters)
        return parameters

    def _expectation(
        self,
        features: FloatArray,
        parameters: _MixtureParameters,
    ) -> tuple[FloatArray, FloatArray]:
        weighted_log_probabilities = np.empty(
            (len(features), self.n_components),
            dtype=np.float64,
        )
        dimension_term = features.shape[1] * np.log(2.0 * np.pi)
        for component in range(self.n_components):
            covariance = parameters.covariances[component]
            try:
                cholesky = np.linalg.cholesky(covariance)
                difference = features - parameters.means[component]
                solved = np.linalg.solve(cholesky, difference.T).T
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    mahalanobis = np.sum(solved * solved, axis=1)
                    log_determinant = 2.0 * np.sum(np.log(np.diag(cholesky)))
                    weighted_log_probabilities[:, component] = -0.5 * (
                        dimension_term + log_determinant + mahalanobis
                    ) + np.log(parameters.weights[component])
            except (np.linalg.LinAlgError, FloatingPointError) as error:
                msg = (
                    f"covariance for component {component} is not numerically positive "
                    "definite; increase reg_covar or rescale the features"
                )
                raise GaussianMixtureNumericalError(msg) from error

        if not np.all(np.isfinite(weighted_log_probabilities)):
            msg = "GaussianMixture component log densities became non-finite; rescale features"
            raise GaussianMixtureNumericalError(msg)
        log_probabilities = _logsumexp(weighted_log_probabilities, axis=1)
        responsibilities = np.exp(weighted_log_probabilities - log_probabilities[:, np.newaxis])
        if not np.all(np.isfinite(responsibilities)):
            msg = "GaussianMixture responsibilities became non-finite; rescale the features"
            raise GaussianMixtureNumericalError(msg)
        return log_probabilities, np.asarray(responsibilities, dtype=np.float64)

    def _validate_parameters(self, parameters: _MixtureParameters) -> None:
        if (
            not np.all(np.isfinite(parameters.weights))
            or not np.all(np.isfinite(parameters.means))
            or not np.all(np.isfinite(parameters.covariances))
        ):
            msg = "GaussianMixture parameters became non-finite; rescale the features"
            raise GaussianMixtureNumericalError(msg)
        if np.any(parameters.weights <= 0.0) or not np.isclose(
            np.sum(parameters.weights),
            1.0,
            rtol=1e-12,
            atol=1e-12,
        ):
            msg = "GaussianMixture weights must be positive and sum to one"
            raise GaussianMixtureNumericalError(msg)
