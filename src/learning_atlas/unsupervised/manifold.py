"""Exact, transductive t-SNE implemented from first principles with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from learning_atlas.core.validation import FloatArray, validate_features
from learning_atlas.unsupervised.base import UnsupervisedModel
from learning_atlas.unsupervised.decomposition import PCA

RandomState = int | np.random.Generator


class PerplexitySearchError(RuntimeError):
    """Raised when a row cannot attain the requested entropy tolerance."""

    def __init__(
        self,
        *,
        row: int,
        target_perplexity: float,
        achieved_perplexity: float,
        iterations: int,
    ) -> None:
        self.row = row
        self.target_perplexity = target_perplexity
        self.achieved_perplexity = achieved_perplexity
        self.iterations = iterations
        super().__init__(
            f"perplexity search failed for row {row} after {iterations} iteration(s): "
            f"target={target_perplexity:.8g}, closest={achieved_perplexity:.8g}; "
            "inspect duplicate distances or increase perplexity_search_max_iter"
        )


class TSNENumericalError(RuntimeError):
    """Raised when optimization produces non-finite numerical state."""

    def __init__(self, *, iteration: int, stage: str) -> None:
        self.iteration = iteration
        self.stage = stage
        super().__init__(
            f"t-SNE produced non-finite {stage} at iteration {iteration}; "
            "rescale features or reduce learning_rate/early_exaggeration"
        )


class TSNEConvergenceError(RuntimeError):
    """Raised when strict mode cannot improve the post-exaggeration objective."""

    def __init__(
        self,
        *,
        iterations: int,
        post_exaggeration_start: float,
        final_kl: float,
        history: tuple[float, ...],
    ) -> None:
        self.iterations = iterations
        self.post_exaggeration_start = post_exaggeration_start
        self.final_kl = final_kl
        self.history = history
        super().__init__(
            "t-SNE did not improve its post-exaggeration objective after "
            f"{iterations} iteration(s): start={post_exaggeration_start:.8g}, "
            f"final={final_kl:.8g}"
        )


@dataclass(frozen=True)
class AffinityResult:
    """Stable high-dimensional affinity diagnostics."""

    conditional_probabilities: FloatArray
    joint_probabilities: FloatArray
    precisions: FloatArray
    achieved_perplexities: FloatArray


def _positive_integer(value: int, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    converted = int(value)
    minimum = 0 if allow_zero else 1
    if converted < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        msg = f"{name} must be {qualifier}"
        raise ValueError(msg)
    return converted


def _real_scalar(
    value: float,
    *,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"{name} must be a real scalar"
        raise TypeError(msg)
    converted = float(value)
    if not np.isfinite(converted):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    if lower is not None:
        valid = converted >= lower if lower_inclusive else converted > lower
        if not valid:
            operator = ">=" if lower_inclusive else ">"
            msg = f"{name} must be {operator} {lower}"
            raise ValueError(msg)
    if upper is not None and converted >= upper:
        msg = f"{name} must be < {upper}"
        raise ValueError(msg)
    return converted


def _generator(seed: RandomState) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        msg = "seed must be a non-negative integer or numpy.random.Generator"
        raise TypeError(msg)
    converted = int(seed)
    if converted < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)
    return np.random.default_rng(converted)


def pairwise_squared_distances(features: ArrayLike) -> FloatArray:
    """Return a finite, symmetric Euclidean squared-distance matrix."""

    validated = validate_features(features, min_samples=2)
    distances = np.empty((len(validated), len(validated)), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for row, observation in enumerate(validated):
            differences = validated - observation
            distances[row] = np.einsum("ij,ij->i", differences, differences)
    if not np.all(np.isfinite(distances)):
        msg = "pairwise squared distances overflowed; rescale the input features"
        raise ValueError(msg)
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, 0.0)
    return np.asarray((distances + distances.T) * 0.5, dtype=np.float64)


def _row_distribution(distances: FloatArray, precision: float) -> tuple[FloatArray, float]:
    with np.errstate(over="ignore"):
        logits = np.asarray(-precision * distances, dtype=np.float64)
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    normalizer = float(np.sum(weights))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        msg = "perplexity search produced an invalid probability normalizer"
        raise ValueError(msg)
    probabilities = np.asarray(weights / normalizer, dtype=np.float64)
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return probabilities, entropy


def _search_row_perplexity(
    distances: FloatArray,
    *,
    row: int,
    perplexity: float,
    tolerance: float,
    max_iter: int,
) -> tuple[FloatArray, float, float]:
    target_entropy = float(np.log(perplexity))
    maximum_entropy = float(np.log(len(distances)))
    if abs(target_entropy - maximum_entropy) <= tolerance:
        probabilities = np.full(len(distances), 1.0 / len(distances), dtype=np.float64)
        return probabilities, 0.0, float(len(distances))

    if float(np.ptp(distances)) == 0.0:
        raise PerplexitySearchError(
            row=row,
            target_perplexity=perplexity,
            achieved_perplexity=float(len(distances)),
            iterations=0,
        )

    precision = 1.0
    lower = 0.0
    upper = np.inf
    best_entropy = maximum_entropy
    best_error = abs(maximum_entropy - target_entropy)

    for _ in range(max_iter):
        probabilities, entropy = _row_distribution(distances, precision)
        difference = entropy - target_entropy
        error = abs(difference)
        if error < best_error:
            best_entropy = entropy
            best_error = error
        if error <= tolerance:
            return probabilities, precision, float(np.exp(entropy))
        if difference > 0.0:
            lower = precision
            precision = precision * 2.0 if np.isinf(upper) else (precision + upper) * 0.5
        else:
            upper = precision
            precision = precision * 0.5 if lower == 0.0 else (precision + lower) * 0.5

    raise PerplexitySearchError(
        row=row,
        target_perplexity=perplexity,
        achieved_perplexity=float(np.exp(best_entropy)),
        iterations=max_iter,
    )


def compute_affinities(
    features: ArrayLike,
    *,
    perplexity: float,
    tolerance: float = 1e-5,
    max_iter: int = 100,
) -> AffinityResult:
    """Compute conditional and symmetric joint t-SNE affinities."""

    validated = validate_features(features, min_samples=2)
    target_perplexity = _real_scalar(
        perplexity,
        name="perplexity",
        lower=1.0,
        lower_inclusive=True,
    )
    maximum_perplexity = validated.shape[0] - 1
    if target_perplexity > maximum_perplexity:
        msg = (
            f"perplexity={target_perplexity:.8g} exceeds the {maximum_perplexity} "
            "available neighbors per sample"
        )
        raise ValueError(msg)
    entropy_tolerance = _real_scalar(tolerance, name="tolerance", lower=0.0)
    search_iterations = _positive_integer(max_iter, name="max_iter")
    distances = pairwise_squared_distances(validated)
    n_samples = len(validated)
    conditional = np.zeros((n_samples, n_samples), dtype=np.float64)
    precisions = np.empty(n_samples, dtype=np.float64)
    achieved = np.empty(n_samples, dtype=np.float64)

    for row in range(n_samples):
        mask = np.arange(n_samples) != row
        probabilities, precision, row_perplexity = _search_row_perplexity(
            distances[row, mask],
            row=row,
            perplexity=target_perplexity,
            tolerance=entropy_tolerance,
            max_iter=search_iterations,
        )
        conditional[row, mask] = probabilities
        precisions[row] = precision
        achieved[row] = row_perplexity

    joint = np.asarray((conditional + conditional.T) / (2.0 * n_samples), dtype=np.float64)
    np.fill_diagonal(joint, 0.0)
    normalizer = float(np.sum(joint))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        msg = "symmetric t-SNE affinities have an invalid normalizer"
        raise ValueError(msg)
    joint /= normalizer
    for diagnostic in (conditional, joint, precisions, achieved):
        diagnostic.setflags(write=False)
    return AffinityResult(
        conditional_probabilities=conditional,
        joint_probabilities=joint,
        precisions=precisions,
        achieved_perplexities=achieved,
    )


def _validate_joint_probabilities(probabilities: ArrayLike, *, n_samples: int) -> FloatArray:
    try:
        array = np.asarray(probabilities)
    except ValueError as error:
        msg = "joint_probabilities must be a rectangular numeric matrix"
        raise ValueError(msg) from error
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        msg = "joint_probabilities must contain real numeric values"
        raise TypeError(msg)
    validated = np.asarray(array, dtype=np.float64)
    if validated.shape != (n_samples, n_samples):
        msg = f"joint_probabilities must have shape ({n_samples}, {n_samples})"
        raise ValueError(msg)
    if not np.all(np.isfinite(validated)) or np.any(validated < 0.0):
        msg = "joint_probabilities must be finite and non-negative"
        raise ValueError(msg)
    if not np.allclose(validated, validated.T, rtol=1e-10, atol=1e-12):
        msg = "joint_probabilities must be symmetric"
        raise ValueError(msg)
    if not np.allclose(np.diag(validated), 0.0, atol=1e-14):
        msg = "joint_probabilities must have a zero diagonal"
        raise ValueError(msg)
    if not np.isclose(float(np.sum(validated)), 1.0, rtol=1e-10, atol=1e-12):
        msg = "joint_probabilities must sum to one"
        raise ValueError(msg)
    return validated


def _student_t_distribution(embedding: FloatArray) -> tuple[FloatArray, FloatArray]:
    distances = pairwise_squared_distances(embedding)
    numerator = np.asarray(1.0 / (1.0 + distances), dtype=np.float64)
    np.fill_diagonal(numerator, 0.0)
    normalizer = float(np.sum(numerator))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        msg = "low-dimensional t-SNE affinities have an invalid normalizer"
        raise ValueError(msg)
    return np.asarray(numerator / normalizer, dtype=np.float64), numerator


def student_t_probabilities(embedding: ArrayLike) -> FloatArray:
    """Return normalized low-dimensional Student-t affinities."""

    validated = validate_features(embedding, min_samples=2)
    probabilities, _ = _student_t_distribution(validated)
    return probabilities


def _kl_from_probabilities(joint: FloatArray, low_dimensional: FloatArray) -> float:
    positive = joint > 0.0
    safe_low = np.maximum(low_dimensional[positive], np.finfo(np.float64).tiny)
    value = float(np.sum(joint[positive] * np.log(joint[positive] / safe_low)))
    if not np.isfinite(value):
        msg = "t-SNE KL divergence is non-finite"
        raise ValueError(msg)
    return value


def kl_divergence(joint_probabilities: ArrayLike, embedding: ArrayLike) -> float:
    """Evaluate the exact t-SNE KL objective for one embedding."""

    validated_embedding = validate_features(embedding, min_samples=2)
    joint = _validate_joint_probabilities(
        joint_probabilities,
        n_samples=len(validated_embedding),
    )
    low_dimensional, _ = _student_t_distribution(validated_embedding)
    return _kl_from_probabilities(joint, low_dimensional)


def _gradient_from_probabilities(
    embedding: FloatArray,
    joint: FloatArray,
    *,
    exaggeration: float,
) -> FloatArray:
    low_dimensional, numerator = _student_t_distribution(embedding)
    attractions = exaggeration * joint - low_dimensional
    weights = np.asarray(attractions * numerator, dtype=np.float64)
    row_sums = np.sum(weights, axis=1)
    gradient = 4.0 * (embedding * row_sums[:, np.newaxis] - weights @ embedding)
    return np.asarray(gradient, dtype=np.float64)


def tsne_gradient(
    embedding: ArrayLike,
    joint_probabilities: ArrayLike,
    *,
    exaggeration: float = 1.0,
) -> FloatArray:
    """Return the exact gradient of the optionally exaggerated t-SNE objective."""

    validated_embedding = validate_features(embedding, min_samples=2)
    joint = _validate_joint_probabilities(
        joint_probabilities,
        n_samples=len(validated_embedding),
    )
    factor = _real_scalar(exaggeration, name="exaggeration", lower=0.0)
    return _gradient_from_probabilities(validated_embedding, joint, exaggeration=factor)


class TSNE(UnsupervisedModel):
    """Exact O(n²), transductive t-distributed stochastic neighbor embedding.

    The estimator deliberately exposes no ``transform`` method: the learned
    coordinates belong only to the observations supplied to :meth:`fit` or
    :meth:`fit_transform`. ``converged_`` records the declared optimization
    contract—finite post-exaggeration KL improvement—not a false claim that KL
    decreases at every step or that a global optimum was found.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        perplexity: float = 30.0,
        learning_rate: float = 100.0,
        max_iter: int = 1_000,
        early_exaggeration: float = 12.0,
        early_exaggeration_iter: int = 100,
        initial_momentum: float = 0.5,
        final_momentum: float = 0.8,
        min_gain: float = 0.01,
        min_grad_norm: float = 1e-7,
        n_iter_without_progress: int = 100,
        kl_tolerance: float = 1e-7,
        perplexity_tolerance: float = 1e-5,
        perplexity_search_max_iter: int = 100,
        init: str = "pca",
        seed: RandomState = 0,
        raise_on_nonconvergence: bool = False,
    ) -> None:
        super().__init__()
        self.n_components = _positive_integer(n_components, name="n_components")
        self.perplexity = _real_scalar(
            perplexity,
            name="perplexity",
            lower=1.0,
            lower_inclusive=True,
        )
        self.learning_rate = _real_scalar(learning_rate, name="learning_rate", lower=0.0)
        self.max_iter = _positive_integer(max_iter, name="max_iter")
        self.early_exaggeration = _real_scalar(
            early_exaggeration,
            name="early_exaggeration",
            lower=1.0,
            lower_inclusive=True,
        )
        self.early_exaggeration_iter = _positive_integer(
            early_exaggeration_iter,
            name="early_exaggeration_iter",
            allow_zero=True,
        )
        if self.early_exaggeration_iter >= self.max_iter:
            msg = "early_exaggeration_iter must be smaller than max_iter"
            raise ValueError(msg)
        self.initial_momentum = _real_scalar(
            initial_momentum,
            name="initial_momentum",
            lower=0.0,
            upper=1.0,
            lower_inclusive=True,
        )
        self.final_momentum = _real_scalar(
            final_momentum,
            name="final_momentum",
            lower=0.0,
            upper=1.0,
            lower_inclusive=True,
        )
        self.min_gain = _real_scalar(min_gain, name="min_gain", lower=0.0)
        self.min_grad_norm = _real_scalar(
            min_grad_norm,
            name="min_grad_norm",
            lower=0.0,
            lower_inclusive=True,
        )
        self.n_iter_without_progress = _positive_integer(
            n_iter_without_progress,
            name="n_iter_without_progress",
        )
        self.kl_tolerance = _real_scalar(
            kl_tolerance,
            name="kl_tolerance",
            lower=0.0,
            lower_inclusive=True,
        )
        self.perplexity_tolerance = _real_scalar(
            perplexity_tolerance,
            name="perplexity_tolerance",
            lower=0.0,
        )
        self.perplexity_search_max_iter = _positive_integer(
            perplexity_search_max_iter,
            name="perplexity_search_max_iter",
        )
        if init not in {"pca", "random"}:
            msg = "init must be one of: pca, random"
            raise ValueError(msg)
        self.init = init
        _generator(seed)
        self.seed = seed
        if not isinstance(raise_on_nonconvergence, bool):
            msg = "raise_on_nonconvergence must be a boolean"
            raise TypeError(msg)
        self.raise_on_nonconvergence = raise_on_nonconvergence
        self._embedding: FloatArray | None = None
        self._conditional_probabilities: FloatArray | None = None
        self._joint_probabilities: FloatArray | None = None
        self._precisions: FloatArray | None = None
        self._perplexities: FloatArray | None = None
        self._kl_history: tuple[float, ...] | None = None
        self._gradient_norm_history: tuple[float, ...] | None = None
        self._initial_kl: float | None = None
        self._post_exaggeration_start_kl: float | None = None
        self._kl_divergence: float | None = None
        self._n_iter: int | None = None
        self._converged: bool | None = None
        self._stop_reason: str | None = None

    @property
    def embedding_(self) -> FloatArray:
        """Defensive copy of the fitted training embedding."""

        self._require_fitted()
        assert self._embedding is not None
        return self._embedding.copy()

    @property
    def conditional_probabilities_(self) -> FloatArray:
        """Per-row high-dimensional neighbor probabilities."""

        self._require_fitted()
        assert self._conditional_probabilities is not None
        return self._conditional_probabilities.copy()

    @property
    def joint_probabilities_(self) -> FloatArray:
        """Symmetric normalized high-dimensional affinities."""

        self._require_fitted()
        assert self._joint_probabilities is not None
        return self._joint_probabilities.copy()

    @property
    def precisions_(self) -> FloatArray:
        """Gaussian precision selected independently for every training row."""

        self._require_fitted()
        assert self._precisions is not None
        return self._precisions.copy()

    @property
    def perplexities_(self) -> FloatArray:
        """Achieved per-row perplexities after entropy search."""

        self._require_fitted()
        assert self._perplexities is not None
        return self._perplexities.copy()

    @property
    def kl_history_(self) -> tuple[float, ...]:
        """Unexaggerated KL divergence after each optimizer update."""

        self._require_fitted()
        assert self._kl_history is not None
        return self._kl_history

    @property
    def loss_history_(self) -> tuple[float, ...]:
        """Alias for :attr:`kl_history_` for shared reporting code."""

        return self.kl_history_

    @property
    def gradient_norm_history_(self) -> tuple[float, ...]:
        """Euclidean gradient norm after each objective evaluation."""

        self._require_fitted()
        assert self._gradient_norm_history is not None
        return self._gradient_norm_history

    @property
    def initial_kl_(self) -> float:
        """KL divergence of the named initialization."""

        self._require_fitted()
        assert self._initial_kl is not None
        return self._initial_kl

    @property
    def post_exaggeration_start_kl_(self) -> float:
        """Predeclared baseline immediately before post-exaggeration descent."""

        self._require_fitted()
        assert self._post_exaggeration_start_kl is not None
        return self._post_exaggeration_start_kl

    @property
    def kl_divergence_(self) -> float:
        """Final unexaggerated KL divergence."""

        self._require_fitted()
        assert self._kl_divergence is not None
        return self._kl_divergence

    @property
    def n_iter_(self) -> int:
        """Number of completed optimizer updates."""

        self._require_fitted()
        assert self._n_iter is not None
        return self._n_iter

    @property
    def converged_(self) -> bool:
        """Whether final KL improved from the post-exaggeration baseline."""

        self._require_fitted()
        assert self._converged is not None
        return self._converged

    @property
    def stop_reason_(self) -> str:
        """Machine-readable reason the optimizer stopped."""

        self._require_fitted()
        assert self._stop_reason is not None
        return self._stop_reason

    def fit(self, features: ArrayLike) -> Self:
        """Fit and retain a transductive embedding for the training observations."""

        self.fit_transform(features)
        return self

    def fit_transform(self, features: ArrayLike) -> FloatArray:
        """Optimize and return the transductive training embedding."""

        validated = validate_features(features, min_samples=3)
        if self.perplexity > len(validated) - 1:
            msg = (
                f"perplexity={self.perplexity:.8g} exceeds the {len(validated) - 1} "
                "available neighbors per sample"
            )
            raise ValueError(msg)
        affinities = compute_affinities(
            validated,
            perplexity=self.perplexity,
            tolerance=self.perplexity_tolerance,
            max_iter=self.perplexity_search_max_iter,
        )
        embedding = self._initialize_embedding(validated)
        velocity = np.zeros_like(embedding)
        gains = np.ones_like(embedding)
        kl_values: list[float] = []
        gradient_norms: list[float] = []
        initial_low, _ = _student_t_distribution(embedding)
        initial_kl = _kl_from_probabilities(affinities.joint_probabilities, initial_low)
        post_exaggeration_start: float | None = None
        best_post_kl = np.inf
        last_improvement = self.early_exaggeration_iter
        stop_reason = "max_iter"

        for iteration in range(self.max_iter):
            if iteration == self.early_exaggeration_iter:
                current_low, _ = _student_t_distribution(embedding)
                post_exaggeration_start = _kl_from_probabilities(
                    affinities.joint_probabilities,
                    current_low,
                )
                best_post_kl = post_exaggeration_start
                last_improvement = iteration

            exaggeration = (
                self.early_exaggeration if iteration < self.early_exaggeration_iter else 1.0
            )
            try:
                gradient = _gradient_from_probabilities(
                    embedding,
                    affinities.joint_probabilities,
                    exaggeration=exaggeration,
                )
            except ValueError as error:
                raise TSNENumericalError(iteration=iteration, stage="gradient") from error
            gradient_norm = float(np.linalg.norm(gradient))
            if not np.isfinite(gradient_norm) or not np.all(np.isfinite(gradient)):
                raise TSNENumericalError(iteration=iteration, stage="gradient")

            sign_changed = (gradient > 0.0) != (velocity > 0.0)
            gains = np.where(sign_changed, gains + 0.2, gains * 0.8)
            gains = np.maximum(gains, self.min_gain)
            momentum = (
                self.initial_momentum
                if iteration < self.early_exaggeration_iter
                else self.final_momentum
            )
            with np.errstate(over="ignore", invalid="ignore"):
                velocity = momentum * velocity - self.learning_rate * gains * gradient
                embedding = np.asarray(embedding + velocity, dtype=np.float64)
                embedding -= np.mean(embedding, axis=0)
            if not np.all(np.isfinite(embedding)) or not np.all(np.isfinite(velocity)):
                raise TSNENumericalError(iteration=iteration, stage="embedding update")

            try:
                low_dimensional, _ = _student_t_distribution(embedding)
            except ValueError as error:
                raise TSNENumericalError(
                    iteration=iteration,
                    stage="low-dimensional affinities",
                ) from error
            current_kl = _kl_from_probabilities(
                affinities.joint_probabilities,
                low_dimensional,
            )
            kl_values.append(current_kl)
            gradient_norms.append(gradient_norm)

            if iteration >= self.early_exaggeration_iter:
                if current_kl < best_post_kl - self.kl_tolerance:
                    best_post_kl = current_kl
                    last_improvement = iteration
                if gradient_norm <= self.min_grad_norm:
                    stop_reason = "min_grad_norm"
                    break
                if iteration - last_improvement >= self.n_iter_without_progress:
                    stop_reason = "no_progress"
                    break

        assert post_exaggeration_start is not None
        history = tuple(kl_values)
        final_kl = history[-1]
        converged = final_kl < post_exaggeration_start - self.kl_tolerance
        if stop_reason == "max_iter" and converged:
            stop_reason = "max_iter_with_post_exaggeration_improvement"
        elif not converged:
            stop_reason = "post_exaggeration_objective_did_not_improve"

        if not converged and self.raise_on_nonconvergence:
            raise TSNEConvergenceError(
                iterations=len(history),
                post_exaggeration_start=post_exaggeration_start,
                final_kl=final_kl,
                history=history,
            )

        self._embedding = embedding.copy()
        self._conditional_probabilities = affinities.conditional_probabilities.copy()
        self._joint_probabilities = affinities.joint_probabilities.copy()
        self._precisions = affinities.precisions.copy()
        self._perplexities = affinities.achieved_perplexities.copy()
        self._kl_history = history
        self._gradient_norm_history = tuple(gradient_norms)
        self._initial_kl = initial_kl
        self._post_exaggeration_start_kl = post_exaggeration_start
        self._kl_divergence = final_kl
        self._n_iter = len(history)
        self._converged = converged
        self._stop_reason = stop_reason
        self.n_features_in_ = validated.shape[1]
        return embedding.copy()

    def _initialize_embedding(self, features: FloatArray) -> FloatArray:
        if self.init == "random":
            return np.asarray(
                _generator(self.seed).normal(0.0, 1e-4, size=(len(features), self.n_components)),
                dtype=np.float64,
            )
        maximum_components = min(features.shape)
        if self.n_components > maximum_components:
            msg = (
                f"PCA initialization requires n_components <= {maximum_components}; "
                "use init='random' for a wider embedding"
            )
            raise ValueError(msg)
        embedding = PCA(self.n_components).fit_transform(features)
        scale = float(np.std(embedding[:, 0], ddof=1))
        if scale > 0.0:
            embedding = embedding * (1e-4 / scale)
        return np.asarray(embedding, dtype=np.float64)


__all__ = [
    "TSNE",
    "AffinityResult",
    "PerplexitySearchError",
    "TSNEConvergenceError",
    "TSNENumericalError",
    "compute_affinities",
    "kl_divergence",
    "pairwise_squared_distances",
    "student_t_probabilities",
    "tsne_gradient",
]
