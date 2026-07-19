"""Numerical, probabilistic, and failure-contract tests for Gaussian-mixture EM."""

import pickle
from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as ReferenceGaussianMixture

from learning_atlas.unsupervised.base import NotFittedError
from learning_atlas.unsupervised.mixture import (
    GaussianMixture,
    GaussianMixtureConvergenceError,
    GaussianMixtureNumericalError,
)

pytestmark = pytest.mark.unit


def _mixture_data(
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    means = np.array([[-4.0, -1.0], [0.0, 4.0], [4.0, 0.0]], dtype=np.float64)
    covariances = np.array(
        [
            [[1.0, 0.45], [0.45, 0.8]],
            [[0.7, -0.2], [-0.2, 1.2]],
            [[1.1, 0.2], [0.2, 0.5]],
        ],
        dtype=np.float64,
    )
    counts = np.array([150, 210, 240], dtype=np.int64)
    features = np.vstack(
        [
            generator.multivariate_normal(mean, covariance, size=int(count))
            for mean, covariance, count in zip(means, covariances, counts, strict=True)
        ]
    )
    labels = np.repeat(np.arange(3, dtype=np.int64), counts)
    return features, labels, means, counts / np.sum(counts)


def _row_order(values: np.ndarray) -> np.ndarray:
    return np.lexsort(tuple(values[:, column] for column in reversed(range(values.shape[1]))))


def test_em_recovers_mixture_and_publishes_normalized_probabilities() -> None:
    features, truth, expected_means, expected_weights = _mixture_data()
    model = GaussianMixture(
        3,
        n_init=3,
        max_iter=300,
        tol=1e-6,
        reg_covar=1e-6,
        random_state=42,
    )
    fitted_labels = model.fit_predict(features)
    order = _row_order(model.means_)

    np.testing.assert_allclose(model.means_[order], expected_means, atol=0.2)
    np.testing.assert_allclose(model.weights_[order], expected_weights, atol=0.02)
    expected_covariances = np.array(
        [
            [[1.0, 0.45], [0.45, 0.8]],
            [[0.7, -0.2], [-0.2, 1.2]],
            [[1.1, 0.2], [0.2, 0.5]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(model.covariances_[order], expected_covariances, atol=0.12)
    assert adjusted_rand_score(truth, model.labels_) > 0.99
    np.testing.assert_array_equal(fitted_labels, model.labels_)
    np.testing.assert_array_equal(model.predict(features), model.labels_)
    probabilities = model.predict_proba(features)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(probabilities, model.responsibilities_, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(model.covariances_) > 0.0)
    assert model.converged_ is True
    assert np.isfinite(model.score(features))
    np.testing.assert_allclose(model.score_samples(features).mean(), model.lower_bound_)


def test_em_log_likelihood_is_nondecreasing_for_every_restart() -> None:
    features, _, _, _ = _mixture_data(seed=7)
    model = GaussianMixture(3, n_init=4, max_iter=300, tol=1e-7, random_state=19).fit(features)

    for history in model.restart_histories_:
        assert len(history) >= 2
        assert all(
            current >= previous - 1e-9 * max(1.0, abs(previous))
            for previous, current in pairwise(history)
        )
    assert len(model.log_likelihood_history_) == model.n_iter_ + 1
    np.testing.assert_allclose(
        model.lower_bound_history_,
        np.asarray(model.log_likelihood_history_) / len(features),
    )
    assert model.lower_bound_ == max(model.restart_lower_bounds_)


def test_gaussian_mixture_agrees_with_independent_sklearn_oracle() -> None:
    features, _, _, _ = _mixture_data(seed=13)
    actual = GaussianMixture(
        3,
        n_init=4,
        max_iter=300,
        tol=1e-7,
        reg_covar=1e-6,
        random_state=9,
    ).fit(features)
    reference = ReferenceGaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=4,
        max_iter=300,
        tol=1e-7,
        reg_covar=1e-6,
        random_state=9,
        init_params="kmeans",
    ).fit(features)

    assert adjusted_rand_score(reference.predict(features), actual.labels_) > 0.99
    assert actual.score(features) == pytest.approx(float(reference.score(features)), abs=2e-3)


def test_single_component_score_matches_closed_form_density() -> None:
    generator = np.random.default_rng(4)
    features = generator.multivariate_normal(
        [1.5, -2.0],
        [[1.2, 0.35], [0.35, 0.8]],
        size=180,
    )
    model = GaussianMixture(1, max_iter=100, tol=1e-10, reg_covar=1e-7).fit(features)
    difference = features - model.means_[0]
    covariance = model.covariances_[0]
    sign, log_determinant = np.linalg.slogdet(covariance)
    assert sign == 1.0
    mahalanobis = np.einsum(
        "ni,ij,nj->n",
        difference,
        np.linalg.solve(covariance, np.eye(features.shape[1])),
        difference,
    )
    expected = -0.5 * (features.shape[1] * np.log(2.0 * np.pi) + log_determinant + mahalanobis)

    np.testing.assert_allclose(model.score_samples(features), expected, rtol=1e-10, atol=1e-10)


def test_information_criteria_match_parameter_count_definitions() -> None:
    features, _, _, _ = _mixture_data(seed=33)
    model = GaussianMixture(3, n_init=2, max_iter=300, random_state=5).fit(features)
    log_likelihood = float(np.sum(model.score_samples(features)))
    feature_count = features.shape[1]
    expected_parameters = (
        model.n_components
        - 1
        + model.n_components * feature_count
        + model.n_components * feature_count * (feature_count + 1) // 2
    )

    assert model.n_parameters_ == expected_parameters
    assert model.aic(features) == pytest.approx(2.0 * expected_parameters - 2.0 * log_likelihood)
    assert model.bic(features) == pytest.approx(
        np.log(len(features)) * expected_parameters - 2.0 * log_likelihood
    )

    unfitted = GaussianMixture(2)
    with pytest.raises(NotFittedError, match="not fitted"):
        _ = unfitted.n_parameters_
    with pytest.raises(NotFittedError, match="not fitted"):
        unfitted.aic(features)


def test_covariance_regularization_handles_collinear_observations() -> None:
    coordinate = np.linspace(-3.0, 3.0, 100)
    features = np.column_stack((coordinate, 2.0 * coordinate))
    regularized = GaussianMixture(
        2,
        reg_covar=1e-5,
        max_iter=300,
        tol=1e-6,
        random_state=1,
    ).fit(features)

    assert np.all(np.linalg.eigvalsh(regularized.covariances_) >= 1e-5 - 1e-12)
    assert np.all(np.isfinite(regularized.score_samples(features)))
    with pytest.raises(GaussianMixtureNumericalError, match="reg_covar"):
        GaussianMixture(2, reg_covar=0.0, random_state=1).fit(features)


def test_random_initialization_is_reproducible_and_seed_local() -> None:
    features, _, _, _ = _mixture_data(seed=21)
    first = GaussianMixture(
        3,
        init="random",
        n_init=2,
        max_iter=300,
        tol=1e-5,
        random_state=31,
    ).fit(features)
    replay = GaussianMixture(
        3,
        init="random",
        n_init=2,
        max_iter=300,
        tol=1e-5,
        random_state=31,
    ).fit(features)

    np.testing.assert_array_equal(first.means_, replay.means_)
    assert first.restart_histories_ == replay.restart_histories_


def test_restart_zero_is_invariant_when_more_restarts_are_requested() -> None:
    features, _, _, _ = _mixture_data(seed=10)
    one = GaussianMixture(3, n_init=1, max_iter=300, tol=1e-7, random_state=7).fit(features)
    three = GaussianMixture(3, n_init=3, max_iter=300, tol=1e-7, random_state=7).fit(features)

    assert one.restart_histories_[0] == three.restart_histories_[0]
    assert one.restart_lower_bounds_[0] == three.restart_lower_bounds_[0]
    exposed = three.covariances_
    exposed[:] = 0.0
    assert not np.all(three.covariances_ == 0.0)


def test_serialization_and_public_arrays_preserve_owned_fitted_state() -> None:
    features, _, _, _ = _mixture_data(seed=44)
    model = GaussianMixture(3, n_init=2, max_iter=300, random_state=23).fit(features)
    restored = pickle.loads(pickle.dumps(model, protocol=5))

    np.testing.assert_array_equal(restored.weights_, model.weights_)
    np.testing.assert_array_equal(restored.means_, model.means_)
    np.testing.assert_array_equal(restored.covariances_, model.covariances_)
    np.testing.assert_array_equal(restored.responsibilities_, model.responsibilities_)
    np.testing.assert_array_equal(restored.predict(features), model.predict(features))
    assert restored.restart_histories_ == model.restart_histories_

    public_arrays = (
        model.weights_,
        model.means_,
        model.covariances_,
        model.responsibilities_,
        model.labels_,
    )
    for exposed in public_arrays:
        exposed[...] = 0
    assert np.any(model.weights_ != 0.0)
    assert np.any(model.means_ != 0.0)
    assert np.any(model.covariances_ != 0.0)
    assert np.any(model.responsibilities_ != 0.0)
    assert np.any(model.labels_ != 0)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: GaussianMixture(0), ValueError, "n_components"),
        (lambda: GaussianMixture(2, n_init=0), ValueError, "n_init"),
        (lambda: GaussianMixture(2, max_iter=0), ValueError, "max_iter"),
        (lambda: GaussianMixture(2, tol=-1.0), ValueError, "tol"),
        (lambda: GaussianMixture(2, tol=True), TypeError, "real number"),
        (lambda: GaussianMixture(2, reg_covar=-1.0), ValueError, "reg_covar"),
        (
            lambda: GaussianMixture(2, init="spectral"),  # type: ignore[arg-type]
            ValueError,
            "init",
        ),
        (lambda: GaussianMixture(False), TypeError, "integer"),
        (lambda: GaussianMixture(2, random_state=2**32), ValueError, "unsigned"),
    ],
)
def test_invalid_hyperparameters_fail_early(
    factory: Callable[[], object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        factory()


def test_fit_and_inference_reject_invalid_inputs_and_refit_is_atomic() -> None:
    model = GaussianMixture(2)
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict_proba([[0.0]])
    with pytest.raises(ValueError, match="2D"):
        model.fit([0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        model.fit([[0.0], [np.inf]])
    with pytest.raises(ValueError, match="distinct"):
        model.fit(np.ones((3, 1), dtype=np.float64))

    features, _, _, _ = _mixture_data(seed=5)
    model.fit(features)
    before = model.means_
    with pytest.raises(ValueError, match="expects 2"):
        model.score_samples([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="distinct"):
        model.fit(np.ones((4, 3), dtype=np.float64))
    assert model.n_features_in_ == 2
    np.testing.assert_array_equal(model.means_, before)


def test_nonconvergence_exposes_history_without_publishing_state() -> None:
    generator = np.random.default_rng(8)
    features = np.concatenate((generator.normal(-1.0, 1.5, 150), generator.normal(1.0, 1.5, 150)))[
        :, np.newaxis
    ]
    model = GaussianMixture(2, max_iter=1, tol=0.0, random_state=0)

    with pytest.raises(GaussianMixtureConvergenceError, match="increase max_iter") as captured:
        model.fit(features)

    assert captured.value.restart == 0
    assert captured.value.max_iter == 1
    assert len(captured.value.log_likelihood_history) == 2
    assert model.n_features_in_ is None
    with pytest.raises(NotFittedError):
        _ = model.weights_


def test_collapsed_components_raise_instead_of_publishing_invalid_weights() -> None:
    model = GaussianMixture(2)
    features = np.array([[-1.0], [0.0], [1.0]], dtype=np.float64)
    responsibilities = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    with pytest.raises(GaussianMixtureNumericalError, match="collapsed"):
        model._maximization(features, responsibilities)


@pytest.mark.parametrize("init", ["kmeans", "random"])
def test_extreme_finite_scale_raises_actionable_numerical_error(init: str) -> None:
    features = np.array([[-1e308, 0.0], [1e308, 0.0]], dtype=np.float64)

    with pytest.raises(GaussianMixtureNumericalError, match="rescale"):
        GaussianMixture(2, init=init, random_state=0).fit(features)  # type: ignore[arg-type]
