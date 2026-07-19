"""Numerical, differential, and defensive tests for Naive Bayes models."""

import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB

from learning_atlas.core.estimators import NotFittedError
from learning_atlas.supervised.naive_bayes import GaussianNB, MultinomialNB

pytestmark = pytest.mark.unit


def test_gaussian_nb_matches_sklearn_moments_predictions_and_probabilities() -> None:
    rng = np.random.default_rng(123)
    first = rng.normal(loc=[-1.0, 0.5, 2.0], scale=[0.7, 1.2, 0.5], size=(60, 3))
    second = rng.normal(loc=[1.0, -0.5, 0.0], scale=[1.1, 0.8, 1.0], size=(45, 3))
    features = np.vstack((first, second))
    targets = np.concatenate((np.full(60, -2.0), np.full(45, 8.0)))
    queries = rng.normal(size=(18, 3))
    model = GaussianNB(var_smoothing=1e-8).fit(features, targets)
    oracle = SklearnGaussianNB(var_smoothing=1e-8).fit(features, targets)

    np.testing.assert_allclose(model.theta_, oracle.theta_, atol=1e-12)
    np.testing.assert_allclose(model.var_, oracle.var_, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(model.predict(queries), oracle.predict(queries))
    np.testing.assert_allclose(
        model.predict_proba(queries), oracle.predict_proba(queries), atol=1e-12
    )
    np.testing.assert_allclose(
        np.exp(model.predict_log_proba(queries)), model.predict_proba(queries)
    )


def test_gaussian_variance_floor_handles_constant_features_and_extreme_log_scores() -> None:
    constant_features = np.ones((6, 300))
    targets = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    model = GaussianNB(var_smoothing=0.0).fit(constant_features, targets)
    probabilities = model.predict_proba(np.full((2, 300), 1e6))

    assert model.epsilon_ == np.finfo(np.float64).eps
    assert np.all(model.var_ > 0.0)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
    np.testing.assert_allclose(probabilities, [[0.5, 0.5], [0.5, 0.5]])


def test_gaussian_custom_prior_changes_an_uninformative_posterior() -> None:
    model = GaussianNB(class_prior=[0.2, 0.8]).fit(
        [[0.0], [0.0], [0.0], [0.0]],
        [3.0, 3.0, 9.0, 9.0],
    )

    np.testing.assert_allclose(model.predict_proba([[0.0]]), [[0.2, 0.8]])
    np.testing.assert_array_equal(model.predict([[0.0]]), [9.0])
    np.testing.assert_allclose(np.exp(model.class_log_prior_), [0.2, 0.8])
    np.testing.assert_array_equal(model.class_count_, [2.0, 2.0])


def test_gaussian_posterior_matches_hand_computation() -> None:
    model = GaussianNB(var_smoothing=0.0).fit(
        [[-1.0], [1.0], [3.0], [5.0]],
        [0.0, 0.0, 1.0, 1.0],
    )

    expected_first_class = 1.0 / (1.0 + np.exp(-4.0 / model.var_[0, 0]))
    np.testing.assert_allclose(
        model.predict_proba([[1.0]]),
        [[expected_first_class, 1.0 - expected_first_class]],
        atol=1e-12,
    )


def test_multinomial_nb_matches_sklearn_in_log_space() -> None:
    rng = np.random.default_rng(44)
    features = rng.poisson(lam=2.5, size=(80, 7)).astype(np.float64)
    targets = np.where(features[:, 0] + features[:, 1] > 5.0, 6.0, -4.0)
    queries = rng.poisson(lam=3.0, size=(15, 7)).astype(np.float64)
    model = MultinomialNB(alpha=0.25).fit(features, targets)
    oracle = SklearnMultinomialNB(alpha=0.25).fit(features, targets)

    np.testing.assert_allclose(model.feature_count_, oracle.feature_count_)
    np.testing.assert_allclose(model.feature_log_prob_, oracle.feature_log_prob_, atol=1e-12)
    np.testing.assert_allclose(model.class_log_prior_, oracle.class_log_prior_, atol=1e-12)
    np.testing.assert_array_equal(model.classes_, oracle.classes_)
    np.testing.assert_array_equal(model.class_count_, oracle.class_count_)
    np.testing.assert_array_equal(model.predict(queries), oracle.predict(queries))
    np.testing.assert_allclose(
        model.predict_proba(queries), oracle.predict_proba(queries), atol=1e-12
    )


def test_multinomial_laplace_smoothing_keeps_unseen_features_finite() -> None:
    model = MultinomialNB(alpha=1.0, fit_prior=False).fit(
        [[3.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 2.0, 0.0]],
        [0.0, 0.0, 1.0, 1.0],
    )

    assert np.all(np.isfinite(model.feature_log_prob_))
    np.testing.assert_allclose(np.exp(model.class_log_prior_), [0.5, 0.5])
    probability = model.predict_proba([[0.0, 0.0, 1_000.0]])
    assert np.all(np.isfinite(probability))
    np.testing.assert_allclose(np.sum(probability, axis=1), [1.0])
    np.testing.assert_array_equal(model.predict([[0.0, 0.0, 0.0]]), [0.0])


def test_multinomial_high_dimensional_posteriors_remain_normalized_in_log_space() -> None:
    generator = np.random.default_rng(2026)
    features = generator.poisson(2.0, size=(40, 1_000)).astype(np.float64)
    targets = np.repeat([0.0, 1.0], 20)
    queries = generator.poisson(50.0, size=(4, 1_000)).astype(np.float64)
    model = MultinomialNB(alpha=0.5).fit(features, targets)

    log_probability = model.predict_log_proba(queries)
    probability = model.predict_proba(queries)

    assert np.all(np.isfinite(log_probability))
    assert np.all(np.isfinite(probability))
    np.testing.assert_allclose(np.exp(log_probability), probability)
    np.testing.assert_allclose(np.sum(probability, axis=1), 1.0)


@pytest.mark.parametrize(
    ("model", "query"),
    [
        (
            GaussianNB().fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 1.0, 1.0]),
            [[1e308]],
        ),
        (
            MultinomialNB().fit(
                [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
                [0.0, 0.0, 1.0, 1.0],
            ),
            [[1e308, 1e308]],
        ),
    ],
)
def test_extreme_finite_inference_rejects_nonfinite_joint_scores(
    model: GaussianNB | MultinomialNB,
    query: list[list[float]],
) -> None:
    for inference in (model.predict_log_proba, model.predict_proba):
        with pytest.raises(ValueError, match=r"non-finite joint log likelihoods.*rescale"):
            inference(query)


@pytest.mark.parametrize(
    ("model", "features"),
    [
        (GaussianNB(), [[-1e308], [1e308], [-1e308], [1e308]]),
        (MultinomialNB(), [[1e308], [1e308], [1e308], [1e308]]),
    ],
)
def test_extreme_finite_training_rejects_nonfinite_sufficient_statistics(
    model: GaussianNB | MultinomialNB,
    features: list[list[float]],
) -> None:
    with pytest.raises(ValueError, match=r"fitting produced non-finite.*rescale"):
        model.fit(features, [0.0, 0.0, 1.0, 1.0])
    assert not model.is_fitted


def test_multinomial_posterior_matches_hand_computation() -> None:
    # With alpha=1, P(feature_0 | class_0)=3/4 and
    # P(feature_0 | class_1)=1/4; equal priors normalize to 3/4 and 1/4.
    model = MultinomialNB(alpha=1.0).fit([[2.0, 0.0], [0.0, 2.0]], [0.0, 1.0])

    np.testing.assert_allclose(model.predict_proba([[1.0, 0.0]]), [[0.75, 0.25]])


def test_multinomial_rejects_negative_counts_during_fit_and_inference() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MultinomialNB().fit([[1.0, -1.0], [0.0, 2.0]], [0.0, 1.0])

    fitted = MultinomialNB().fit([[1.0, 0.0], [0.0, 1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        fitted.predict([[0.0, -0.01]])


@pytest.mark.parametrize(
    ("prior", "error", "message"),
    [
        ([1.0], ValueError, "exactly 2"),
        ([0.5, 0.4], ValueError, "sum to 1"),
        ([1.0, 0.0], ValueError, "strictly positive"),
        ([0.5, np.inf], ValueError, "finite"),
        (["left", "right"], TypeError, "real numeric"),
        ([[0.5], [0.25, 0.25]], ValueError, "one-dimensional numeric"),
    ],
)
def test_class_prior_validation_is_strict(
    prior: list[object],
    error: type[Exception],
    message: str,
) -> None:
    model = GaussianNB(class_prior=prior)
    with pytest.raises(error, match=message):
        model.fit([[0.0], [1.0]], [0.0, 1.0])


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: GaussianNB(var_smoothing=-1.0), ValueError, "var_smoothing"),
        (lambda: GaussianNB(var_smoothing=np.inf), ValueError, "var_smoothing"),
        (lambda: MultinomialNB(alpha=0.0), ValueError, "alpha"),
        (lambda: MultinomialNB(alpha=np.nan), ValueError, "alpha"),
        (lambda: GaussianNB(var_smoothing=True), TypeError, "var_smoothing"),
        (lambda: MultinomialNB(alpha="one"), TypeError, "alpha"),
        (lambda: MultinomialNB(fit_prior=1), TypeError, "fit_prior"),
    ],
)
def test_invalid_hyperparameters_fail_early(
    factory: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        factory()  # type: ignore[operator]


def test_fitted_state_shape_and_defensive_copy_contracts() -> None:
    model = GaussianNB()
    with pytest.raises(NotFittedError, match="call fit"):
        model.predict([[0.0]])
    fitted = model.fit([[0.0, 1.0], [1.0, 0.0]], [0.0, 1.0])
    means = fitted.theta_
    means[:] = 999.0
    assert not np.all(fitted.theta_ == 999.0)
    with pytest.raises(ValueError, match="expects 2"):
        fitted.predict([[0.0]])
