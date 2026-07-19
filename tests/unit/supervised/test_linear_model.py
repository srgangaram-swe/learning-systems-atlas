"""Recovery, stability, convergence, and oracle tests for linear estimators."""

from itertools import pairwise

import numpy as np
import pytest
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.linear_model import Ridge as SklearnRidge

from learning_atlas.core.estimators import NotFittedError
from learning_atlas.supervised.datasets import make_classification, make_regression
from learning_atlas.supervised.linear_model import (
    ConvergenceError,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from learning_atlas.supervised.metrics import accuracy_score, log_loss, r2_score
from learning_atlas.supervised.preprocessing import StandardScaler, train_test_split

pytestmark = pytest.mark.unit


def test_ols_solvers_recover_truth_and_agree_on_well_conditioned_problem() -> None:
    dataset = make_regression(
        n_samples=400,
        n_features=7,
        n_informative=7,
        noise=0.0,
        bias=4.25,
        seed=13,
    )
    direct = LinearRegression(solver="lstsq").fit(dataset.features, dataset.targets)
    gradient = LinearRegression(
        solver="gradient_descent",
        max_iter=5_000,
        tol=1e-10,
        seed=29,
    ).fit(dataset.features, dataset.targets)

    np.testing.assert_allclose(direct.coef_, dataset.coefficients, atol=1e-11)
    np.testing.assert_allclose(gradient.coef_, dataset.coefficients, atol=2e-9)
    np.testing.assert_allclose(direct.coef_, gradient.coef_, atol=2e-9)
    assert direct.intercept_ == pytest.approx(dataset.intercept, abs=1e-12)
    assert gradient.intercept_ == pytest.approx(dataset.intercept, abs=1e-10)
    assert direct.n_iter_ == 1
    assert gradient.n_iter_ < gradient.max_iter
    assert direct.converged_ is True
    assert r2_score(dataset.targets, gradient.predict(dataset.features)) == pytest.approx(1.0)


def test_gradient_descent_is_seed_reproducible_and_objective_is_monotone() -> None:
    dataset = make_regression(n_samples=160, n_features=5, noise=0.2, seed=4)
    first = LinearRegression(solver="gradient_descent", tol=1e-9, seed=17).fit(
        dataset.features, dataset.targets
    )
    second = LinearRegression(solver="gradient_descent", tol=1e-9, seed=17).fit(
        dataset.features, dataset.targets
    )

    np.testing.assert_array_equal(first.coef_, second.coef_)
    assert first.loss_history_ == second.loss_history_
    assert all(current <= previous + 1e-14 for previous, current in pairwise(first.loss_history_))


def test_ols_without_intercept_and_degenerate_features_have_defined_behavior() -> None:
    features = np.zeros((5, 2), dtype=np.float64)
    targets = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    no_intercept = LinearRegression(fit_intercept=False).fit(features, targets)
    gradient = LinearRegression(solver="gradient_descent").fit(features, targets)

    assert no_intercept.intercept_ == 0.0
    np.testing.assert_array_equal(no_intercept.coef_, [0.0, 0.0])
    np.testing.assert_array_equal(gradient.coef_, [0.0, 0.0])
    np.testing.assert_array_equal(gradient.predict([[0.0, 0.0]]), [3.0])
    assert gradient.n_iter_ == 0


def test_ridge_alpha_zero_is_ols_and_norm_shrinks_monotonically() -> None:
    dataset = make_regression(n_samples=220, n_features=8, noise=1.0, bias=2.0, seed=12)
    ols = LinearRegression().fit(dataset.features, dataset.targets)
    models = [
        Ridge(alpha=alpha).fit(dataset.features, dataset.targets) for alpha in (0.0, 1.0, 20.0)
    ]

    np.testing.assert_allclose(models[0].coef_, ols.coef_, atol=1e-12)
    np.testing.assert_allclose(models[0].predict(dataset.features), ols.predict(dataset.features))
    norms = [float(np.linalg.norm(model.coef_)) for model in models]
    assert norms[0] >= norms[1] >= norms[2]


def test_ridge_matches_sklearn_oracle_and_never_penalizes_intercept() -> None:
    dataset = make_regression(n_samples=180, n_features=6, noise=0.4, bias=75.0, seed=8)
    model = Ridge(alpha=3.5).fit(dataset.features, dataset.targets)
    oracle = SklearnRidge(alpha=3.5).fit(dataset.features, dataset.targets)

    np.testing.assert_allclose(model.coef_, oracle.coef_, rtol=1e-11, atol=1e-11)
    assert model.intercept_ == pytest.approx(float(oracle.intercept_), abs=1e-11)
    assert model.intercept_ == pytest.approx(75.0, abs=0.1)


def test_lasso_sets_irrelevant_features_exactly_to_zero_and_matches_oracle() -> None:
    dataset = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=3,
        coefficient_scale=4.0,
        noise=0.05,
        shuffle=False,
        seed=10,
    )
    alpha = 0.1
    model = Lasso(alpha=alpha, tol=1e-10).fit(dataset.features, dataset.targets)
    oracle = SklearnLasso(alpha=alpha, max_iter=10_000, tol=1e-10).fit(
        dataset.features, dataset.targets
    )

    np.testing.assert_array_equal(model.coef_[3:], 0.0)
    np.testing.assert_allclose(model.coef_, oracle.coef_, atol=2e-8)
    assert model.n_iter_ < model.max_iter
    assert all(current <= previous + 1e-12 for previous, current in pairwise(model.loss_history_))

    residual = model.predict(dataset.features) - dataset.targets
    smooth_gradient = np.asarray(dataset.features.T @ residual / len(residual), dtype=np.float64)
    active = model.coef_ != 0.0
    assert np.any(active)
    assert np.any(~active)
    np.testing.assert_allclose(
        smooth_gradient[active],
        -alpha * np.sign(model.coef_[active]),
        rtol=0.0,
        atol=1e-8,
    )
    assert np.all(np.abs(smooth_gradient[~active]) <= alpha + 1e-8)
    assert float(np.mean(residual)) == pytest.approx(0.0, abs=1e-12)


def test_lasso_alpha_zero_reduces_exactly_to_ols() -> None:
    dataset = make_regression(n_samples=100, n_features=5, noise=0.2, seed=14)
    lasso = Lasso(alpha=0.0).fit(dataset.features, dataset.targets)
    ols = LinearRegression().fit(dataset.features, dataset.targets)

    np.testing.assert_array_equal(lasso.coef_, ols.coef_)
    assert lasso.intercept_ == ols.intercept_
    assert lasso.loss_history_ == ols.loss_history_


def test_logistic_regression_beats_accuracy_and_log_loss_baselines() -> None:
    dataset = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=7,
        class_sep=0.8,
        seed=20,
    )
    split = train_test_split(
        dataset.features,
        dataset.targets,
        test_size=0.25,
        seed=31,
        stratify=dataset.targets,
    )
    scaler = StandardScaler().fit(split.x_train)
    x_train = scaler.transform(split.x_train)
    x_test = scaler.transform(split.x_test)
    model = LogisticRegression(alpha=0.01, tol=1e-7, seed=7).fit(x_train, split.y_train)
    predictions = model.predict(x_test)
    probability = model.predict_proba(x_test)
    prevalence = float(np.mean(split.y_train))

    assert accuracy_score(split.y_test, predictions) >= 0.90
    assert accuracy_score(split.y_test, predictions) > max(prevalence, 1.0 - prevalence) + 0.30
    assert log_loss(split.y_test, probability[:, 1]) < log_loss(
        split.y_test, np.full(len(split.y_test), prevalence)
    )
    np.testing.assert_allclose(np.sum(probability, axis=1), 1.0, atol=1e-15)
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_logistic_probabilities_agree_with_independent_oracle() -> None:
    dataset = make_classification(n_samples=400, n_features=6, class_sep=0.5, flip_y=0.03, seed=2)
    features = StandardScaler().fit_transform(dataset.features)
    alpha = 0.025
    model = LogisticRegression(alpha=alpha, tol=1e-8, max_iter=20_000, seed=3).fit(
        features, dataset.targets
    )
    oracle = SklearnLogisticRegression(
        C=1.0 / (alpha * len(features)),
        solver="lbfgs",
        tol=1e-10,
        max_iter=20_000,
    ).fit(features, dataset.targets)

    correlation = np.corrcoef(
        model.predict_proba(features)[:, 1], oracle.predict_proba(features)[:, 1]
    )[0, 1]
    assert correlation > 0.999999
    np.testing.assert_allclose(model.coef_, oracle.coef_[0], atol=2e-5)
    assert model.intercept_ == pytest.approx(float(oracle.intercept_[0]), abs=2e-5)


def test_logistic_supports_nonstandard_binary_labels_and_extreme_logits_without_overflow() -> None:
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    targets = np.asarray([-7.0, -7.0, 4.0, 4.0])
    model = LogisticRegression(alpha=0.1, tol=1e-8).fit(features, targets)

    with np.errstate(over="raise", invalid="raise"):
        probability = model.predict_proba([[-1e9], [1e9]])
    np.testing.assert_array_equal(model.classes_, [-7.0, 4.0])
    np.testing.assert_array_equal(model.predict([[-1e9], [1e9]]), [-7.0, 4.0])
    np.testing.assert_allclose(np.sum(probability, axis=1), 1.0)
    assert probability[0, 1] == 0.0
    assert probability[1, 1] == 1.0


def test_logistic_training_is_seed_deterministic() -> None:
    dataset = make_classification(n_samples=120, n_features=4, seed=17)
    first = LogisticRegression(seed=99).fit(dataset.features, dataset.targets)
    second = LogisticRegression(seed=99).fit(dataset.features, dataset.targets)

    np.testing.assert_array_equal(first.coef_, second.coef_)
    assert first.intercept_ == second.intercept_
    assert first.loss_history_ == second.loss_history_


def test_logistic_without_intercept_handles_zero_design_and_regularization() -> None:
    features = np.zeros((4, 2), dtype=np.float64)
    targets = np.asarray([0.0, 1.0, 0.0, 1.0])
    model = LogisticRegression(alpha=0.0, fit_intercept=False).fit(features, targets)

    np.testing.assert_array_equal(model.coef_, [0.0, 0.0])
    assert model.intercept_ == 0.0
    np.testing.assert_array_equal(model.predict_proba([[0.0, 0.0]]), [[0.5, 0.5]])
    assert model.n_iter_ == 0


@pytest.mark.parametrize(
    "estimator",
    [LinearRegression(), Ridge(), Lasso(), LogisticRegression()],
)
def test_estimators_enforce_fitted_and_feature_shape_contracts(estimator: object) -> None:
    with pytest.raises(NotFittedError, match="not fitted"):
        _ = estimator.coef_  # type: ignore[attr-defined]
    with pytest.raises(NotFittedError, match="not fitted"):
        _ = estimator.intercept_  # type: ignore[attr-defined]
    with pytest.raises(NotFittedError, match="not fitted"):
        _ = estimator.loss_history_  # type: ignore[attr-defined]
    with pytest.raises(NotFittedError, match="not fitted"):
        _ = estimator.n_iter_  # type: ignore[attr-defined]
    with pytest.raises(NotFittedError, match="not fitted"):
        _ = estimator.converged_  # type: ignore[attr-defined]
    fitted = estimator.fit([[0.0], [1.0]], [0.0, 1.0])  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="expects 1"):
        fitted.predict([[0.0, 1.0]])


def test_exported_coefficients_are_defensive_copies() -> None:
    model = LinearRegression().fit([[0.0], [1.0]], [1.0, 3.0])
    exported = model.coef_
    exported[0] = 999.0
    assert model.coef_[0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "model",
    [
        LinearRegression(solver="gradient_descent", max_iter=1, tol=1e-30),
        Lasso(alpha=0.01, max_iter=1, tol=1e-30),
        LogisticRegression(max_iter=1, tol=1e-30),
    ],
)
def test_nonconvergence_raises_with_history_and_does_not_publish_state(model: object) -> None:
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    targets = np.asarray([0.0, 0.0, 1.0, 1.0])
    with pytest.raises(ConvergenceError) as captured:
        model.fit(features, targets)  # type: ignore[attr-defined]

    assert captured.value.iterations == 1
    assert len(captured.value.loss_history) >= 1
    assert model.is_fitted is False  # type: ignore[attr-defined]


def test_logistic_rejects_nonbinary_targets() -> None:
    with pytest.raises(ValueError, match="exactly two classes"):
        LogisticRegression().fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="exactly two classes"):
        LogisticRegression().fit([[0.0], [1.0]], [1.0, 1.0])


@pytest.mark.parametrize(
    ("constructor", "kwargs", "exception", "message"),
    [
        (LinearRegression, {"solver": "magic"}, ValueError, "solver"),
        (LinearRegression, {"fit_intercept": 1}, TypeError, "boolean"),
        (LinearRegression, {"learning_rate": 0.0}, ValueError, "positive"),
        (LinearRegression, {"max_iter": 0}, ValueError, "positive"),
        (LinearRegression, {"tol": np.nan}, ValueError, "positive"),
        (Ridge, {"alpha": -1.0}, ValueError, "non-negative"),
        (Ridge, {"fit_intercept": "yes"}, TypeError, "boolean"),
        (Lasso, {"max_iter": True}, TypeError, "integer"),
        (LogisticRegression, {"alpha": np.inf}, ValueError, "non-negative"),
        (LogisticRegression, {"learning_rate": "fast"}, TypeError, "real scalar"),
    ],
)
def test_hyperparameter_validation_is_actionable(
    constructor: object,
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        constructor(**kwargs)  # type: ignore[operator]
