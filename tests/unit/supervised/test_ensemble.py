"""Behavioral and statistical tests for from-scratch ensemble learners."""

from __future__ import annotations

import numpy as np
import pytest

from learning_atlas.core.estimators import NotFittedError
from learning_atlas.supervised.ensemble import (
    GradientBoostingBinaryClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    _binary_log_loss,
    _monotonic_step,
    _sigmoid,
)
from learning_atlas.supervised.tree import DecisionTreeClassifier, DecisionTreeRegressor

pytestmark = pytest.mark.unit


def _classification_benchmark() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(123)
    training = generator.normal(size=(240, 5))
    held_out = generator.normal(size=(300, 5))

    def labels(features: np.ndarray) -> np.ndarray:
        score = features[:, 0] * features[:, 1] + 0.5 * features[:, 2] - 0.3 * features[:, 3] ** 2
        return (score > 0.0).astype(np.float64)

    training_targets = labels(training)
    held_out_targets = labels(held_out)
    flipped = generator.choice(len(training_targets), 50, replace=False)
    training_targets[flipped] = 1.0 - training_targets[flipped]
    return training, training_targets, held_out, held_out_targets


def _regression_benchmark() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(4)
    training = generator.uniform(-3.0, 3.0, size=(180, 4))
    held_out = generator.uniform(-3.0, 3.0, size=(300, 4))

    def response(features: np.ndarray) -> np.ndarray:
        return np.sin(1.5 * features[:, 0]) + 0.4 * features[:, 1] ** 2 - 0.6 * features[:, 2]

    training_targets = response(training) + generator.normal(0.0, 0.8, len(training))
    return training, training_targets, held_out, response(held_out)


def test_random_forest_classifier_beats_single_tree_and_oob_tracks_holdout() -> None:
    x_train, y_train, x_test, y_test = _classification_benchmark()
    tree = DecisionTreeClassifier(random_state=7).fit(x_train, y_train)
    forest = RandomForestClassifier(n_estimators=45, max_features="sqrt", random_state=7).fit(
        x_train, y_train
    )

    tree_accuracy = tree.score(x_test, y_test)
    forest_accuracy = forest.score(x_test, y_test)
    assert forest_accuracy > tree_accuracy + 0.10
    assert forest_accuracy > 0.74
    assert abs(forest.oob_score_ - forest_accuracy) < 0.15
    assert forest.oob_coverage_ == 1.0
    assert np.all(np.isfinite(forest.oob_decision_function_))
    np.testing.assert_allclose(forest.predict_proba(x_test).sum(axis=1), 1.0)


def test_random_forest_regressor_reduces_noisy_single_tree_error() -> None:
    x_train, y_train, x_test, y_test = _regression_benchmark()
    tree = DecisionTreeRegressor(random_state=9).fit(x_train, y_train)
    forest = RandomForestRegressor(
        n_estimators=35,
        max_features=0.75,
        random_state=9,
    ).fit(x_train, y_train)

    tree_mse = float(np.mean((tree.predict(x_test) - y_test) ** 2))
    forest_mse = float(np.mean((forest.predict(x_test) - y_test) ** 2))
    assert forest_mse < 0.4 * tree_mse
    assert forest.score(x_test, y_test) > 0.80
    assert forest.oob_coverage_ > 0.99
    assert forest.oob_score_ > 0.50
    assert np.all(np.isfinite(forest.oob_prediction_))
    assert np.sum(forest.feature_importances_) == pytest.approx(1.0)


def test_forest_seed_replay_is_exact() -> None:
    generator = np.random.default_rng(8)
    features = generator.normal(size=(70, 4))
    targets = (features[:, 0] - features[:, 1] > 0.0).astype(np.float64)
    first = RandomForestClassifier(n_estimators=12, max_samples=0.8, random_state=22).fit(
        features, targets
    )
    second = RandomForestClassifier(n_estimators=12, max_samples=0.8, random_state=22).fit(
        features, targets
    )

    np.testing.assert_array_equal(first.predict(features), second.predict(features))
    np.testing.assert_allclose(first.predict_proba(features), second.predict_proba(features))
    np.testing.assert_allclose(first.oob_decision_function_, second.oob_decision_function_)
    assert first.oob_score_ == second.oob_score_
    assert [tree.tree_ for tree in first.estimators_] == [tree.tree_ for tree in second.estimators_]


def test_bootstrap_class_alignment_handles_trees_missing_rare_class() -> None:
    features = np.arange(16.0).reshape(8, 2)
    targets = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0])
    model = RandomForestClassifier(n_estimators=15, max_features=1, random_state=0).fit(
        features, targets
    )

    assert any(len(tree.classes_) == 1 for tree in model.estimators_)
    np.testing.assert_array_equal(model.classes_, [0.0, 9.0])
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (8, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.sum(model.feature_importances_) == pytest.approx(1.0)
    assert model.predict(features).shape == (8,)


def test_non_bootstrap_forest_exposes_disabled_oob_state() -> None:
    features = np.arange(12.0).reshape(6, 2)
    targets = np.arange(6.0)
    model = RandomForestRegressor(
        n_estimators=3,
        bootstrap=False,
        oob_score=False,
        max_features=None,
        random_state=1,
    ).fit(features, targets)

    np.testing.assert_allclose(model.predict(features), targets)
    assert model.oob_coverage_ == 0.0
    assert np.isnan(model.oob_score_)
    assert np.all(np.isnan(model.oob_prediction_))


def test_oob_edge_cases_report_coverage_instead_of_inventing_score() -> None:
    one_sample = RandomForestClassifier(n_estimators=2, random_state=3).fit([[1.0]], [4.0])
    assert one_sample.oob_coverage_ == 0.0
    assert np.isnan(one_sample.oob_score_)
    assert np.all(np.isnan(one_sample.oob_decision_function_))
    np.testing.assert_allclose(one_sample.predict_proba([[1.0]]), [[1.0]])

    constant = RandomForestRegressor(n_estimators=20, random_state=5).fit(
        np.arange(20.0).reshape(-1, 1), np.full(20, 2.0)
    )
    assert constant.oob_coverage_ > 0.95
    assert constant.oob_score_ == 1.0
    np.testing.assert_array_equal(constant.feature_importances_, [0.0])


def test_integer_bootstrap_sample_count_and_entropy_are_supported() -> None:
    features = np.arange(30.0).reshape(15, 2)
    targets = (features[:, 0] > 12.0).astype(np.float64)
    model = RandomForestClassifier(
        n_estimators=5,
        criterion="entropy",
        max_depth=3,
        max_samples=8,
        oob_score=True,
        random_state=10,
    ).fit(features, targets)
    assert len(model.estimators_) == 5
    assert 0.0 < model.oob_coverage_ <= 1.0


@pytest.mark.parametrize(
    ("factory", "error", "match"),
    [
        (lambda: RandomForestClassifier(n_estimators=0), ValueError, "n_estimators"),
        (lambda: RandomForestClassifier(n_estimators=2.5), TypeError, "n_estimators"),
        (lambda: RandomForestClassifier(criterion="variance"), ValueError, "criterion"),
        (lambda: RandomForestRegressor(criterion="gini"), ValueError, "criterion"),
        (lambda: RandomForestClassifier(bootstrap=1), TypeError, "bootstrap"),
        (lambda: RandomForestClassifier(oob_score=1), TypeError, "oob_score"),
        (lambda: RandomForestClassifier(max_samples=0), ValueError, "max_samples"),
        (lambda: RandomForestClassifier(max_samples=True), TypeError, "max_samples"),
        (lambda: RandomForestClassifier(max_samples=1.2), ValueError, "max_samples"),
        (lambda: RandomForestClassifier(max_samples=[]), TypeError, "max_samples"),
        (
            lambda: RandomForestClassifier(bootstrap=False, max_samples=2, oob_score=False),
            ValueError,
            "max_samples",
        ),
        (
            lambda: RandomForestClassifier(bootstrap=False, oob_score=True),
            ValueError,
            "requires bootstrap",
        ),
    ],
)
def test_invalid_forest_parameters_fail_early(
    factory: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        factory()  # type: ignore[operator]


def test_max_samples_cannot_exceed_training_set() -> None:
    model = RandomForestRegressor(n_estimators=2, max_samples=4)
    with pytest.raises(ValueError, match="exceeds the fitted sample count"):
        model.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    assert not model.is_fitted


def test_forest_contract_requires_fit_and_preserves_fitted_state_on_bad_refit() -> None:
    model = RandomForestRegressor(n_estimators=3, random_state=1)
    with pytest.raises(NotFittedError):
        _ = model.estimators_
    with pytest.raises(NotFittedError):
        model.predict([[0.0]])

    features = np.arange(6.0).reshape(-1, 1)
    model.fit(features, np.arange(6.0))
    expected = model.predict(features)
    with pytest.raises(ValueError, match="inconsistent samples"):
        model.fit(features, [0.0])
    np.testing.assert_allclose(model.predict(features), expected)


def test_gradient_boosting_regression_loss_is_monotonic_and_beats_stump() -> None:
    generator = np.random.default_rng(5)
    features = generator.uniform(-2.0, 2.0, size=(220, 3))
    targets = np.sin(2.0 * features[:, 0]) + features[:, 1] ** 2 - 0.5 * features[:, 2]
    stump = DecisionTreeRegressor(max_depth=1).fit(features, targets)
    model = GradientBoostingRegressor(
        n_estimators=60,
        learning_rate=0.15,
        max_depth=1,
        random_state=2,
    ).fit(features, targets)

    assert np.all(np.diff(model.train_loss_) <= 1e-14)
    assert model.train_loss_[-1] < 0.12 * model.train_loss_[0]
    assert model.score(features, targets) > stump.score(features, targets) + 0.45
    assert model.n_estimators_ == 60
    assert model.loss_history_ == model.train_loss_
    assert model.validation_loss_ == ()
    assert model.initial_prediction_ == pytest.approx(float(np.mean(targets)))
    assert np.sum(model.feature_importances_) == pytest.approx(1.0)
    assert all(0.0 < weight <= 0.15 for weight in model.stage_weights_)


def test_gradient_boosting_binary_log_loss_is_monotonic_and_beats_stump() -> None:
    generator = np.random.default_rng(6)
    features = generator.normal(size=(300, 3))
    targets = np.where(features[:, 0] ** 2 + features[:, 1] ** 2 > 1.2, 9.0, -3.0)
    stump = DecisionTreeClassifier(max_depth=1).fit(features, targets)
    model = GradientBoostingBinaryClassifier(
        n_estimators=100,
        learning_rate=0.2,
        max_depth=1,
        random_state=2,
    ).fit(features, targets)

    assert isinstance(model, GradientBoostingClassifier)
    assert np.all(np.diff(model.train_loss_) <= 1e-14)
    assert model.train_loss_[-1] < 0.55 * model.train_loss_[0]
    assert model.score(features, targets) > stump.score(features, targets) + 0.30
    probabilities = model.predict_proba(features)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.classes_, [-3.0, 9.0])
    np.testing.assert_array_equal(
        model.predict(features),
        model.classes_[(model.decision_function(features) >= 0.0).astype(int)],
    )
    assert np.sum(model.feature_importances_) == pytest.approx(1.0)
    assert model.loss_history_ == model.train_loss_


def test_early_stopping_halts_on_validation_plateau_for_both_tasks() -> None:
    features = np.ones((40, 2))
    targets = np.tile([0.0, 1.0], 20)
    regressor = GradientBoostingRegressor(
        n_estimators=50,
        early_stopping_rounds=3,
        validation_fraction=0.2,
        tol=1e-6,
        random_state=1,
    ).fit(features, targets)
    classifier = GradientBoostingClassifier(
        n_estimators=50,
        early_stopping_rounds=3,
        validation_fraction=0.2,
        tol=1e-6,
        random_state=1,
    ).fit(features, targets)

    for model in (regressor, classifier):
        assert model.n_estimators_ == 4
        assert len(model.validation_loss_) == model.n_estimators_
        assert np.all(np.diff(model.train_loss_) <= 1e-14)
        np.testing.assert_allclose(model.validation_loss_, model.validation_loss_[0])


def test_boosting_seed_replay_and_failed_refit_are_transactional() -> None:
    features = np.random.default_rng(12).normal(size=(80, 3))
    targets = (features[:, 0] + features[:, 2] > 0.0).astype(np.float64)
    first = GradientBoostingClassifier(n_estimators=12, random_state=44).fit(features, targets)
    second = GradientBoostingClassifier(n_estimators=12, random_state=44).fit(features, targets)
    np.testing.assert_allclose(
        first.decision_function(features), second.decision_function(features)
    )
    assert first.train_loss_ == second.train_loss_

    expected = first.predict(features)
    with pytest.raises(ValueError, match="exactly two"):
        first.fit(features[:3], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(first.predict(features), expected)


def test_stable_binary_probability_math_handles_extreme_log_odds() -> None:
    raw = np.asarray([-1_000.0, 0.0, 1_000.0])
    probabilities = _sigmoid(raw)
    np.testing.assert_allclose(probabilities, [0.0, 0.5, 1.0], atol=1e-15)
    assert np.isfinite(_binary_log_loss(np.asarray([0.0, 1.0, 1.0]), raw))


def test_monotonic_stage_guard_rejects_a_direction_that_never_reduces_loss() -> None:
    calls = 0

    def adversarial_loss(_observed: np.ndarray, _prediction: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 1.0

    with pytest.raises(RuntimeError, match="non-increasing loss"):
        _monotonic_step(
            np.zeros(1),
            np.ones(1),
            learning_rate=1.0,
            loss=adversarial_loss,
            observed=np.zeros(1),
        )
    assert calls == 61


@pytest.mark.parametrize(
    ("factory", "error", "match"),
    [
        (lambda: GradientBoostingRegressor(n_estimators=0), ValueError, "n_estimators"),
        (lambda: GradientBoostingRegressor(learning_rate=0.0), ValueError, "learning_rate"),
        (lambda: GradientBoostingRegressor(learning_rate=1.1), ValueError, "learning_rate"),
        (lambda: GradientBoostingRegressor(max_depth=0), ValueError, "max_depth"),
        (
            lambda: GradientBoostingRegressor(min_samples_split=1),
            ValueError,
            "min_samples_split",
        ),
        (
            lambda: GradientBoostingRegressor(min_samples_leaf=0),
            ValueError,
            "min_samples_leaf",
        ),
        (
            lambda: GradientBoostingRegressor(early_stopping_rounds=0),
            ValueError,
            "early_stopping_rounds",
        ),
        (
            lambda: GradientBoostingRegressor(validation_fraction=0.0),
            ValueError,
            "validation_fraction",
        ),
        (
            lambda: GradientBoostingRegressor(validation_fraction=1.0),
            ValueError,
            "validation_fraction",
        ),
        (lambda: GradientBoostingRegressor(tol=-1.0), ValueError, "tol"),
    ],
)
def test_invalid_boosting_parameters_fail_early(
    factory: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        factory()  # type: ignore[operator]


def test_classifier_requires_binary_targets_and_early_stop_needs_class_support() -> None:
    features = np.arange(6.0).reshape(3, 2)
    with pytest.raises(ValueError, match="exactly two"):
        GradientBoostingClassifier(n_estimators=2).fit(features, [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="two samples from each class"):
        GradientBoostingClassifier(n_estimators=2, early_stopping_rounds=1, random_state=1).fit(
            features, [0.0, 0.0, 1.0]
        )


def test_regression_early_stop_requires_train_and_validation_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        GradientBoostingRegressor(n_estimators=2, early_stopping_rounds=1).fit([[1.0]], [2.0])


def test_boosting_contract_rejects_inference_before_fit() -> None:
    regressor = GradientBoostingRegressor(n_estimators=2)
    classifier = GradientBoostingClassifier(n_estimators=2)
    with pytest.raises(NotFittedError):
        regressor.predict([[0.0]])
    with pytest.raises(NotFittedError):
        classifier.decision_function([[0.0]])
    with pytest.raises(NotFittedError):
        _ = classifier.train_loss_
