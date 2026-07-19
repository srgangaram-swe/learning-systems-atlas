"""Mathematical, contract, and determinism tests for from-scratch CART."""

from __future__ import annotations

import numpy as np
import pytest

from learning_atlas.core.estimators import NotFittedError
from learning_atlas.supervised.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    TreeNode,
)

pytestmark = pytest.mark.unit


def test_classifier_perfectly_fits_axis_aligned_data_with_both_criteria() -> None:
    features = np.asarray(
        [[0.0, 2.0], [0.2, -1.0], [0.8, 4.0], [1.2, 0.0], [1.8, 5.0], [2.0, -2.0]]
    )
    targets = np.asarray([-3.0, -3.0, -3.0, 7.0, 7.0, 7.0])

    for criterion in ("gini", "entropy"):
        model = DecisionTreeClassifier(criterion=criterion).fit(features, targets)
        np.testing.assert_array_equal(model.predict(features), targets)
        np.testing.assert_allclose(model.predict_proba(features).sum(axis=1), 1.0)
        np.testing.assert_array_equal(model.classes_, [-3.0, 7.0])
        assert model.score(features, targets) == 1.0
        assert model.depth_ == 1
        assert model.n_leaves_ == 2
        np.testing.assert_allclose(model.feature_importances_, [1.0, 0.0])


def test_regressor_perfect_fit_and_depth_limit_have_expected_bias() -> None:
    features = np.arange(8.0).reshape(-1, 1)
    targets = np.asarray([0.0, 0.0, 1.0, 1.0, 4.0, 4.0, 9.0, 9.0])
    unrestricted = DecisionTreeRegressor().fit(features, targets)
    stump = DecisionTreeRegressor(max_depth=0).fit(features, targets)

    np.testing.assert_allclose(unrestricted.predict(features), targets)
    np.testing.assert_allclose(stump.predict(features), np.mean(targets))
    assert unrestricted.score(features, targets) == 1.0
    assert stump.score(features, targets) == 0.0
    assert unrestricted.depth_ > stump.depth_ == 0
    assert unrestricted.n_leaves_ == 4
    assert stump.n_leaves_ == 1


def test_classifier_depth_limit_resists_noisy_training_labels_on_clean_holdout() -> None:
    generator = np.random.default_rng(2026)
    training_features = np.sort(generator.uniform(-3.0, 3.0, 240))[:, np.newaxis]
    clean_training_targets = (training_features[:, 0] > 0.0).astype(np.float64)
    noisy_training_targets = clean_training_targets.copy()
    flipped = generator.choice(len(noisy_training_targets), size=48, replace=False)
    noisy_training_targets[flipped] = 1.0 - noisy_training_targets[flipped]
    holdout_features = np.linspace(-3.0, 3.0, 1_201)[:, np.newaxis]
    holdout_targets = (holdout_features[:, 0] > 0.0).astype(np.float64)

    depth_limited = DecisionTreeClassifier(max_depth=1).fit(
        training_features, noisy_training_targets
    )
    unrestricted = DecisionTreeClassifier().fit(training_features, noisy_training_targets)

    assert unrestricted.score(training_features, noisy_training_targets) == 1.0
    assert unrestricted.score(training_features, noisy_training_targets) > depth_limited.score(
        training_features, noisy_training_targets
    )
    assert depth_limited.score(holdout_features, holdout_targets) > 0.98
    assert depth_limited.score(holdout_features, holdout_targets) > (
        unrestricted.score(holdout_features, holdout_targets) + 0.20
    )
    assert depth_limited.depth_ == 1
    assert unrestricted.depth_ > depth_limited.depth_


def test_regressor_depth_limit_reduces_noise_interpolation_error_on_clean_holdout() -> None:
    generator = np.random.default_rng(2027)
    training_features = np.sort(generator.uniform(-3.0, 3.0, 240))[:, np.newaxis]

    def response(values: np.ndarray) -> np.ndarray:
        return np.sin(1.5 * values) + 0.2 * values

    clean_training_targets = response(training_features[:, 0])
    noisy_training_targets = clean_training_targets + generator.normal(
        0.0, 0.65, len(clean_training_targets)
    )
    holdout_features = np.linspace(-3.0, 3.0, 1_201)[:, np.newaxis]
    holdout_targets = response(holdout_features[:, 0])

    depth_limited = DecisionTreeRegressor(max_depth=3).fit(
        training_features, noisy_training_targets
    )
    unrestricted = DecisionTreeRegressor().fit(training_features, noisy_training_targets)
    limited_train_rmse = float(
        np.sqrt(np.mean((depth_limited.predict(training_features) - noisy_training_targets) ** 2))
    )
    unrestricted_train_rmse = float(
        np.sqrt(np.mean((unrestricted.predict(training_features) - noisy_training_targets) ** 2))
    )
    limited_holdout_rmse = float(
        np.sqrt(np.mean((depth_limited.predict(holdout_features) - holdout_targets) ** 2))
    )
    unrestricted_holdout_rmse = float(
        np.sqrt(np.mean((unrestricted.predict(holdout_features) - holdout_targets) ** 2))
    )

    assert unrestricted_train_rmse == pytest.approx(0.0, abs=1e-15)
    assert unrestricted_train_rmse < limited_train_rmse
    assert limited_holdout_rmse < 0.60 * unrestricted_holdout_rmse
    assert depth_limited.depth_ == 3
    assert unrestricted.depth_ > depth_limited.depth_


def test_equal_gain_splits_choose_lowest_feature_then_threshold() -> None:
    duplicated_features = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    classifier = DecisionTreeClassifier(max_depth=1).fit(duplicated_features, [0.0, 0.0, 1.0, 1.0])
    assert classifier.tree_.feature_index == 0
    assert classifier.tree_.threshold == pytest.approx(1.5)

    equal_threshold_gain = DecisionTreeClassifier(max_depth=1).fit(
        np.arange(3.0).reshape(-1, 1), [0.0, 1.0, 0.0]
    )
    assert equal_threshold_gain.tree_.threshold == pytest.approx(0.5)


@pytest.mark.parametrize("tree_type", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_adjacent_float_threshold_preserves_a_strict_partition(tree_type: type[object]) -> None:
    lower = np.nextafter(1.0, np.inf)
    upper = np.nextafter(lower, np.inf)
    features = np.asarray([[lower], [upper]])
    model = tree_type().fit(features, [0.0, 1.0])  # type: ignore[attr-defined]

    np.testing.assert_array_equal(model.predict(features), [0.0, 1.0])  # type: ignore[attr-defined]
    assert model.tree_.threshold == lower  # type: ignore[attr-defined]


def test_leaf_probabilities_and_class_ties_follow_sorted_class_order() -> None:
    features = np.ones((4, 2))
    model = DecisionTreeClassifier().fit(features, [8.0, -2.0, 8.0, -2.0])
    np.testing.assert_array_equal(model.predict(features), -2.0)
    np.testing.assert_allclose(model.predict_proba(features), 0.5)
    assert model.tree_.is_leaf
    assert model.tree_.probabilities == (0.5, 0.5)
    np.testing.assert_array_equal(model.feature_importances_, [0.0, 0.0])


def test_minimum_sample_controls_prevent_invalid_children() -> None:
    features = np.arange(6.0).reshape(-1, 1)
    targets = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    leaf_limited = DecisionTreeClassifier(min_samples_leaf=3).fit(features, targets)
    split_limited = DecisionTreeClassifier(min_samples_split=7).fit(features, targets)

    assert not leaf_limited.tree_.is_leaf
    assert leaf_limited.tree_.left is not None
    assert leaf_limited.tree_.right is not None
    assert leaf_limited.tree_.left.n_samples >= 3
    assert leaf_limited.tree_.right.n_samples >= 3
    assert split_limited.tree_.is_leaf


def test_feature_subsampling_replays_seed_and_does_not_mutate_generator() -> None:
    generator = np.random.default_rng(91)
    control = np.random.default_rng(91)
    features = np.random.default_rng(4).normal(size=(80, 6))
    targets = (features[:, 1] + features[:, 4] > 0.0).astype(np.float64)

    first = DecisionTreeClassifier(max_depth=4, max_features=2, random_state=generator).fit(
        features, targets
    )
    second = DecisionTreeClassifier(max_depth=4, max_features=2, random_state=91).fit(
        features, targets
    )
    np.testing.assert_array_equal(first.predict(features), second.predict(features))
    np.testing.assert_allclose(first.feature_importances_, second.feature_importances_)
    assert first.tree_ == second.tree_
    assert generator.integers(0, 10_000) == control.integers(0, 10_000)


def test_fractional_and_named_feature_subsampling_are_supported() -> None:
    features = np.random.default_rng(3).normal(size=(50, 5))
    targets = features[:, 0] - features[:, 2]
    for max_features in (0.4, "sqrt", "log2", None):
        model = DecisionTreeRegressor(max_depth=3, max_features=max_features, random_state=8).fit(
            features, targets
        )
        assert model.predict(features[:4]).shape == (4,)
        assert np.sum(model.feature_importances_) == pytest.approx(1.0)


def test_failed_refit_is_transactional() -> None:
    features = np.arange(6.0).reshape(-1, 1)
    model = DecisionTreeRegressor().fit(features, np.arange(6.0))
    expected = model.predict(features)

    with pytest.raises(ValueError, match="only finite"):
        model.fit([[0.0], [np.nan]], [0.0, 1.0])

    np.testing.assert_allclose(model.predict(features), expected)
    assert model.n_features_in_ == 1


def test_constant_regression_target_uses_regressor_score_convention() -> None:
    features = np.arange(4.0).reshape(-1, 1)
    exact = DecisionTreeRegressor(max_depth=0).fit(features, np.full(4, 2.0))
    assert exact.score(features, np.full(4, 2.0)) == 1.0
    assert exact.tree_.impurity == 0.0


@pytest.mark.parametrize(
    ("factory", "error", "match"),
    [
        (lambda: DecisionTreeClassifier(criterion="variance"), ValueError, "criterion"),
        (lambda: DecisionTreeRegressor(criterion="gini"), ValueError, "criterion"),
        (lambda: DecisionTreeClassifier(max_depth=-1), ValueError, "max_depth"),
        (lambda: DecisionTreeClassifier(max_depth=1.5), TypeError, "max_depth"),
        (lambda: DecisionTreeClassifier(min_samples_split=1), ValueError, "min_samples_split"),
        (lambda: DecisionTreeClassifier(min_samples_split=True), TypeError, "min_samples_split"),
        (lambda: DecisionTreeClassifier(min_samples_leaf=0), ValueError, "min_samples_leaf"),
        (lambda: DecisionTreeClassifier(max_features=0), ValueError, "max_features"),
        (lambda: DecisionTreeClassifier(max_features=True), TypeError, "max_features"),
        (lambda: DecisionTreeClassifier(max_features=1.2), ValueError, "max_features"),
        (lambda: DecisionTreeClassifier(max_features="all"), ValueError, "max_features"),
        (lambda: DecisionTreeClassifier(max_features=[]), TypeError, "max_features"),
        (
            lambda: DecisionTreeClassifier(min_impurity_decrease=-0.1),
            ValueError,
            "min_impurity_decrease",
        ),
        (lambda: DecisionTreeClassifier(random_state=-1), ValueError, "random_state"),
        (lambda: DecisionTreeClassifier(random_state=True), TypeError, "random_state"),
    ],
)
def test_invalid_hyperparameters_fail_early(
    factory: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        factory()  # type: ignore[operator]


def test_max_features_cannot_exceed_fitted_width() -> None:
    model = DecisionTreeRegressor(max_features=3)
    with pytest.raises(ValueError, match="exceeds the fitted feature count"):
        model.fit([[0.0, 1.0], [1.0, 0.0]], [0.0, 1.0])
    assert not model.is_fitted


def test_inference_contract_rejects_unfitted_and_wrong_width() -> None:
    model = DecisionTreeClassifier()
    with pytest.raises(NotFittedError, match="call fit"):
        model.predict([[0.0]])
    with pytest.raises(NotFittedError):
        model.predict_proba([[0.0]])

    model.fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="expects 1"):
        model.predict([[0.0, 1.0]])


def test_tree_node_leaf_contract_is_structural() -> None:
    leaf = TreeNode(prediction=1.0, impurity=0.0, n_samples=2)
    branch = TreeNode(
        prediction=0.0,
        impurity=0.5,
        n_samples=4,
        feature_index=0,
        threshold=0.5,
        left=leaf,
        right=leaf,
    )
    assert leaf.is_leaf
    assert not branch.is_leaf
