"""Mathematical, differential, and contract tests for nearest neighbors."""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor as SklearnKNeighborsRegressor

from learning_atlas.core.estimators import NotFittedError
from learning_atlas.supervised.datasets import make_blobs
from learning_atlas.supervised.neighbors import KNeighborsClassifier, KNeighborsRegressor

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("metric", "p"),
    [("euclidean", 2.0), ("manhattan", 2.0), ("minkowski", 3.0)],
)
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_classifier_matches_independent_sklearn_oracle(
    metric: str,
    p: float,
    weights: str,
) -> None:
    rng = np.random.default_rng(2026)
    features = rng.normal(size=(45, 4))
    targets = np.where(features[:, 0] - 0.4 * features[:, 1] > 0.0, 7.0, -3.0)
    queries = rng.normal(size=(12, 4))
    model = KNeighborsClassifier(
        n_neighbors=5,
        metric=metric,  # type: ignore[arg-type]
        p=p,
        weights=weights,  # type: ignore[arg-type]
    ).fit(features, targets)
    oracle = SklearnKNeighborsClassifier(
        n_neighbors=5,
        metric=metric,
        p=p,
        weights=weights,
        algorithm="brute",
    ).fit(features, targets)

    np.testing.assert_array_equal(model.predict(queries), oracle.predict(queries))
    np.testing.assert_allclose(model.predict_proba(queries), oracle.predict_proba(queries))


@pytest.mark.parametrize(
    ("metric", "p"),
    [("euclidean", 2.0), ("manhattan", 2.0), ("minkowski", 4.0)],
)
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_regressor_matches_independent_sklearn_oracle(
    metric: str,
    p: float,
    weights: str,
) -> None:
    rng = np.random.default_rng(91)
    features = rng.uniform(-2.0, 2.0, size=(40, 3))
    targets = features @ np.asarray([1.5, -0.2, 0.7]) + rng.normal(0.0, 0.05, size=40)
    queries = rng.uniform(-2.0, 2.0, size=(9, 3))
    model = KNeighborsRegressor(
        n_neighbors=4,
        metric=metric,  # type: ignore[arg-type]
        p=p,
        weights=weights,  # type: ignore[arg-type]
    ).fit(features, targets)
    oracle = SklearnKNeighborsRegressor(
        n_neighbors=4,
        metric=metric,
        p=p,
        weights=weights,
        algorithm="brute",
    ).fit(features, targets)

    np.testing.assert_allclose(model.predict(queries), oracle.predict(queries), atol=1e-12)


def test_kneighbors_uses_vectorized_metrics_and_stable_training_index_ties() -> None:
    features = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    targets = np.asarray([0.0, 1.0, 1.0])
    query = np.asarray([[1.0, 0.0]])

    euclidean = KNeighborsClassifier(n_neighbors=3, metric="euclidean").fit(features, targets)
    distances, indices = euclidean.kneighbors(query)
    np.testing.assert_allclose(distances, [[1.0, 1.0, np.sqrt(5.0)]])
    np.testing.assert_array_equal(indices, [[0, 1, 2]])

    manhattan = KNeighborsClassifier(n_neighbors=3, metric="manhattan").fit(features, targets)
    np.testing.assert_allclose(manhattan.kneighbors(query)[0], [[1.0, 1.0, 3.0]])
    minkowski = KNeighborsClassifier(n_neighbors=3, metric="minkowski", p=3.0).fit(
        features, targets
    )
    np.testing.assert_allclose(minkowski.kneighbors(query)[0][0, 2], 9.0 ** (1.0 / 3.0))


def test_class_vote_ties_resolve_to_smallest_sorted_label() -> None:
    model = KNeighborsClassifier(n_neighbors=2).fit([[-1.0], [1.0]], [7.0, 3.0])

    np.testing.assert_array_equal(model.classes_, [3.0, 7.0])
    np.testing.assert_allclose(model.predict_proba([[0.0]]), [[0.5, 0.5]])
    np.testing.assert_array_equal(model.predict([[0.0]]), [3.0])


def test_distance_weighting_uses_only_exact_matches() -> None:
    features = [[0.0], [0.0], [1.0]]
    classifier = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(
        features, [2.0, 4.0, 4.0]
    )
    regressor = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(
        features, [2.0, 4.0, 100.0]
    )

    np.testing.assert_allclose(classifier.predict_proba([[0.0]]), [[0.5, 0.5]])
    np.testing.assert_array_equal(classifier.predict([[0.0]]), [2.0])
    np.testing.assert_allclose(regressor.predict([[0.0]]), [3.0])


def test_distance_weighting_remains_finite_for_subnormal_distances() -> None:
    features = np.asarray([[0.0], [1.0e-323]])
    query = np.asarray([[5.0e-324]])
    classifier = KNeighborsClassifier(n_neighbors=2, weights="distance").fit(features, [0.0, 1.0])
    regressor = KNeighborsRegressor(n_neighbors=2, weights="distance").fit(features, [0.0, 2.0])

    probabilities = classifier.predict_proba(query)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities, [[0.5, 0.5]])
    np.testing.assert_allclose(regressor.predict(query), [1.0])


def test_one_neighbor_memorizes_unique_training_samples_exactly() -> None:
    features = np.asarray([[-2.0, 1.0], [0.0, 3.0], [4.0, -1.0]])
    class_targets = np.asarray([9.0, -3.0, 2.0])
    regression_targets = np.asarray([1.25, -8.5, 11.0])

    classifier = KNeighborsClassifier(n_neighbors=1).fit(features, class_targets)
    regressor = KNeighborsRegressor(n_neighbors=1).fit(features, regression_targets)
    np.testing.assert_array_equal(classifier.predict(features), class_targets)
    np.testing.assert_array_equal(regressor.predict(features), regression_targets)


def test_distance_weighting_produces_a_smooth_proximity_sensitive_boundary() -> None:
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    targets = np.asarray([0.0, 0.0, 1.0, 1.0])
    queries = np.linspace(-0.9, 0.9, 9)[:, np.newaxis]
    uniform = KNeighborsClassifier(n_neighbors=4, weights="uniform").fit(features, targets)
    weighted = KNeighborsClassifier(n_neighbors=4, weights="distance").fit(features, targets)

    uniform_positive = uniform.predict_proba(queries)[:, 1]
    weighted_positive = weighted.predict_proba(queries)[:, 1]
    np.testing.assert_allclose(uniform_positive, 0.5)
    assert np.all(np.diff(weighted_positive) > 0.0)
    assert np.ptp(weighted_positive) > 0.35


def test_make_blobs_distance_weighting_can_override_a_uniform_neighbor_majority() -> None:
    dataset = make_blobs(
        n_samples=9,
        centers=[[-4.0, 0.0], [4.0, 0.0]],
        cluster_std=0.25,
        shuffle=True,
        seed=31,
    )
    class_one_anchor = dataset.features[np.flatnonzero(dataset.targets == 1)[0]]
    query = (class_one_anchor + np.asarray([1.0e-3, 0.0]))[np.newaxis, :]
    uniform = KNeighborsClassifier(n_neighbors=9, weights="uniform").fit(
        dataset.features, dataset.targets
    )
    distance = KNeighborsClassifier(n_neighbors=9, weights="distance").fit(
        dataset.features, dataset.targets
    )

    np.testing.assert_array_equal(np.bincount(dataset.targets), [5, 4])
    np.testing.assert_allclose(uniform.predict_proba(query), [[5.0 / 9.0, 4.0 / 9.0]])
    np.testing.assert_array_equal(uniform.predict(query), [0.0])
    assert distance.predict_proba(query)[0, 1] > 0.99
    np.testing.assert_array_equal(distance.predict(query), [1.0])


def test_distance_weighting_smooths_probability_steps_along_a_blob_path() -> None:
    dataset = make_blobs(
        n_samples=80,
        centers=[[-3.0, 0.0], [3.0, 0.0]],
        cluster_std=0.6,
        shuffle=True,
        seed=2026,
    )
    path = np.column_stack((np.linspace(-3.0, 3.0, 601), np.zeros(601)))
    uniform = KNeighborsClassifier(n_neighbors=15, weights="uniform").fit(
        dataset.features,
        dataset.targets,
    )
    weighted = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(
        dataset.features,
        dataset.targets,
    )

    uniform_probability = uniform.predict_proba(path)[:, 1]
    weighted_probability = weighted.predict_proba(path)[:, 1]
    uniform_step = float(np.max(np.abs(np.diff(uniform_probability))))
    weighted_step = float(np.max(np.abs(np.diff(weighted_probability))))

    assert weighted_step < uniform_step
    assert len(np.unique(np.round(weighted_probability, 8))) > 5 * len(
        np.unique(uniform_probability)
    )
    assert np.ptp(weighted_probability) > 0.99


def test_fit_owns_training_snapshot_and_score_contracts_work() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    targets = np.asarray([0.0, 0.0, 1.0, 1.0])
    classifier = KNeighborsClassifier(n_neighbors=1).fit(features, targets)
    regressor = KNeighborsRegressor(n_neighbors=1).fit(features, targets)
    features[:] = 99.0
    targets[:] = 99.0

    np.testing.assert_array_equal(classifier.predict([[0.0], [3.0]]), [0.0, 1.0])
    assert classifier.score([[0.0], [3.0]], [0.0, 1.0]) == 1.0
    assert regressor.score([[0.0], [3.0]], [0.0, 1.0]) == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_neighbors": 0}, "n_neighbors"),
        ({"weights": "rank"}, "weights"),
        ({"metric": "cosine"}, "metric"),
        ({"p": 0.5}, "p"),
        ({"p": np.inf}, "p"),
    ],
)
def test_invalid_hyperparameters_fail_early(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        KNeighborsClassifier(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"n_neighbors": True}, {"p": True}, {"p": "two"}])
def test_non_numeric_hyperparameters_raise_type_error(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        KNeighborsClassifier(**kwargs)  # type: ignore[arg-type]


def test_fitted_and_inference_boundaries_are_actionable() -> None:
    model = KNeighborsRegressor(n_neighbors=3)
    with pytest.raises(NotFittedError, match="call fit"):
        model.predict([[0.0]])
    with pytest.raises(ValueError, match="exceeds the 2 available"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="exceeds the 2 available"):
        KNeighborsClassifier(n_neighbors=3).fit([[0.0], [1.0]], [0.0, 1.0])

    fitted = KNeighborsRegressor(n_neighbors=1).fit([[0.0, 1.0]], [2.0])
    with pytest.raises(ValueError, match="expects 2"):
        fitted.predict([[0.0]])
    extreme = KNeighborsRegressor(n_neighbors=1).fit([[1e308]], [1.0])
    with pytest.raises(ValueError, match="distance computation produced non-finite"):
        extreme.predict([[-1e308]])
