"""Mathematical, deterministic, and failure-contract tests for K-means."""

import pickle
from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest
from sklearn.cluster import KMeans as ReferenceKMeans
from sklearn.metrics import adjusted_rand_score

from learning_atlas.unsupervised.base import NotFittedError
from learning_atlas.unsupervised.kmeans import (
    KMeans,
    KMeansConvergenceError,
    KMeansNumericalError,
    _assign_with_empty_cluster_repair,
)

pytestmark = pytest.mark.unit


def _separated_blobs(seed: int = 2026) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    centers = np.array([[-5.0, -1.0], [0.0, 5.0], [5.0, 0.5]], dtype=np.float64)
    features = np.vstack([generator.normal(center, 0.45, size=(90, 2)) for center in centers])
    labels = np.repeat(np.arange(len(centers), dtype=np.int64), 90)
    return features, labels, centers


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    return values[
        np.lexsort(tuple(values[:, column] for column in reversed(range(values.shape[1]))))
    ]


def test_kmeans_recovers_blobs_and_publishes_consistent_state() -> None:
    features, truth, expected_centers = _separated_blobs()
    model = KMeans(3, n_init=6, tol=1e-8, random_state=42).fit(features)

    np.testing.assert_allclose(
        _sorted_rows(model.cluster_centers_),
        _sorted_rows(expected_centers),
        atol=0.15,
    )
    assert adjusted_rand_score(truth, model.labels_) == pytest.approx(1.0)
    np.testing.assert_array_equal(model.predict(features), model.labels_)
    distances = model.transform(features)
    assert distances.shape == (len(features), 3)
    assert model.inertia_ == pytest.approx(
        np.sum(distances[np.arange(len(features)), model.labels_] ** 2)
    )
    assert model.converged_ is True
    assert model.n_iter_ == len(model.inertia_history_)
    assert all(
        current <= previous + 1e-10 for previous, current in pairwise(model.inertia_history_)
    )


def test_kmeans_plus_plus_beats_random_initialization_over_fixed_seed_suite() -> None:
    generator = np.random.default_rng(2026)
    centers = np.array(
        [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0], [20.0, 20.0]],
        dtype=np.float64,
    )
    sizes = (400, 150, 150, 50, 20)
    features = np.vstack(
        [
            generator.normal(center, 0.7, size=(size, 2))
            for center, size in zip(centers, sizes, strict=True)
        ]
    )

    plus_plus = [
        KMeans(5, init="k-means++", n_init=1, tol=1e-10, random_state=seed).fit(features).inertia_
        for seed in range(12)
    ]
    random = [
        KMeans(5, init="random", n_init=1, tol=1e-10, random_state=seed).fit(features).inertia_
        for seed in range(12)
    ]

    assert np.mean(plus_plus) < 0.5 * np.mean(random)


def test_kmeans_matches_independent_reference_partition_and_objective() -> None:
    features, _, _ = _separated_blobs(seed=11)
    actual = KMeans(3, n_init=10, tol=1e-10, random_state=7).fit(features)
    reference = ReferenceKMeans(
        n_clusters=3,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-10,
        random_state=7,
        algorithm="lloyd",
    ).fit(features)

    assert adjusted_rand_score(reference.labels_, actual.labels_) == pytest.approx(1.0)
    assert actual.inertia_ == pytest.approx(float(reference.inertia_), rel=1e-10, abs=1e-10)


def test_restart_streams_are_order_isolated_and_best_restart_is_selected() -> None:
    features, _, _ = _separated_blobs(seed=17)
    one = KMeans(3, n_init=1, random_state=91).fit(features)
    four = KMeans(3, n_init=4, random_state=91).fit(features)

    assert one.restart_histories_[0] == four.restart_histories_[0]
    assert one.restart_inertias_[0] == four.restart_inertias_[0]
    assert four.inertia_ == min(four.restart_inertias_)
    assert len(four.restart_histories_) == 4


def test_empty_cluster_repair_moves_stable_farthest_donors() -> None:
    distances = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [4.0, 14.0, 24.0],
            [9.0, 19.0, 29.0],
        ],
        dtype=np.float64,
    )

    first, repaired = _assign_with_empty_cluster_repair(distances, 3)
    second, _ = _assign_with_empty_cluster_repair(distances, 3)

    assert repaired is True
    np.testing.assert_array_equal(first, np.array([0, 0, 2, 1], dtype=np.int64))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.bincount(first, minlength=3), np.array([2, 1, 1]))


def test_distance_ties_choose_lowest_component_and_properties_are_defensive() -> None:
    features = np.array([[-3.0], [-2.0], [2.0], [3.0]], dtype=np.float64)
    model = KMeans(2, n_init=2, random_state=3).fit(features)
    midpoint = np.mean(model.cluster_centers_, axis=0, keepdims=True)

    assert model.predict(midpoint)[0] == 0
    exposed = model.cluster_centers_
    exposed[:] = 999.0
    exposed_labels = model.labels_
    exposed_labels[:] = 99
    assert not np.all(model.cluster_centers_ == 999.0)
    assert not np.all(model.labels_ == 99)


def test_serialization_preserves_restart_state_and_inference() -> None:
    features, _, _ = _separated_blobs(seed=61)
    model = KMeans(3, n_init=4, random_state=19).fit(features)

    restored = pickle.loads(pickle.dumps(model, protocol=5))

    np.testing.assert_array_equal(restored.cluster_centers_, model.cluster_centers_)
    np.testing.assert_array_equal(restored.labels_, model.labels_)
    np.testing.assert_array_equal(restored.predict(features), model.predict(features))
    assert restored.inertia_history_ == model.inertia_history_
    assert restored.restart_histories_ == model.restart_histories_
    assert restored.restart_inertias_ == model.restart_inertias_


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: KMeans(0), ValueError, "n_clusters"),
        (lambda: KMeans(2, n_init=0), ValueError, "n_init"),
        (lambda: KMeans(2, max_iter=0), ValueError, "max_iter"),
        (lambda: KMeans(2, tol=-1.0), ValueError, "tol"),
        (lambda: KMeans(2, tol=np.inf), ValueError, "tol"),
        (lambda: KMeans(2, tol=True), TypeError, "real number"),
        (lambda: KMeans(2, init="forgy"), ValueError, "init"),  # type: ignore[arg-type]
        (lambda: KMeans(True), TypeError, "integer"),
        (lambda: KMeans(2, random_state=2**32), ValueError, "unsigned"),
    ],
)
def test_invalid_hyperparameters_fail_early(
    factory: Callable[[], object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        factory()


def test_fit_and_inference_validate_shape_finiteness_and_distinct_count() -> None:
    model = KMeans(2)
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict([[0.0]])
    with pytest.raises(ValueError, match="2D"):
        model.fit([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        model.fit([[0.0], [np.nan]])
    with pytest.raises(ValueError, match="distinct"):
        model.fit(np.ones((3, 1), dtype=np.float64))

    fitted = model.fit([[-1.0, 0.0], [-0.8, 0.1], [1.0, 0.0], [0.8, -0.1]])
    before = fitted.cluster_centers_
    with pytest.raises(ValueError, match="expects 2"):
        fitted.predict([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="distinct"):
        fitted.fit(np.ones((4, 3), dtype=np.float64))
    assert fitted.n_features_in_ == 2
    np.testing.assert_array_equal(fitted.cluster_centers_, before)


def test_nonconvergence_exposes_history_without_publishing_partial_state() -> None:
    generator = np.random.default_rng(9)
    features = np.vstack(
        [
            generator.normal([0.0, 0.0], 2.0, size=(50, 2)),
            generator.normal([3.0, 1.0], 2.0, size=(50, 2)),
            generator.normal([0.0, 5.0], 2.0, size=(50, 2)),
        ]
    )
    model = KMeans(3, n_init=1, max_iter=1, tol=0.0, random_state=0)

    with pytest.raises(KMeansConvergenceError, match="increase max_iter") as captured:
        model.fit(features)

    assert captured.value.restart == 0
    assert captured.value.max_iter == 1
    assert len(captured.value.inertia_history) == 1
    assert model.n_features_in_ is None
    with pytest.raises(NotFittedError):
        _ = model.labels_


def test_extreme_finite_scale_raises_actionable_numerical_error() -> None:
    features = np.array([[-1e308, 0.0], [1e308, 0.0]], dtype=np.float64)

    with pytest.raises(KMeansNumericalError, match="rescale"):
        KMeans(2, random_state=0).fit(features)
    with pytest.raises(KMeansNumericalError, match="cluster means"):
        KMeans(1, random_state=0).fit(np.full((3, 1), 1e308, dtype=np.float64))
