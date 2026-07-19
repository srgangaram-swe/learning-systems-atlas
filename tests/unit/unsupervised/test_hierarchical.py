"""Hierarchy invariants, linkage definitions, and error paths."""

import pickle

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative

from learning_atlas.unsupervised.base import NotFittedError
from learning_atlas.unsupervised.hierarchical import (
    AgglomerativeClustering,
    cut_linkage,
)
from learning_atlas.unsupervised.metrics import adjusted_rand_score

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("linkage", "expected_distances"),
    [
        ("single", [1.0, 2.0, 7.0]),
        ("complete", [1.0, 3.0, 10.0]),
        ("average", [1.0, 2.5, 25.0 / 3.0]),
        ("ward", [1.0, np.sqrt(25.0 / 3.0), np.sqrt(625.0 / 6.0)]),
    ],
)
def test_linkages_reproduce_hand_derived_merges(
    linkage: str,
    expected_distances: list[float],
) -> None:
    features = np.array([[0.0], [2.0], [3.0], [10.0]], dtype=np.float64)

    model = AgglomerativeClustering(n_clusters=2, linkage=linkage).fit(features)  # type: ignore[arg-type]

    np.testing.assert_array_equal(model.children_, [[1, 2], [0, 4], [3, 5]])
    np.testing.assert_allclose(model.distances_, expected_distances, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(model.cluster_sizes_, [2, 3, 4])


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_merge_history_is_valid_chronological_and_scipy_compatible(linkage: str) -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(13, 3))
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit(features)  # type: ignore[arg-type]
    history = model.linkage_matrix_

    assert history.dtype == np.float64
    assert history.shape == (len(features) - 1, 4)
    assert np.all(np.diff(history[:, 2]) >= -1e-14)
    assert history[-1, 3] == len(features)
    active = set(range(len(features)))
    for step, (left_raw, right_raw, distance, size) in enumerate(history):
        left, right = int(left_raw), int(right_raw)
        assert left in active
        assert right in active
        assert left != right
        assert distance >= 0.0
        assert size >= 2.0
        active.remove(left)
        active.remove(right)
        active.add(len(features) + step)
    assert len(active) == 1


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_every_valid_cut_has_exact_cluster_count_and_matches_sklearn(linkage: str) -> None:
    features = np.array(
        [[0.0, 0.0], [0.2, 0.1], [3.8, -0.1], [4.3, 0.2], [8.5, 3.0], [9.2, 3.4]],
        dtype=np.float64,
    )
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit(features)  # type: ignore[arg-type]

    for n_clusters in range(1, len(features) + 1):
        cut = model.cut(n_clusters)
        assert len(np.unique(cut)) == n_clusters
        np.testing.assert_array_equal(cut, cut_linkage(model.linkage_matrix_, n_clusters))

    reference = SklearnAgglomerative(n_clusters=3, linkage=linkage).fit_predict(features)
    assert adjusted_rand_score(reference, model.labels_) == 1.0


def test_row_permutation_preserves_untied_partition() -> None:
    features = np.array(
        [[-4.0, 0.0], [-3.8, 0.1], [0.0, 2.0], [0.3, 2.2], [7.0, -1.0], [7.8, -0.8]],
        dtype=np.float64,
    )
    permutation = np.array([4, 1, 5, 2, 0, 3], dtype=np.int64)
    original = AgglomerativeClustering(n_clusters=3, linkage="average").fit_predict(features)
    shuffled = AgglomerativeClustering(n_clusters=3, linkage="average").fit_predict(
        features[permutation]
    )
    mapped = np.empty_like(shuffled)
    mapped[permutation] = shuffled

    assert adjusted_rand_score(original, mapped) == 1.0


def test_equal_distance_ties_and_duplicates_are_deterministic() -> None:
    features = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)

    first = AgglomerativeClustering(n_clusters=2, linkage="complete").fit(features)
    second = AgglomerativeClustering(n_clusters=2, linkage="complete").fit(features)

    np.testing.assert_array_equal(first.linkage_matrix_, second.linkage_matrix_)
    np.testing.assert_array_equal(first.children_, [[0, 1], [2, 3], [4, 5]])
    np.testing.assert_array_equal(first.labels_, [0, 0, 1, 1])


def test_singleton_hierarchy_has_canonical_empty_history() -> None:
    model = AgglomerativeClustering(n_clusters=1, linkage="single").fit([[7.0, -2.0]])

    assert model.linkage_matrix_.shape == (0, 4)
    assert model.linkage_matrix_.dtype == np.float64
    assert model.children_.shape == (0, 2)
    np.testing.assert_array_equal(model.labels_, [0])
    np.testing.assert_array_equal(model.cut(1), [0])


def test_hierarchy_round_trips_and_public_arrays_are_immutable() -> None:
    features = np.array([[0.0], [0.2], [3.0], [3.4]], dtype=np.float64)
    model = AgglomerativeClustering(n_clusters=2, linkage="average").fit(features)

    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_array_equal(restored.linkage_matrix_, model.linkage_matrix_)
    np.testing.assert_array_equal(restored.labels_, model.labels_)
    with pytest.raises(ValueError):
        model.linkage_matrix_[0, 0] = 99.0
    assert not hasattr(model, "predict")


def test_fitted_properties_and_cut_fail_before_fit() -> None:
    model = AgglomerativeClustering()

    for attribute in (
        "labels_",
        "linkage_matrix_",
        "children_",
        "distances_",
        "cluster_sizes_",
    ):
        with pytest.raises(NotFittedError, match="not fitted"):
            getattr(model, attribute)
    with pytest.raises(NotFittedError):
        model.cut(1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_clusters": 0}, "n_clusters"),
        ({"n_clusters": True}, "n_clusters"),
        ({"linkage": "centroid"}, "linkage"),
        ({"metric": "manhattan", "linkage": "ward"}, "Euclidean"),
        ({"metric": "cosine", "linkage": "single"}, "metric"),
        ({"max_pairwise_elements": 0}, "max_pairwise_elements"),
    ],
)
def test_invalid_hyperparameters_are_actionable(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AgglomerativeClustering(**kwargs)  # type: ignore[arg-type]


def test_invalid_training_requests_and_resource_limits_fail_actionably() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        AgglomerativeClustering(n_clusters=3).fit([[0.0], [1.0]])
    with pytest.raises(ValueError, match="finite"):
        AgglomerativeClustering().fit([[0.0], [np.inf]])
    with pytest.raises(ValueError, match="2D"):
        AgglomerativeClustering().fit([0.0, 1.0])
    with pytest.raises(MemoryError, match="requires 9 elements"):
        AgglomerativeClustering(max_pairwise_elements=8).fit(np.zeros((3, 1)))
    with pytest.raises(ValueError, match="must not exceed"):
        AgglomerativeClustering(n_clusters=1).fit([[0.0], [1.0]]).cut(3)
    with pytest.raises(ValueError, match="positive integer"):
        AgglomerativeClustering(n_clusters=1).fit([[0.0], [1.0]]).cut(0)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[0.0, 1.0, 2.0]], "shape"),
        ([[0.0, 0.0, 1.0, 2.0]], "distinct active"),
        ([[0.5, 1.0, 1.0, 2.0]], "identifiers must be integers"),
        ([[0.0, 1.0, -1.0, 2.0]], "non-negative"),
        ([[0.0, 1.0, 1.0, 3.0]], "incorrect cluster size"),
        (
            [[0.0, 1.0, 2.0, 2.0], [2.0, 3.0, 1.0, 3.0]],
            "nondecreasing",
        ),
        ([[0.0, 1.0, np.nan, 2.0]], "finite"),
    ],
)
def test_malformed_linkage_matrices_are_rejected(matrix: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        cut_linkage(matrix, 1)  # type: ignore[arg-type]
