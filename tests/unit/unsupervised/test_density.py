"""Definition, determinism, differential, and failure tests for NumPy DBSCAN."""

import inspect
import pickle

import numpy as np
import pytest
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_moons

from learning_atlas.unsupervised.base import NotFittedError
from learning_atlas.unsupervised.density import DBSCAN
from learning_atlas.unsupervised.metrics import adjusted_rand_score

pytestmark = pytest.mark.unit


def test_core_chain_border_and_noise_satisfy_density_definitions() -> None:
    features = np.array([[0.0], [0.1], [0.2], [0.3], [2.0]], dtype=np.float64)
    model = DBSCAN(eps=0.11, min_samples=3).fit(features)

    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 0, -1])
    np.testing.assert_array_equal(model.core_sample_indices_, [1, 2])
    np.testing.assert_array_equal(model.core_sample_mask_, [False, True, True, False, False])
    np.testing.assert_array_equal(
        model.sample_roles_, ["border", "core", "core", "border", "noise"]
    )
    np.testing.assert_array_equal(model.neighborhood_counts_, [2, 3, 3, 2, 1])
    np.testing.assert_allclose(model.components_, [[0.1], [0.2]])
    assert np.all(model.k_distances_ >= 0.0)
    assert set(np.unique(model.sample_roles_)) == {"core", "border", "noise"}


def test_exact_eps_boundary_duplicates_all_noise_and_one_cluster() -> None:
    exact = DBSCAN(eps=1.0, min_samples=2).fit_predict([[0.0], [1.0]])
    np.testing.assert_array_equal(exact, [0, 0])

    duplicates = DBSCAN(eps=1.0e-6, min_samples=3).fit_predict(
        [[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]]
    )
    np.testing.assert_array_equal(duplicates, [0, 0, 0])

    all_noise_model = DBSCAN(eps=0.1, min_samples=4).fit([[0.0], [1.0], [2.0]])
    np.testing.assert_array_equal(all_noise_model.labels_, [-1, -1, -1])
    assert len(all_noise_model.core_sample_indices_) == 0
    assert set(all_noise_model.sample_roles_) == {"noise"}
    assert np.all(np.isfinite(all_noise_model.k_distances_))


def test_border_tie_uses_closest_core_then_geometry_key() -> None:
    features = np.array(
        [[-1.4], [-1.3], [-1.2], [-1.1], [1.1], [1.2], [1.3], [1.4], [0.0]],
        dtype=np.float64,
    )

    model = DBSCAN(eps=1.1, min_samples=4).fit(features)

    assert model.sample_roles_[-1] == "border"
    assert model.labels_[-1] == model.labels_[0]
    assert model.labels_[-1] != model.labels_[4]


def test_row_permutation_preserves_partition_roles_and_border_tie() -> None:
    features = np.array(
        [[-1.4], [-1.3], [-1.2], [-1.1], [1.1], [1.2], [1.3], [1.4], [0.0]],
        dtype=np.float64,
    )
    permutation = np.array([8, 4, 2, 7, 0, 5, 3, 1, 6], dtype=np.int64)
    original = DBSCAN(eps=1.1, min_samples=4).fit(features)
    shuffled = DBSCAN(eps=1.1, min_samples=4).fit(features[permutation])
    mapped_labels = np.empty_like(shuffled.labels_)
    mapped_roles = np.empty_like(shuffled.sample_roles_)
    mapped_labels[permutation] = shuffled.labels_
    mapped_roles[permutation] = shuffled.sample_roles_

    assert adjusted_rand_score(original.labels_, mapped_labels) == 1.0
    np.testing.assert_array_equal(original.sample_roles_, mapped_roles)
    assert mapped_labels[-1] == mapped_labels[0]


def test_two_moons_recovery_and_independent_reference_agreement() -> None:
    raw_features, truth = make_moons(n_samples=320, noise=0.055, random_state=42)
    features = np.asarray(raw_features, dtype=np.float64)
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    observed = DBSCAN(eps=0.24, min_samples=5).fit_predict(features)
    reference = SklearnDBSCAN(eps=0.24, min_samples=5).fit_predict(features)

    assert adjusted_rand_score(reference, observed) == 1.0
    assert adjusted_rand_score(truth, observed) > 0.95
    assert float(np.mean(observed != -1)) > 0.98


def test_fit_accepts_features_only_and_model_round_trips() -> None:
    assert tuple(inspect.signature(DBSCAN.fit).parameters) == ("self", "features")
    features = np.array([[0.0], [0.05], [0.1], [1.0]], dtype=np.float64)
    model = DBSCAN(eps=0.11, min_samples=2).fit(features)

    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_array_equal(restored.labels_, model.labels_)
    np.testing.assert_array_equal(restored.sample_roles_, model.sample_roles_)
    with pytest.raises(TypeError):
        model.fit(features, np.zeros(len(features)))  # type: ignore[call-arg]
    assert not hasattr(model, "predict")
    with pytest.raises(ValueError):
        model.labels_[0] = 99


def test_fitted_properties_fail_before_fit() -> None:
    model = DBSCAN()

    for attribute in (
        "labels_",
        "core_sample_indices_",
        "core_sample_mask_",
        "sample_roles_",
        "neighborhood_counts_",
        "k_distances_",
        "components_",
    ):
        with pytest.raises(NotFittedError, match="not fitted"):
            getattr(model, attribute)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"eps": 0.0}, "eps"),
        ({"eps": np.inf}, "eps"),
        ({"eps": True}, "eps"),
        ({"min_samples": 0}, "min_samples"),
        ({"min_samples": True}, "min_samples"),
        ({"max_pairwise_elements": 0}, "max_pairwise_elements"),
    ],
)
def test_invalid_hyperparameters_are_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DBSCAN(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("features", "message"),
    [
        ([1.0, 2.0], "2D"),
        ([[0.0], [np.nan]], "finite"),
        ([["not-numeric"]], "real numeric"),
        (np.empty((0, 1)), "at least 1 sample"),
    ],
)
def test_invalid_training_data_is_rejected(features: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        DBSCAN().fit(features)  # type: ignore[arg-type]


def test_pairwise_resource_limit_fails_before_fitted_state_changes() -> None:
    model = DBSCAN(max_pairwise_elements=8)

    with pytest.raises(MemoryError, match="requires 9 elements"):
        model.fit(np.zeros((3, 2)))
    assert model.n_features_in_ is None
    with pytest.raises(NotFittedError):
        _ = model.labels_
