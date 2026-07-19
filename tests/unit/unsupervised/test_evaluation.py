"""Label-free unsupervised evaluation tests."""

import numpy as np
import pytest

from learning_atlas.unsupervised.evaluation import (
    clustering_selection_score,
    distance_correlation,
    neighborhood_preservation,
    pairwise_squared_distances,
    select_highest,
)

pytestmark = pytest.mark.unit


def test_pairwise_distances_are_symmetric_and_non_negative() -> None:
    features = np.asarray([[0.0, 0.0], [3.0, 4.0], [-1.0, 0.0]])
    distances = pairwise_squared_distances(features)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0.0)
    assert distances[0, 1] == pytest.approx(25.0)
    assert np.all(distances >= 0.0)


def test_identity_embedding_preserves_neighbors_and_distances() -> None:
    features = np.random.default_rng(42).normal(size=(30, 4))
    assert neighborhood_preservation(features, features.copy(), n_neighbors=5) == 1.0
    assert distance_correlation(features, features.copy()) == pytest.approx(1.0)


def test_constant_embedding_has_zero_distance_correlation() -> None:
    features = np.arange(24, dtype=np.float64).reshape(12, 2)
    embedding = np.zeros((12, 2), dtype=np.float64)
    assert distance_correlation(features, embedding) == 0.0


def test_selection_score_rewards_geometry_coverage_and_stability() -> None:
    strong = clustering_selection_score(silhouette=0.8, coverage=1.0, stability=0.9)
    weak = clustering_selection_score(silhouette=0.3, coverage=0.6, stability=0.2)
    assert strong > weak
    assert select_highest({"zeta": 0.5, "alpha": 0.5}) == "alpha"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"silhouette": 2.0, "coverage": 1.0, "stability": 1.0},
        {"silhouette": 0.0, "coverage": -0.1, "stability": 1.0},
        {"silhouette": 0.0, "coverage": 1.0, "stability": float("nan")},
    ],
)
def test_selection_score_rejects_invalid_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        clustering_selection_score(**kwargs)


def test_embedding_metrics_validate_sample_alignment() -> None:
    with pytest.raises(ValueError, match="same number"):
        neighborhood_preservation(np.ones((5, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError, match="n_neighbors"):
        neighborhood_preservation(np.ones((5, 2)), np.ones((5, 2)), n_neighbors=5)
