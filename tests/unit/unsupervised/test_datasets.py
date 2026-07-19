"""Deterministic unsupervised dataset tests."""

import numpy as np
import pytest

from learning_atlas.unsupervised.datasets import clustering_suite, make_moons

pytestmark = pytest.mark.unit


def test_clustering_suite_replays_with_read_only_truth() -> None:
    first = clustering_suite(n_samples=120, seed=42)
    second = clustering_suite(n_samples=120, seed=42)

    assert [dataset.name for dataset in first] == [
        "isotropic_blobs",
        "two_moons",
        "anisotropic_blobs",
        "variable_density_blobs",
    ]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.features, right.features)
        np.testing.assert_array_equal(left.evaluation_labels, right.evaluation_labels)
        assert left.features.shape == (120, 2)
        assert not left.features.flags.writeable
        assert not left.evaluation_labels.flags.writeable


def test_dataset_namespaces_produce_distinct_observations() -> None:
    datasets = clustering_suite(n_samples=120, seed=7)
    fingerprints = {dataset.features.tobytes() for dataset in datasets}
    assert len(fingerprints) == len(datasets)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"n_samples": True}, TypeError, "integer"),
        ({"n_samples": 29}, ValueError, "at least 30"),
        ({"noise": -0.1}, ValueError, "non-negative"),
    ],
)
def test_moons_rejects_invalid_generation_requests(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        make_moons(**kwargs)  # type: ignore[arg-type]
