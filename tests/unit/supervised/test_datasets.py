"""Determinism and ground-truth contracts for NumPy synthetic datasets."""

import numpy as np
import pytest

from learning_atlas.supervised.datasets import (
    make_blobs,
    make_classification,
    make_regression,
)

pytestmark = pytest.mark.unit


def test_regression_is_reproducible_and_exposes_exact_generating_equation() -> None:
    first = make_regression(
        n_samples=80,
        n_features=7,
        n_informative=4,
        noise=0.25,
        bias=3.5,
        seed=11,
    )
    second = make_regression(
        n_samples=80,
        n_features=7,
        n_informative=4,
        noise=0.25,
        bias=3.5,
        seed=11,
    )

    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.targets, second.targets)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_allclose(
        first.targets,
        first.features @ first.coefficients + first.intercept + first.noise,
        rtol=0.0,
        atol=1e-14,
    )
    assert np.count_nonzero(first.coefficients) == 4
    assert first.features.flags.writeable is False
    assert first.coefficients.flags.writeable is False


def test_regression_shuffle_preserves_ground_truth_and_different_seeds_diverge() -> None:
    ordered = make_regression(n_features=5, n_informative=2, shuffle=False, seed=7)
    shuffled = make_regression(n_features=5, n_informative=2, shuffle=True, seed=7)
    different = make_regression(n_features=5, n_informative=2, shuffle=True, seed=8)

    assert np.count_nonzero(ordered.coefficients[:2]) == 2
    assert np.count_nonzero(ordered.coefficients[2:]) == 0
    np.testing.assert_allclose(shuffled.targets, shuffled.features @ shuffled.coefficients)
    assert not np.array_equal(shuffled.features, different.features)


def test_classification_exposes_consistent_hyperplane_and_corruption_boundary() -> None:
    clean = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        class_sep=1.5,
        positive_fraction=0.35,
        seed=19,
    )
    noisy = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        class_sep=1.5,
        positive_fraction=0.35,
        flip_y=1.0,
        seed=19,
    )

    np.testing.assert_allclose(clean.logits, clean.features @ clean.coefficients + clean.intercept)
    np.testing.assert_array_equal(clean.targets, clean.clean_targets)
    np.testing.assert_array_equal(clean.clean_targets, (clean.logits >= 0.0).astype(np.int64))
    np.testing.assert_array_equal(noisy.targets, 1 - noisy.clean_targets)
    assert set(np.unique(clean.targets)) == {0, 1}
    assert clean.targets.flags.writeable is False


def test_classification_repeatability_includes_generator_truth() -> None:
    first = make_classification(seed=41)
    second = make_classification(seed=41)

    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.targets, second.targets)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert first.intercept == second.intercept


def test_blobs_are_reproducible_balanced_and_center_metadata_is_preserved() -> None:
    first = make_blobs(n_samples=101, n_features=3, centers=4, cluster_std=0.2, seed=5)
    second = make_blobs(n_samples=101, n_features=3, centers=4, cluster_std=0.2, seed=5)

    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.targets, second.targets)
    np.testing.assert_array_equal(first.centers, second.centers)
    counts = np.bincount(first.targets)
    assert counts.max() - counts.min() <= 1
    assert first.features.shape == (101, 3)
    assert first.centers.shape == (4, 3)
    assert first.centers.flags.writeable is False


def test_blobs_accept_explicit_centers_and_per_cluster_spread() -> None:
    centers = [[-4.0, 1.0], [4.0, -1.0]]
    dataset = make_blobs(
        n_samples=400,
        centers=centers,
        cluster_std=[0.05, 0.1],
        shuffle=False,
        seed=3,
    )

    np.testing.assert_array_equal(dataset.centers, centers)
    np.testing.assert_array_equal(dataset.cluster_std, [0.05, 0.1])
    np.testing.assert_allclose(dataset.features[:200].mean(axis=0), centers[0], atol=0.02)
    np.testing.assert_allclose(dataset.features[200:].mean(axis=0), centers[1], atol=0.03)


def test_explicit_generator_has_intentional_stateful_semantics() -> None:
    generator = np.random.default_rng(22)
    first = make_regression(n_samples=10, n_features=2, seed=generator)
    second = make_regression(n_samples=10, n_features=2, seed=generator)
    replay = make_regression(n_samples=10, n_features=2, seed=22)

    np.testing.assert_array_equal(first.features, replay.features)
    assert not np.array_equal(first.features, second.features)


@pytest.mark.parametrize(
    ("factory", "kwargs", "exception", "message"),
    [
        (make_regression, {"n_samples": 0}, ValueError, "n_samples"),
        (make_regression, {"n_features": True}, TypeError, "n_features"),
        (make_regression, {"n_features": 2, "n_informative": 3}, ValueError, "must not exceed"),
        (make_regression, {"noise": -1.0}, ValueError, "noise"),
        (make_regression, {"coefficient_scale": np.inf}, ValueError, "coefficient_scale"),
        (make_regression, {"bias": np.nan}, ValueError, "bias"),
        (make_classification, {"n_samples": 1}, ValueError, "at least 2"),
        (make_classification, {"positive_fraction": 0.0}, ValueError, "strictly greater"),
        (make_classification, {"positive_fraction": 1.0}, ValueError, r"\[0, 1\)"),
        (make_classification, {"flip_y": 1.1}, ValueError, "flip_y"),
        (make_classification, {"class_sep": -0.1}, ValueError, "class_sep"),
        (make_blobs, {"centers": 0}, ValueError, "centers"),
        (make_blobs, {"n_samples": 2, "centers": 3}, ValueError, "at least"),
        (make_blobs, {"center_box": (1.0, 1.0)}, ValueError, "center_box"),
        (make_blobs, {"centers": [[0.0], [1.0]], "cluster_std": [1.0]}, ValueError, "one value"),
        (make_blobs, {"cluster_std": 0.0}, ValueError, "strictly positive"),
        (make_blobs, {"cluster_std": "wide"}, TypeError, "real numeric"),
        (make_regression, {"seed": -1}, ValueError, "non-negative"),
        (make_regression, {"seed": "seed"}, TypeError, "integer"),
    ],
)
def test_generators_reject_invalid_parameters(
    factory: object,
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        factory(**kwargs)  # type: ignore[operator]
