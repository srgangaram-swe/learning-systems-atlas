"""Deterministic, truth-bearing datasets for unsupervised evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from learning_atlas.core.reproducibility import derive_named_seed
from learning_atlas.core.validation import FloatArray
from learning_atlas.supervised.datasets import make_blobs

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class StructureDataset:
    """Feature observations plus truth reserved for retrospective evaluation."""

    name: str
    features: FloatArray
    evaluation_labels: IntArray
    details: dict[str, JsonValue]


def _sample_count(n_samples: int, *, minimum: int = 30) -> int:
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        msg = "n_samples must be an integer"
        raise TypeError(msg)
    if int(n_samples) < minimum:
        msg = f"n_samples must be at least {minimum}"
        raise ValueError(msg)
    return int(n_samples)


def _freeze(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)


def make_moons(*, n_samples: int = 300, noise: float = 0.06, seed: int = 0) -> StructureDataset:
    """Generate two interlocking half circles without a framework dependency."""

    sample_count = _sample_count(n_samples)
    if not np.isfinite(noise) or noise < 0.0:
        msg = "noise must be a finite non-negative value"
        raise ValueError(msg)
    generator = np.random.default_rng(seed)
    upper_count = sample_count // 2
    lower_count = sample_count - upper_count
    upper_angles = generator.uniform(0.0, np.pi, size=upper_count)
    lower_angles = generator.uniform(0.0, np.pi, size=lower_count)
    upper = np.column_stack((np.cos(upper_angles), np.sin(upper_angles)))
    lower = np.column_stack((1.0 - np.cos(lower_angles), 0.5 - np.sin(lower_angles)))
    features = np.asarray(np.vstack((upper, lower)), dtype=np.float64)
    features += generator.normal(scale=float(noise), size=features.shape)
    labels = np.concatenate(
        (np.zeros(upper_count, dtype=np.int64), np.ones(lower_count, dtype=np.int64))
    )
    order = generator.permutation(sample_count)
    features = np.asarray(features[order], dtype=np.float64)
    labels = np.asarray(labels[order], dtype=np.int64)
    _freeze(features, labels)
    return StructureDataset(
        name="two_moons",
        features=features,
        evaluation_labels=labels,
        details={"noise": float(noise), "clusters": 2},
    )


def make_anisotropic_blobs(*, n_samples: int = 300, seed: int = 0) -> StructureDataset:
    """Generate elongated, rotated clusters with known retrospective assignments."""

    sample_count = _sample_count(n_samples)
    generated = make_blobs(
        n_samples=sample_count,
        centers=np.asarray([[-4.0, -1.5], [0.0, 3.0], [4.0, -1.0]]),
        cluster_std=np.asarray([0.55, 0.65, 0.50]),
        seed=seed,
    )
    transform = np.asarray([[0.75, -0.85], [0.30, 1.35]], dtype=np.float64)
    features = np.asarray(generated.features @ transform, dtype=np.float64)
    labels = np.asarray(generated.targets, dtype=np.int64).copy()
    _freeze(features, labels)
    return StructureDataset(
        name="anisotropic_blobs",
        features=features,
        evaluation_labels=labels,
        details={"clusters": 3, "transform": transform.tolist()},
    )


def make_variable_density_blobs(*, n_samples: int = 300, seed: int = 0) -> StructureDataset:
    """Generate clusters with deliberately unequal densities."""

    sample_count = _sample_count(n_samples)
    generated = make_blobs(
        n_samples=sample_count,
        centers=np.asarray([[-4.5, -2.5], [0.0, 3.5], [4.5, -1.5]]),
        cluster_std=np.asarray([0.28, 0.62, 1.05]),
        seed=seed,
    )
    features = np.asarray(generated.features, dtype=np.float64).copy()
    labels = np.asarray(generated.targets, dtype=np.int64).copy()
    _freeze(features, labels)
    return StructureDataset(
        name="variable_density_blobs",
        features=features,
        evaluation_labels=labels,
        details={"clusters": 3, "cluster_std": [0.28, 0.62, 1.05]},
    )


def make_isotropic_blobs(*, n_samples: int = 300, seed: int = 0) -> StructureDataset:
    """Generate a well-separated convex reference problem."""

    sample_count = _sample_count(n_samples)
    generated = make_blobs(
        n_samples=sample_count,
        centers=np.asarray([[-4.0, -3.0], [0.0, 4.0], [4.5, -2.0]]),
        cluster_std=0.55,
        seed=seed,
    )
    features = np.asarray(generated.features, dtype=np.float64).copy()
    labels = np.asarray(generated.targets, dtype=np.int64).copy()
    _freeze(features, labels)
    return StructureDataset(
        name="isotropic_blobs",
        features=features,
        evaluation_labels=labels,
        details={"clusters": 3, "cluster_std": 0.55},
    )


def clustering_suite(*, n_samples: int, seed: int) -> tuple[StructureDataset, ...]:
    """Build heterogeneous structures with order-independent dataset streams."""

    factories = (
        ("isotropic_blobs", make_isotropic_blobs),
        ("two_moons", make_moons),
        ("anisotropic_blobs", make_anisotropic_blobs),
        ("variable_density_blobs", make_variable_density_blobs),
    )
    return tuple(
        factory(n_samples=n_samples, seed=derive_named_seed(seed, "unsupervised", "data", name))
        for name, factory in factories
    )
