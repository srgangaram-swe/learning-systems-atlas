"""Deterministic NumPy-only synthetic datasets with inspectable ground truth."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.validation import FloatArray, validate_features

IntArray = NDArray[np.int64]
RandomState: TypeAlias = int | np.random.Generator


@dataclass(frozen=True, slots=True)
class RegressionDataset:
    """Synthetic linear regression observations and their generating parameters."""

    features: FloatArray
    targets: FloatArray
    coefficients: FloatArray
    intercept: float
    noise: FloatArray


@dataclass(frozen=True, slots=True)
class ClassificationDataset:
    """Binary linear classification observations and their latent clean boundary."""

    features: FloatArray
    targets: IntArray
    clean_targets: IntArray
    coefficients: FloatArray
    intercept: float
    logits: FloatArray


@dataclass(frozen=True, slots=True)
class BlobsDataset:
    """Isotropic Gaussian clusters and the centers used to generate them."""

    features: FloatArray
    targets: IntArray
    centers: FloatArray
    cluster_std: FloatArray


def _rng(seed: RandomState) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        msg = "seed must be an integer or numpy.random.Generator"
        raise TypeError(msg)
    if int(seed) < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)
    return np.random.default_rng(int(seed))


def _positive_integer(value: int, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if int(value) < minimum:
        msg = f"{name} must be at least {minimum}"
        raise ValueError(msg)
    return int(value)


def _non_negative(value: float, *, name: str) -> float:
    if not np.isfinite(value) or value < 0.0:
        msg = f"{name} must be a finite non-negative value"
        raise ValueError(msg)
    return float(value)


def _probability(value: float, *, name: str, inclusive_one: bool = True) -> float:
    upper_valid = value <= 1.0 if inclusive_one else value < 1.0
    if not np.isfinite(value) or value < 0.0 or not upper_valid:
        interval = "[0, 1]" if inclusive_one else "[0, 1)"
        msg = f"{name} must be a finite value in {interval}"
        raise ValueError(msg)
    return float(value)


def _readonly(array: NDArray[np.generic]) -> None:
    array.setflags(write=False)


def make_regression(
    *,
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int | None = None,
    noise: float = 0.0,
    bias: float = 0.0,
    coefficient_scale: float = 10.0,
    shuffle: bool = True,
    seed: RandomState = 0,
) -> RegressionDataset:
    """Generate a linear regression problem with exact coefficient provenance."""

    sample_count = _positive_integer(n_samples, name="n_samples")
    feature_count = _positive_integer(n_features, name="n_features")
    informative_count = (
        feature_count
        if n_informative is None
        else _positive_integer(n_informative, name="n_informative")
    )
    if informative_count > feature_count:
        msg = "n_informative must not exceed n_features"
        raise ValueError(msg)
    noise_scale = _non_negative(noise, name="noise")
    coefficient_magnitude = _non_negative(coefficient_scale, name="coefficient_scale")
    if not np.isfinite(bias):
        msg = "bias must be finite"
        raise ValueError(msg)

    generator = _rng(seed)
    features = np.asarray(generator.normal(size=(sample_count, feature_count)), dtype=np.float64)
    coefficients = np.zeros(feature_count, dtype=np.float64)
    magnitudes = generator.uniform(0.25, 1.0, size=informative_count) * coefficient_magnitude
    signs = generator.choice(np.asarray([-1.0, 1.0]), size=informative_count)
    coefficients[:informative_count] = magnitudes * signs
    if shuffle:
        permutation = generator.permutation(feature_count)
        features = features[:, permutation]
        coefficients = coefficients[permutation]
    generated_noise = np.asarray(generator.normal(scale=noise_scale, size=sample_count))
    targets = np.asarray(features @ coefficients + float(bias) + generated_noise, dtype=np.float64)

    for array in (features, targets, coefficients, generated_noise):
        _readonly(array)
    return RegressionDataset(
        features=features,
        targets=targets,
        coefficients=coefficients,
        intercept=float(bias),
        noise=generated_noise,
    )


def make_classification(
    *,
    n_samples: int = 200,
    n_features: int = 10,
    n_informative: int | None = None,
    class_sep: float = 1.0,
    positive_fraction: float = 0.5,
    flip_y: float = 0.0,
    shuffle: bool = True,
    seed: RandomState = 0,
) -> ClassificationDataset:
    """Generate a binary problem with a known, linearly separating hyperplane.

    ``class_sep`` moves observations away from the latent hyperplane without
    changing their clean labels. ``flip_y`` then optionally corrupts labels while
    retaining ``clean_targets`` for recovery and robustness tests.
    """

    sample_count = _positive_integer(n_samples, name="n_samples", minimum=2)
    feature_count = _positive_integer(n_features, name="n_features")
    informative_count = (
        feature_count
        if n_informative is None
        else _positive_integer(n_informative, name="n_informative")
    )
    if informative_count > feature_count:
        msg = "n_informative must not exceed n_features"
        raise ValueError(msg)
    separation = _non_negative(class_sep, name="class_sep")
    fraction = _probability(positive_fraction, name="positive_fraction", inclusive_one=False)
    if fraction == 0.0:
        msg = "positive_fraction must be strictly greater than 0"
        raise ValueError(msg)
    flip_probability = _probability(flip_y, name="flip_y")

    generator = _rng(seed)
    features = np.asarray(generator.normal(size=(sample_count, feature_count)), dtype=np.float64)
    coefficients = np.zeros(feature_count, dtype=np.float64)
    coefficients[:informative_count] = generator.normal(size=informative_count)
    norm = float(np.linalg.norm(coefficients))
    if norm == 0.0:  # Practically unreachable, but protects the mathematical contract.
        coefficients[0] = 1.0
        norm = 1.0
    coefficients /= norm

    initial_scores = features @ coefficients
    threshold = float(np.quantile(initial_scores, 1.0 - fraction, method="linear"))
    initial_labels = (initial_scores >= threshold).astype(np.int64)
    direction = coefficients / float(np.linalg.norm(coefficients))
    signed_class = 2.0 * initial_labels.astype(np.float64) - 1.0
    features += separation * signed_class[:, np.newaxis] * direction[np.newaxis, :]
    intercept = -threshold
    logits = np.asarray(features @ coefficients + intercept, dtype=np.float64)
    clean_targets = (logits >= 0.0).astype(np.int64)
    targets = clean_targets.copy()
    if flip_probability > 0.0:
        flip_mask = generator.random(sample_count) < flip_probability
        targets[flip_mask] = 1 - targets[flip_mask]

    if shuffle:
        permutation = generator.permutation(feature_count)
        features = features[:, permutation]
        coefficients = coefficients[permutation]
    for array in (features, targets, clean_targets, coefficients, logits):
        _readonly(array)
    return ClassificationDataset(
        features=features,
        targets=targets,
        clean_targets=clean_targets,
        coefficients=coefficients,
        intercept=intercept,
        logits=logits,
    )


def make_blobs(
    *,
    n_samples: int = 100,
    n_features: int = 2,
    centers: int | ArrayLike = 3,
    cluster_std: float | ArrayLike = 1.0,
    center_box: tuple[float, float] = (-10.0, 10.0),
    shuffle: bool = True,
    seed: RandomState = 0,
) -> BlobsDataset:
    """Generate Gaussian clusters with deterministic centers and assignments."""

    sample_count = _positive_integer(n_samples, name="n_samples")
    generator = _rng(seed)
    if isinstance(centers, (int, np.integer)) and not isinstance(centers, bool):
        center_count = _positive_integer(int(centers), name="centers")
        feature_count = _positive_integer(n_features, name="n_features")
        lower, upper = center_box
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            msg = "center_box must contain finite increasing bounds"
            raise ValueError(msg)
        generated_centers = np.asarray(
            generator.uniform(lower, upper, size=(center_count, feature_count)), dtype=np.float64
        )
    else:
        generated_centers = validate_features(centers)
        center_count, feature_count = generated_centers.shape
        generated_centers = generated_centers.copy()
    if sample_count < center_count:
        msg = "n_samples must be at least the number of centers"
        raise ValueError(msg)

    raw_std = np.asarray(cluster_std)
    if not np.issubdtype(raw_std.dtype, np.number) or np.issubdtype(
        raw_std.dtype, np.complexfloating
    ):
        msg = "cluster_std must contain real numeric values"
        raise TypeError(msg)
    std = np.asarray(raw_std, dtype=np.float64)
    if std.ndim == 0:
        std = np.full(center_count, float(std), dtype=np.float64)
    if std.ndim != 1 or len(std) != center_count:
        msg = "cluster_std must be a scalar or contain one value per center"
        raise ValueError(msg)
    if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        msg = "cluster_std values must be finite and strictly positive"
        raise ValueError(msg)

    counts = np.full(center_count, sample_count // center_count, dtype=np.int64)
    counts[: sample_count % center_count] += 1
    feature_parts: list[FloatArray] = []
    target_parts: list[IntArray] = []
    for label, count in enumerate(counts):
        points = generator.normal(
            loc=generated_centers[label],
            scale=std[label],
            size=(int(count), feature_count),
        )
        feature_parts.append(np.asarray(points, dtype=np.float64))
        target_parts.append(np.full(int(count), label, dtype=np.int64))
    features = np.vstack(feature_parts)
    targets = np.concatenate(target_parts)
    if shuffle:
        permutation = generator.permutation(sample_count)
        features = features[permutation]
        targets = targets[permutation]

    for array in (features, targets, generated_centers, std):
        _readonly(array)
    return BlobsDataset(
        features=features,
        targets=targets,
        centers=generated_centers,
        cluster_std=std,
    )
