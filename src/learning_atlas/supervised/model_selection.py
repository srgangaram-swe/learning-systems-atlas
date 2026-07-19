"""Deterministic fold construction shared by from-scratch benchmarks."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.validation import validate_targets

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class FoldIndices:
    """One non-overlapping training/validation index boundary."""

    train: IntArray
    validation: IntArray


def _validate_fold_request(n_samples: int, n_splits: int, seed: int) -> None:
    if isinstance(n_samples, bool) or n_samples < 2:
        msg = "n_samples must be an integer of at least 2"
        raise ValueError(msg)
    if isinstance(n_splits, bool) or not 2 <= n_splits <= n_samples:
        msg = "n_splits must be an integer in [2, n_samples]"
        raise ValueError(msg)
    if isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        msg = "seed must be an integer in [0, 2**32 - 1]"
        raise ValueError(msg)


def _build_folds(validation_parts: tuple[IntArray, ...], n_samples: int) -> tuple[FoldIndices, ...]:
    all_indices = np.arange(n_samples, dtype=np.int64)
    folds: list[FoldIndices] = []
    for validation in validation_parts:
        mask = np.ones(n_samples, dtype=np.bool_)
        mask[validation] = False
        folds.append(FoldIndices(train=all_indices[mask], validation=validation))
    return tuple(folds)


def kfold_indices(n_samples: int, n_splits: int, *, seed: int) -> tuple[FoldIndices, ...]:
    """Return seeded K-fold indices with every sample held out exactly once."""

    _validate_fold_request(n_samples, n_splits, seed)
    shuffled = np.random.default_rng(seed).permutation(n_samples).astype(np.int64, copy=False)
    validation_parts = tuple(
        np.asarray(part, dtype=np.int64) for part in np.array_split(shuffled, n_splits)
    )
    return _build_folds(validation_parts, n_samples)


def stratified_kfold_indices(
    targets: ArrayLike,
    n_splits: int,
    *,
    seed: int,
) -> tuple[FoldIndices, ...]:
    """Return seeded folds that distribute every class across validation sets."""

    raw = np.asarray(targets)
    if raw.ndim != 1:
        msg = "targets must be 1D for stratified folds"
        raise ValueError(msg)
    observed = validate_targets(raw, n_samples=len(raw))
    _validate_fold_request(len(observed), n_splits, seed)
    classes, counts = np.unique(observed, return_counts=True)
    if len(classes) < 2:
        msg = "stratified folds require at least two classes"
        raise ValueError(msg)
    if int(np.min(counts)) < n_splits:
        msg = "every class must contain at least n_splits samples"
        raise ValueError(msg)

    root = np.random.SeedSequence(seed)
    class_seeds = root.spawn(len(classes))
    per_fold: list[list[IntArray]] = [[] for _ in range(n_splits)]
    for class_value, class_seed in zip(classes, class_seeds, strict=True):
        class_indices = np.flatnonzero(observed == class_value).astype(np.int64, copy=False)
        shuffled = np.random.default_rng(class_seed).permutation(class_indices)
        for fold_index, part in enumerate(np.array_split(shuffled, n_splits)):
            per_fold[fold_index].append(np.asarray(part, dtype=np.int64))

    order_seeds = root.spawn(n_splits)
    validation_parts: list[IntArray] = []
    for pieces, order_seed in zip(per_fold, order_seeds, strict=True):
        combined = np.concatenate(pieces)
        ordered = np.random.default_rng(order_seed).permutation(combined)
        validation_parts.append(np.asarray(ordered, dtype=np.int64))
    return _build_folds(tuple(validation_parts), len(observed))
