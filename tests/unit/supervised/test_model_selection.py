"""Deterministic fold construction and validation tests."""

import numpy as np
import pytest

from learning_atlas.supervised.model_selection import kfold_indices, stratified_kfold_indices

pytestmark = pytest.mark.unit


def test_kfold_is_reproducible_complete_and_disjoint() -> None:
    first = kfold_indices(23, 5, seed=17)
    second = kfold_indices(23, 5, seed=17)

    assert len(first) == 5
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.train, right.train)
        np.testing.assert_array_equal(left.validation, right.validation)
        assert not set(left.train).intersection(left.validation)
    held_out = np.concatenate([fold.validation for fold in first])
    np.testing.assert_array_equal(np.sort(held_out), np.arange(23))
    assert (
        max(len(fold.validation) for fold in first) - min(len(fold.validation) for fold in first)
        <= 1
    )


def test_stratified_folds_preserve_every_class() -> None:
    targets = np.repeat(np.array([2.0, 7.0, 11.0]), [15, 10, 5])
    folds = stratified_kfold_indices(targets, 5, seed=9)

    for fold in folds:
        assert set(targets[fold.validation]) == {2.0, 7.0, 11.0}
        assert not set(fold.train).intersection(fold.validation)
    held_out = np.concatenate([fold.validation for fold in folds])
    np.testing.assert_array_equal(np.sort(held_out), np.arange(len(targets)))


@pytest.mark.parametrize(
    ("n_samples", "n_splits", "seed"),
    ((1, 2, 0), (5, 1, 0), (5, 6, 0), (5, True, 0), (5, 2, -1)),
)
def test_kfold_rejects_invalid_requests(n_samples: int, n_splits: int, seed: int) -> None:
    with pytest.raises(ValueError):
        kfold_indices(n_samples, n_splits, seed=seed)


def test_stratified_folds_reject_invalid_targets() -> None:
    with pytest.raises(ValueError, match="1D"):
        stratified_kfold_indices([[0, 1], [1, 0]], 2, seed=0)
    with pytest.raises(ValueError, match="two classes"):
        stratified_kfold_indices(np.zeros(6), 2, seed=0)
    with pytest.raises(ValueError, match="every class"):
        stratified_kfold_indices([0, 0, 0, 1], 2, seed=0)
