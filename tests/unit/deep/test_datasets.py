"""Determinism, invariants, and validation tests for deep-learning generators."""

from collections.abc import Callable

import numpy as np
import pytest

from learning_atlas.deep.datasets import make_temporal_xor, make_two_moons, make_xor

pytestmark = pytest.mark.unit


def _assert_legacy_rng_state_equal(left: tuple[object, ...], right: tuple[object, ...]) -> None:
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: make_xor(128, noise=0.15, seed=5),
        lambda: make_two_moons(129, noise=0.1, seed=6),
    ],
    ids=("xor", "two-moons"),
)
def test_planar_generators_replay_exactly_and_return_immutable_arrays(
    factory: Callable[[], object],
) -> None:
    first = factory()
    second = factory()

    np.testing.assert_array_equal(first.features, second.features)  # type: ignore[attr-defined]
    np.testing.assert_array_equal(first.targets, second.targets)  # type: ignore[attr-defined]
    assert first.features.dtype == np.float64  # type: ignore[attr-defined]
    assert first.targets.dtype == np.int64  # type: ignore[attr-defined]
    assert first.features.flags.writeable is False  # type: ignore[attr-defined]
    assert first.targets.flags.writeable is False  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="read-only"):
        first.features[0, 0] = 99.0  # type: ignore[attr-defined]


def test_xor_without_noise_has_exact_quadrant_labels() -> None:
    dataset = make_xor(256, noise=0.0, seed=11)

    expected = (dataset.features[:, 0] * dataset.features[:, 1] < 0.0).astype(np.int64)

    np.testing.assert_array_equal(dataset.targets, expected)
    assert dataset.features.shape == (256, 2)
    assert set(np.unique(dataset.features)) == {-1.0, 1.0}
    assert set(np.unique(dataset.targets)) == {0, 1}


def test_two_moons_without_noise_has_balanced_labels_and_geometric_support() -> None:
    dataset = make_two_moons(129, noise=0.0, seed=12)
    counts = np.bincount(dataset.targets, minlength=2)

    assert dataset.features.shape == (129, 2)
    assert abs(int(counts[0]) - int(counts[1])) == 1
    assert np.all((-1.0 <= dataset.features[:, 0]) & (dataset.features[:, 0] <= 2.0))
    assert np.all((-1.0 <= dataset.features[:, 1]) & (dataset.features[:, 1] <= 1.0))


def test_temporal_xor_encodes_variable_lengths_padding_and_target_rule() -> None:
    dataset = make_temporal_xor(
        256,
        min_length=3,
        max_length=11,
        seed=17,
    )
    positions = np.arange(dataset.sequences.shape[1])[np.newaxis, :]
    valid = positions < dataset.lengths[:, np.newaxis]
    tokens = dataset.sequences[:, :, 0]
    final_tokens = tokens[np.arange(len(tokens)), dataset.lengths - 1]

    assert dataset.sequences.shape == (256, 11, 1)
    assert dataset.lengths.shape == (256,)
    assert dataset.targets.shape == (256,)
    assert dataset.pad_value == 0.0
    assert dataset.lengths.min() >= 3
    assert dataset.lengths.max() <= 11
    assert len(np.unique(dataset.lengths)) > 1
    assert set(np.unique(tokens[valid])) == {-1.0, 1.0}
    np.testing.assert_array_equal(tokens[~valid], 0.0)
    np.testing.assert_array_equal(dataset.targets, (tokens[:, 0] != final_tokens).astype(np.int64))
    assert dataset.sequences.flags.writeable is False
    assert dataset.lengths.flags.writeable is False
    assert dataset.targets.flags.writeable is False


def test_distinct_seeds_change_each_generated_problem() -> None:
    xor_first = make_xor(64, seed=1)
    xor_second = make_xor(64, seed=2)
    moons_first = make_two_moons(64, seed=1)
    moons_second = make_two_moons(64, seed=2)
    sequence_first = make_temporal_xor(64, seed=1)
    sequence_second = make_temporal_xor(64, seed=2)

    assert not np.array_equal(xor_first.features, xor_second.features)
    assert not np.array_equal(moons_first.features, moons_second.features)
    assert not np.array_equal(sequence_first.sequences, sequence_second.sequences)


def test_generators_do_not_mutate_numpy_legacy_global_rng() -> None:
    np.random.seed(8675309)
    before = np.random.get_state()

    make_xor(32, seed=3)
    make_two_moons(32, seed=4)
    make_temporal_xor(32, seed=5)

    after = np.random.get_state()
    _assert_legacy_rng_state_equal(before, after)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: make_xor(7, seed=0), ValueError, "at least 8"),
        (lambda: make_xor(True, seed=0), TypeError, "integer"),
        (lambda: make_xor(8, noise=-0.1, seed=0), ValueError, "non-negative"),
        (lambda: make_xor(8, noise=np.nan, seed=0), ValueError, "finite"),
        (lambda: make_two_moons(7, seed=0), ValueError, "at least 8"),
        (lambda: make_two_moons(8, noise=np.inf, seed=0), ValueError, "finite"),
        (lambda: make_temporal_xor(7, seed=0), ValueError, "at least 8"),
        (lambda: make_temporal_xor(True, seed=0), TypeError, "integer"),
        (lambda: make_temporal_xor(8, min_length=1, seed=0), ValueError, "at least 2"),
        (
            lambda: make_temporal_xor(8, min_length=5, max_length=4, seed=0),
            ValueError,
            "max_length",
        ),
    ],
)
def test_invalid_generator_arguments_raise_actionable_errors(
    factory: Callable[[], object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        factory()
