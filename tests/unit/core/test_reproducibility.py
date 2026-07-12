"""Seed and environment-provenance tests."""

from pathlib import Path

import numpy as np
import pytest

from learning_atlas.core.reproducibility import (
    derive_seed,
    environment_metadata,
    generator_for_seed,
    git_state,
)

pytestmark = pytest.mark.unit


def test_explicit_generators_replay_without_global_state() -> None:
    first = generator_for_seed(91)
    second = generator_for_seed(91)
    np.testing.assert_allclose(first.normal(size=3), second.normal(size=3))


def test_derived_streams_are_distinct_and_stable() -> None:
    first = [derive_seed(42, stream) for stream in range(4)]
    second = [derive_seed(42, stream) for stream in range(4)]
    assert first == second
    assert len(set(first)) == 4


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_must_fit_uint32(seed: int) -> None:
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        generator_for_seed(seed)


def test_stream_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="stream"):
        derive_seed(42, -1)


def test_git_state_degrades_gracefully_outside_repository(tmp_path: Path) -> None:
    state = git_state(tmp_path)
    assert state.commit is None
    assert state.dirty is None


def test_environment_metadata_records_required_provenance() -> None:
    metadata = environment_metadata()
    assert metadata["python"]
    assert metadata["platform"]
    assert metadata["git"]
    assert metadata["packages"]["scikit-learn"]
