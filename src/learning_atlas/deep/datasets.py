"""Deterministic NumPy-only generators for the deep-learning studies.

Planar tasks exercise the from-scratch autograd MLP; the temporal-XOR task
provides a variable-length, genuinely long-range sequence problem in which the
class depends on the first and the final token of every sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from learning_atlas.core.reproducibility import generator_for_seed

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _positive_count(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if value < minimum:
        msg = f"{name} must be at least {minimum}"
        raise ValueError(msg)
    return value


def _non_negative_scale(value: float, *, name: str) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale < 0.0:
        msg = f"{name} must be a finite non-negative value"
        raise ValueError(msg)
    return scale


def _readonly(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)


@dataclass(frozen=True, slots=True)
class PlanarDataset:
    """A two-dimensional classification task with integer labels."""

    name: str
    features: FloatArray
    targets: IntArray


@dataclass(frozen=True, slots=True)
class SequenceDataset:
    """Zero-padded variable-length sequences with explicit true lengths."""

    name: str
    sequences: FloatArray
    lengths: IntArray
    targets: IntArray
    pad_value: float


def make_xor(n_samples: int = 200, *, noise: float = 0.15, seed: int) -> PlanarDataset:
    """Sample the XOR quadrant problem, the canonical linearly inseparable task."""

    count = _positive_count(n_samples, name="n_samples", minimum=8)
    scale = _non_negative_scale(noise, name="noise")
    rng = generator_for_seed(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, 2))
    # Guarantee both labels even at the smallest supported sample count; this
    # avoids a seed-dependent invalid classification dataset.
    signs[0] = (1.0, 1.0)
    signs[1] = (1.0, -1.0)
    features = signs + rng.normal(0.0, scale, size=(count, 2))
    targets = (signs[:, 0] * signs[:, 1] < 0.0).astype(np.int64)
    _readonly(features, targets)
    return PlanarDataset(name="xor", features=features, targets=targets)


def make_two_moons(n_samples: int = 240, *, noise: float = 0.1, seed: int) -> PlanarDataset:
    """Sample two interleaving half-circles requiring a nonlinear boundary."""

    count = _positive_count(n_samples, name="n_samples", minimum=8)
    scale = _non_negative_scale(noise, name="noise")
    rng = generator_for_seed(seed)
    n_outer = count // 2
    n_inner = count - n_outer
    outer_angles = rng.uniform(0.0, np.pi, size=n_outer)
    inner_angles = rng.uniform(0.0, np.pi, size=n_inner)
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([1.0 - np.cos(inner_angles), 0.5 - np.sin(inner_angles)])
    features = np.concatenate([outer, inner]) + rng.normal(0.0, scale, size=(count, 2))
    targets = np.concatenate([np.zeros(n_outer, dtype=np.int64), np.ones(n_inner, dtype=np.int64)])
    shuffled = rng.permutation(count)
    features = np.asarray(features[shuffled], dtype=np.float64)
    targets = np.asarray(targets[shuffled], dtype=np.int64)
    _readonly(features, targets)
    return PlanarDataset(name="two_moons", features=features, targets=targets)


def make_temporal_xor(
    n_sequences: int = 1_000,
    *,
    min_length: int = 6,
    max_length: int = 24,
    seed: int,
) -> SequenceDataset:
    """Sample the temporal-XOR task: the label couples the first and last token.

    Every token is drawn from ``{-1, +1}`` so no magnitude cue marks the
    informative positions, and sequence lengths vary, so the final token has no
    fixed position in a zero-padded layout. The label is ``1`` exactly when the
    first and last tokens differ, which forces a sequence model to carry the
    opening token across the entire distractor tail.
    """

    count = _positive_count(n_sequences, name="n_sequences", minimum=8)
    shortest = _positive_count(min_length, name="min_length", minimum=2)
    longest = _positive_count(max_length, name="max_length", minimum=2)
    if longest < shortest:
        msg = "max_length must be greater than or equal to min_length"
        raise ValueError(msg)
    rng = generator_for_seed(seed)
    lengths = rng.integers(shortest, longest + 1, size=count, dtype=np.int64)
    tokens = rng.choice(np.asarray([-1.0, 1.0]), size=(count, longest))
    tokens[0, 0] = 1.0
    tokens[0, lengths[0] - 1] = 1.0
    tokens[1, 0] = 1.0
    tokens[1, lengths[1] - 1] = -1.0
    sequences = np.zeros((count, longest, 1), dtype=np.float64)
    positions = np.arange(longest)
    valid = positions[np.newaxis, :] < lengths[:, np.newaxis]
    sequences[:, :, 0] = np.where(valid, tokens, 0.0)
    first_tokens = sequences[:, 0, 0]
    last_tokens = sequences[np.arange(count), lengths - 1, 0]
    targets = (first_tokens != last_tokens).astype(np.int64)
    _readonly(sequences, lengths, targets)
    return SequenceDataset(
        name="temporal_xor",
        sequences=sequences,
        lengths=lengths,
        targets=targets,
        pad_value=0.0,
    )
