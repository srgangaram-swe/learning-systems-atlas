"""Bounded numerical contracts shared by the reinforcement-learning laboratory."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ReinforcementError(RuntimeError):
    """A validated experiment failed to converge or violated its runtime contract."""


def integer(value: int, name: str, minimum: int, maximum: int) -> int:
    """Reject bools and non-integral or out-of-budget public parameters."""

    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def scalar(value: float, name: str, minimum: float, maximum: float) -> float:
    """Require a finite scalar in a closed interval."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return float(value)


def vector(values: FloatArray, name: str, *, size: int | None = None) -> FloatArray:
    """Copy a finite, nonempty, bounded rank-one real vector."""

    array = np.asarray(values)
    if (
        array.ndim != 1
        or not 1 <= array.size <= 1_000_000
        or array.dtype.kind not in "fiu"
        or (size is not None and array.size != size)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite nonempty numeric vector of the required size")
    return np.array(array, dtype=np.float64, copy=True)
