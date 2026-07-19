"""Dataset fingerprint behavior tests."""

import numpy as np
import pytest

from learning_atlas.core.data import array_fingerprint

pytestmark = pytest.mark.unit


def test_array_fingerprint_is_stable_and_sensitive_to_shape_dtype_and_values() -> None:
    base = np.array([[1.0, 2.0]], dtype=np.float64)
    assert array_fingerprint(base) == array_fingerprint(base.copy())
    assert array_fingerprint(base) != array_fingerprint(base.reshape(2, 1))
    assert array_fingerprint(base) != array_fingerprint(base.astype(np.float32))
    changed = base.copy()
    changed[0, 0] = 3.0
    assert array_fingerprint(base) != array_fingerprint(changed)
