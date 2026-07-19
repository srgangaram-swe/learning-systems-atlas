"""Small data-integrity helpers shared by benchmark domains."""

import hashlib

import numpy as np
from numpy.typing import NDArray


def array_fingerprint(*arrays: NDArray[np.generic]) -> str:
    """Hash array shape, dtype, and contiguous values into a stable dataset identity."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()
