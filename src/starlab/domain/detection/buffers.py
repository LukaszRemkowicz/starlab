"""Tile-sized arrays whose pages return to the OS as soon as the last view is dropped."""

import mmap
from collections.abc import Sequence

import numpy as np
from numpy.typing import DTypeLike, NDArray


def anonymous_array[T: np.generic](shape: Sequence[int], dtype: type[T] | DTypeLike) -> NDArray[T]:
    """Zero-filled array on its own anonymous mapping.

    The system allocator caches freed multi-megabyte blocks, so a burst of tile buffers would
    stay resident long after the scan; an unmapped mapping is returned immediately.
    """
    count: int = int(np.prod(shape))
    size: int = max(count * np.dtype(dtype).itemsize, 1)
    return np.frombuffer(mmap.mmap(-1, size), dtype, count).reshape(shape)
