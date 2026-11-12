"""Exact rolling 7x7 filter values at a sparse set of pixel centers."""

import numpy as np
from numpy.typing import NDArray

from ..array_types import BoolArray, DoubleArray, FloatArray, IntArray
from ..jit import njit


@njit(cache=True, nogil=True)
def uniform7_at(luminance: FloatArray, allowed: BoolArray | None, centers: IntArray) -> FloatArray:  # noqa: C901, PLR0915
    """Return mean, square mean and mask fraction in the input center order.

    Match SciPy NI_UniformFilter1D: accumulate doubles from the line origin,
    update with added-minus-removed, divide by seven, then store float32.
    The axis-0 output is rounded before axis 1. Only requested rows need the
    horizontal pass; the vertical pass keeps three width-sized accumulators.
    Keep this rolling state machine intact: splitting its axis updates would
    obscure accumulator lifetimes and SciPy's exact rounding/iteration order.
    """
    result: FloatArray = np.empty((len(centers), 3), np.float32)
    if len(centers) == 0:
        return result
    height, width = luminance.shape
    order: NDArray[np.intp] = np.argsort(centers[:, 0] * width + centers[:, 1])
    last_y: int = centers[order[-1], 0]
    columns: int = min(width, np.max(centers[:, 1]) + 4)
    means: DoubleArray = np.zeros(columns, np.float64)
    squares: DoubleArray = np.zeros(columns, np.float64)
    counts: DoubleArray = np.zeros(0 if allowed is None else columns, np.float64)
    row_mean: FloatArray = np.empty(columns, np.float32)
    row_square: FloatArray = np.empty(columns, np.float32)
    row_count: FloatArray = np.empty(0 if allowed is None else columns, np.float32)
    for dy in range(-3, 4):
        sample_y: int = min(max(dy, 0), height - 1)
        for x in range(columns):
            if allowed is None or allowed[sample_y, x]:
                value: np.float32 = luminance[sample_y, x]
                means[x] += np.float64(value)
                squares[x] += np.float64(np.float32(value * value))
                if allowed is not None:
                    counts[x] += 1.0
    target: int = 0
    for y in range(last_y + 1):
        if y > 0:
            added_y, removed_y = min(y + 3, height - 1), max(y - 4, 0)
            for x in range(columns):
                added: float = 0.0
                removed: float = 0.0
                added_square: float = 0.0
                removed_square: float = 0.0
                added_count: float = 0.0
                removed_count: float = 0.0
                if allowed is None or allowed[added_y, x]:
                    value = luminance[added_y, x]
                    added = np.float64(value)
                    added_square = np.float64(np.float32(value * value))
                    added_count = 1.0
                if allowed is None or allowed[removed_y, x]:
                    value = luminance[removed_y, x]
                    removed = np.float64(value)
                    removed_square = np.float64(np.float32(value * value))
                    removed_count = 1.0
                means[x] += added - removed
                squares[x] += added_square - removed_square
                if allowed is not None:
                    counts[x] += added_count - removed_count
        if centers[order[target], 0] != y:
            continue
        end: int = target + 1
        while end < len(order) and centers[order[end], 0] == y:
            end += 1
        last_x: int = centers[order[end - 1], 1]
        for x in range(min(width, last_x + 4)):
            row_mean[x] = np.float32(means[x] / 7.0)
            row_square[x] = np.float32(squares[x] / 7.0)
            if allowed is not None:
                row_count[x] = np.float32(counts[x] / 7.0)
        mean: float = 0.0
        square: float = 0.0
        count: float = 0.0
        for dx in range(-3, 4):
            sample_x: int = min(max(dx, 0), width - 1)
            mean += np.float64(row_mean[sample_x])
            square += np.float64(row_square[sample_x])
            if allowed is not None:
                count += np.float64(row_count[sample_x])
        for x in range(last_x + 1):
            if x > 0:
                added_x, removed_x = min(x + 3, width - 1), max(x - 4, 0)
                mean += np.float64(row_mean[added_x]) - np.float64(row_mean[removed_x])
                square += np.float64(row_square[added_x]) - np.float64(row_square[removed_x])
                if allowed is not None:
                    count += np.float64(row_count[added_x]) - np.float64(row_count[removed_x])
            while target < end and centers[order[target], 1] == x:
                index: np.intp = order[target]
                result[index, 0] = np.float32(mean / 7.0)
                result[index, 1] = np.float32(square / 7.0)
                result[index, 2] = np.float32(1 if allowed is None else count / 7.0)
                target += 1
    return result
