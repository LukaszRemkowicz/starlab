"""Exact rolling 7x7 filter values at a sparse set of pixel centers."""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def uniform7_at(lum, allowed, centers):
    """Return mean, square mean and mask fraction in the input center order.

    Match SciPy NI_UniformFilter1D: accumulate doubles from the line origin,
    update with added-minus-removed, divide by seven, then store float32.
    The axis-0 output is rounded before axis 1. Only requested rows need the
    horizontal pass; the vertical pass keeps three width-sized accumulators.
    """
    result = np.empty((len(centers), 3), np.float32)
    if len(centers) == 0:
        return result
    height, width = lum.shape
    order = np.argsort(centers[:, 0] * width + centers[:, 1])
    last_y = centers[order[-1], 0]
    columns = min(width, np.max(centers[:, 1]) + 4)
    means = np.zeros(columns, np.float64)
    squares = np.zeros(columns, np.float64)
    counts = np.zeros(0 if allowed is None else columns, np.float64)
    row_mean = np.empty(columns, np.float32)
    row_square = np.empty(columns, np.float32)
    row_count = np.empty(0 if allowed is None else columns, np.float32)
    for dy in range(-3, 4):
        yy = min(max(dy, 0), height - 1)
        for x in range(columns):
            if allowed is None or allowed[yy, x]:
                value = lum[yy, x]
                means[x] += np.float64(value)
                squares[x] += np.float64(np.float32(value * value))
                if allowed is not None:
                    counts[x] += 1.0
    target = 0
    for y in range(last_y + 1):
        if y > 0:
            added_y, removed_y = min(y + 3, height - 1), max(y - 4, 0)
            for x in range(columns):
                added = removed = added_square = removed_square = 0.0
                added_count = removed_count = 0.0
                if allowed is None or allowed[added_y, x]:
                    value = lum[added_y, x]
                    added = np.float64(value)
                    added_square = np.float64(np.float32(value * value))
                    added_count = 1.0
                if allowed is None or allowed[removed_y, x]:
                    value = lum[removed_y, x]
                    removed = np.float64(value)
                    removed_square = np.float64(np.float32(value * value))
                    removed_count = 1.0
                means[x] += added - removed
                squares[x] += added_square - removed_square
                if allowed is not None:
                    counts[x] += added_count - removed_count
        if centers[order[target], 0] != y:
            continue
        end = target + 1
        while end < len(order) and centers[order[end], 0] == y:
            end += 1
        last_x = centers[order[end - 1], 1]
        for x in range(min(width, last_x + 4)):
            row_mean[x] = np.float32(means[x] / 7.0)
            row_square[x] = np.float32(squares[x] / 7.0)
            if allowed is not None:
                row_count[x] = np.float32(counts[x] / 7.0)
        mean = square = count = 0.0
        for dx in range(-3, 4):
            xx = min(max(dx, 0), width - 1)
            mean += np.float64(row_mean[xx])
            square += np.float64(row_square[xx])
            if allowed is not None:
                count += np.float64(row_count[xx])
        for x in range(last_x + 1):
            if x > 0:
                added_x, removed_x = min(x + 3, width - 1), max(x - 4, 0)
                mean += np.float64(row_mean[added_x]) - np.float64(row_mean[removed_x])
                square += np.float64(row_square[added_x]) - np.float64(row_square[removed_x])
                if allowed is not None:
                    count += np.float64(row_count[added_x]) - np.float64(row_count[removed_x])
            while target < end and centers[order[target], 1] == x:
                index = order[target]
                result[index, 0] = np.float32(mean / 7.0)
                result[index, 1] = np.float32(square / 7.0)
                result[index, 2] = np.float32(1 if allowed is None else count / 7.0)
                target += 1
    return result
