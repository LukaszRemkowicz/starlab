"""Sparse peak screens with the same connected components as the SciPy path."""

import numpy as np
from numba import njit
from scipy import ndimage

from .peak_filters import uniform7_at

SPARSE_FLOOR_LIMIT = 256


@njit(cache=True, nogil=True)
def _local_mean(lum, allowed, y, x, radius):
    """Match uniform_filter's axis order and intermediate float32 arrays."""
    size = radius * 2 + 1
    mean = square = count = 0.0
    for dx in range(-radius, radius + 1):
        xx = min(max(x + dx, 0), lum.shape[1] - 1)
        column = column_square = column_count = 0.0
        for dy in range(-radius, radius + 1):
            yy = min(max(y + dy, 0), lum.shape[0] - 1)
            if allowed is None or allowed[yy, xx]:
                value = lum[yy, xx]
                column += float(value)
                column_square += float(np.float32(value * value))
                column_count += 1.0
        mean += float(np.float32(column / size))
        square += float(np.float32(column_square / size))
        count += float(np.float32(column_count / size))
    return np.float32(mean / size), np.float32(square / size), np.float32(count / size)


@njit(cache=True, nogil=True)
def _select(rgb, lum, allowed, threshold, support_threshold, coarse, global_floor):
    height, width = lum.shape
    selected = np.zeros((height, width), np.bool_)
    unresolved = np.empty((SPARSE_FLOOR_LIMIT, 2), np.int64)
    unresolved_count = 0
    need_coarse = False
    for y in range(height):
        for x in range(width):
            if allowed is not None and not allowed[y, x]:
                continue
            if max(rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2]) < threshold:
                continue
            value = lum[y, x]
            # Every allowed local minimum is at least the global minimum,
            # including when the latter belongs to an excluded pixel.
            if np.float32(value - global_floor) < np.float32(0.13):
                continue
            if coarse and value < np.float32(0.75):
                continue
            maximum = True
            support = 0
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    if allowed is None or allowed[yy, xx]:
                        if lum[yy, xx] > value:
                            maximum = False
                        if max(rgb[yy, xx, 0], rgb[yy, xx, 1], rgb[yy, xx, 2]) > support_threshold:
                            support += 1
            if not maximum or support < 3:
                continue
            if coarse:
                mean, _, _ = _local_mean(lum, None, y, x, 1)
                # Rolling double sums can round differently at a float32 tie.
                if abs(float(mean) - 0.6) <= 0.000001:
                    need_coarse = True
                elif mean < np.float32(0.6):
                    continue
            # A single qualifying sample proves the 129x129 minimum test.
            # Bounded probes avoid quadratic work on broad flat plateaus;
            # unresolved cases get a complete sparse search below.
            contrast = False
            for dy in (-64, 0, 64):
                yy = min(max(y + dy, 0), height - 1)
                for dx in (-64, 0, 64):
                    xx = min(max(x + dx, 0), width - 1)
                    if allowed is None or allowed[yy, xx]:
                        if np.float32(value - lum[yy, xx]) >= np.float32(0.13):
                            contrast = True
                            break
                if contrast:
                    break
            if not contrast:
                if unresolved_count < SPARSE_FLOOR_LIMIT:
                    unresolved[unresolved_count, 0] = y
                    unresolved[unresolved_count, 1] = x
                unresolved_count += 1
            selected[y, x] = True
    return selected, unresolved, unresolved_count, need_coarse


@njit(cache=True, nogil=True)
def _resolve_floor(selected, lum, allowed, unresolved):
    """Evaluate the full minimum predicate only at unproven candidates."""
    height, width = lum.shape
    for index in range(len(unresolved)):
        y, x = unresolved[index]
        value = lum[y, x]
        contrast = False
        for yy in range(max(0, y - 64), min(height, y + 65)):
            for xx in range(max(0, x - 64), min(width, x + 65)):
                if allowed is None or allowed[yy, xx]:
                    if np.float32(value - lum[yy, xx]) >= np.float32(0.13):
                        contrast = True
                        break
            if contrast:
                break
        selected[y, x] = contrast


@njit(cache=True, nogil=True)
def _contrast(lum, allowed, centers):
    keep = np.ones(len(centers), np.bool_)
    uncertain = np.zeros(len(centers), np.bool_)
    for index in range(len(centers)):
        y, x = centers[index]
        value = lum[y, x]
        if value >= np.float32(0.75):
            continue
        mean, square, count = _local_mean(lum, allowed, y, x, 3)
        if allowed is not None:
            if count > 0:
                mean = np.float32(mean / count)
                square = np.float32(square / count)
            else:
                mean = square = np.float32(0)
        variance = np.float32(square - np.float32(mean * mean))
        std = np.float32(np.sqrt(max(variance, np.float32(0))))
        score = np.float32(np.float32(value - mean) + std)
        keep[index] = score >= np.float32(0.08)
        # This generous bound includes amplified rounding in masked variance
        # near zero. Only boundary decisions need the full rolling filters.
        uncertainty = 0.002 * (1 + abs(float(mean)) + abs(float(square)))
        uncertain[index] = not np.isfinite(score) or abs(float(score) - 0.08) <= uncertainty
    return keep, uncertain


def peak_positions(image, threshold, core, coarse=False):
    """Return global component bounding-box centers, retaining halo topology."""
    lum, allowed = image.lum, image.allowed
    selected, unresolved, unresolved_count, need_coarse = _select(
        image.rgb,
        lum,
        allowed,
        np.float32(threshold),
        np.float32(threshold * 0.05),
        coarse,
        np.float32(lum.min()),
    )
    # Dense plateaus use the linear-time filter rather than one window per
    # pixel. Sparse cases avoid allocating and filtering an entire tile.
    if 0 < unresolved_count <= min(SPARSE_FLOOR_LIMIT, max(8, lum.size // 64)):
        _resolve_floor(selected, lum, allowed, unresolved[:unresolved_count])
    elif unresolved_count:
        floor = ndimage.minimum_filter(
            lum if allowed is None else np.where(allowed, lum, np.inf),
            size=129,
            mode="nearest",
        )
        selected &= lum - floor >= np.float32(0.13)
    if need_coarse:
        selected &= ndimage.uniform_filter(lum, size=3, mode="nearest") >= np.float32(0.6)
    labels, _ = ndimage.label(selected)
    centers = []
    for region in ndimage.find_objects(labels):
        if region is not None:
            y = (region[0].start + region[0].stop - 1) // 2
            x = (region[1].start + region[1].stop - 1) // 2
            if core[0] <= x + image.ox < core[2] and core[1] <= y + image.oy < core[3]:
                centers.append((y, x))
    if not centers:
        return []
    centers = np.asarray(centers, dtype=np.int64)
    keep = np.ones(len(centers), np.bool_)
    if not coarse:
        keep, uncertain = _contrast(lum, allowed, centers)
        if uncertain.any():
            mean, square, count = uniform7_at(lum, allowed, centers[uncertain]).T
            if allowed is not None:
                mean = np.divide(mean, count, out=np.zeros_like(mean), where=count > 0)
                square = np.divide(square, count, out=np.zeros_like(square), where=count > 0)
            y, x = centers[uncertain].T
            std = np.sqrt(np.maximum(square - mean**2, 0))
            keep[uncertain] = (lum[y, x] >= 0.75) | (lum[y, x] - mean + std >= 0.08)
    return [(int(x) + image.ox, int(y) + image.oy) for (y, x), yes in zip(centers, keep) if yes]
