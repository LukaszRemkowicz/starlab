"""Sparse peak screens with the same connected components as the SciPy path."""

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..array_types import BoolArray, Bounds, FloatArray, IntArray
from ..jit import njit
from .candidate import Image
from .peak_filters import uniform7_at
from .records import IndexArray

SPARSE_FLOOR_LIMIT = 256


@njit(cache=True, nogil=True)
def _local_mean(
    luminance: FloatArray, allowed: BoolArray | None, y: int, x: int, radius: int
) -> tuple[np.float32, np.float32, np.float32]:
    """Match uniform_filter's axis order and intermediate float32 arrays."""
    size: int = radius * 2 + 1
    mean: float = 0.0
    square: float = 0.0
    count: float = 0.0
    for dx in range(-radius, radius + 1):
        sample_x: int = min(max(x + dx, 0), luminance.shape[1] - 1)
        column: float = 0.0
        column_square: float = 0.0
        column_count: float = 0.0
        for dy in range(-radius, radius + 1):
            sample_y: int = min(max(y + dy, 0), luminance.shape[0] - 1)
            if allowed is None or allowed[sample_y, sample_x]:
                value: np.float32 = luminance[sample_y, sample_x]
                column += float(value)
                column_square += float(np.float32(value * value))
                column_count += 1.0
        mean += float(np.float32(column / size))
        square += float(np.float32(column_square / size))
        count += float(np.float32(column_count / size))
    return np.float32(mean / size), np.float32(square / size), np.float32(count / size)


@njit(cache=True, nogil=True)
def _select(  # noqa: C901
    rgb: FloatArray,
    luminance: FloatArray,
    allowed: BoolArray | None,
    threshold: np.float32,
    support_threshold: np.float32,
    coarse: bool,
    global_floor: np.float32,
) -> tuple[BoolArray, IntArray, int, bool]:
    """Keep the ordered early-rejection/proof cascade together to bound sparse work."""
    height, width = luminance.shape
    selected: BoolArray = np.zeros((height, width), np.bool_)
    unresolved: IntArray = np.empty((SPARSE_FLOOR_LIMIT, 2), np.int64)
    unresolved_count: int = 0
    need_coarse: bool = False
    for y in range(height):
        for x in range(width):
            if allowed is not None and not allowed[y, x]:
                continue
            if max(rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2]) < threshold:
                continue
            value: np.float32 = luminance[y, x]
            # Every allowed local minimum is at least the global minimum,
            # including when the latter belongs to an excluded pixel.
            if np.float32(value - global_floor) < np.float32(0.13):
                continue
            if coarse and value < np.float32(0.75):
                continue
            maximum: bool = True
            support: int = 0
            sample_y: int
            for sample_y in range(max(0, y - 1), min(height, y + 2)):
                sample_x: int
                for sample_x in range(max(0, x - 1), min(width, x + 2)):
                    if allowed is None or allowed[sample_y, sample_x]:
                        if luminance[sample_y, sample_x] > value:
                            maximum = False
                        if (
                            max(
                                rgb[sample_y, sample_x, 0],
                                rgb[sample_y, sample_x, 1],
                                rgb[sample_y, sample_x, 2],
                            )
                            > support_threshold
                        ):
                            support += 1
            if not maximum or support < 3:
                continue
            if coarse:
                mean, _, _ = _local_mean(luminance, None, y, x, 1)
                # Rolling double sums can round differently at a float32 tie.
                if abs(float(mean) - 0.6) <= 0.000001:
                    need_coarse = True
                elif mean < np.float32(0.6):
                    continue
            # A single qualifying sample proves the 129x129 minimum test.
            # Bounded probes avoid quadratic work on broad flat plateaus;
            # unresolved cases get a complete sparse search below.
            contrast: bool = False
            for dy in (-64, 0, 64):
                sample_y = min(max(y + dy, 0), height - 1)
                for dx in (-64, 0, 64):
                    sample_x = min(max(x + dx, 0), width - 1)
                    if allowed is None or allowed[sample_y, sample_x]:
                        if np.float32(value - luminance[sample_y, sample_x]) >= np.float32(0.13):
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
def _resolve_floor(
    selected: BoolArray, luminance: FloatArray, allowed: BoolArray | None, unresolved: IntArray
) -> None:
    """Evaluate the full minimum predicate only at unproven candidates."""
    height, width = luminance.shape
    for index in range(len(unresolved)):
        y, x = unresolved[index]
        value: np.float32 = luminance[y, x]
        contrast: bool = False
        for sample_y in range(max(0, y - 64), min(height, y + 65)):
            for sample_x in range(max(0, x - 64), min(width, x + 65)):
                if allowed is None or allowed[sample_y, sample_x]:
                    if np.float32(value - luminance[sample_y, sample_x]) >= np.float32(0.13):
                        contrast = True
                        break
            if contrast:
                break
        selected[y, x] = contrast


@njit(cache=True, nogil=True)
def _contrast(
    luminance: FloatArray, allowed: BoolArray | None, centers: IntArray
) -> tuple[BoolArray, BoolArray]:
    """Screen compact-source contrast and mark decisions needing exact rolling filters."""
    keep: BoolArray = np.ones(len(centers), np.bool_)
    uncertain: BoolArray = np.zeros(len(centers), np.bool_)
    for index in range(len(centers)):
        y, x = centers[index]
        value: np.float32 = luminance[y, x]
        if value >= np.float32(0.75):
            continue
        square: np.float32
        mean: np.float32
        mean, square, count = _local_mean(luminance, allowed, y, x, 3)
        if allowed is not None:
            if count > 0:
                mean = np.float32(mean / count)
                square = np.float32(square / count)
            else:
                mean = square = np.float32(0)
        variance: np.float32 = np.float32(square - np.float32(mean * mean))
        stddev: np.float32 = np.float32(np.sqrt(max(variance, np.float32(0))))
        score: np.float32 = np.float32(np.float32(value - mean) + stddev)
        keep[index] = score >= np.float32(0.08)
        # This generous bound includes amplified rounding in masked variance
        # near zero. Only boundary decisions need the full rolling filters.
        uncertainty: float = 0.002 * (1 + abs(float(mean)) + abs(float(square)))
        uncertain[index] = not np.isfinite(score) or abs(float(score) - 0.08) <= uncertainty
    return keep, uncertain


def _component_centers(
    selected: NDArray[np.bool_], image: Image, core: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    """Find component bounding-box centers after retaining the complete halo topology."""
    labels: IndexArray
    labels, _ = ndimage.label(selected)
    centers: list[tuple[int, int]] = []
    for region in ndimage.find_objects(labels):
        if region is not None:
            y: int = (region[0].start + region[0].stop - 1) // 2
            x: int = (region[1].start + region[1].stop - 1) // 2
            if core[0] <= x + image.origin_x < core[2] and core[1] <= y + image.origin_y < core[3]:
                centers.append((y, x))
    return centers


def peak_positions(
    image: Image, threshold: float, core: Bounds, coarse: bool = False
) -> list[tuple[int, int]]:
    """Return global component bounding-box centers, retaining halo topology."""
    luminance, allowed = image.luminance, image.allowed
    selected, unresolved, unresolved_count, need_coarse = _select(
        image.rgb,
        luminance,
        allowed,
        np.float32(threshold),
        np.float32(threshold * 0.05),
        coarse,
        np.float32(luminance.min()),
    )
    # Dense plateaus use the linear-time filter rather than one window per
    # pixel. Sparse cases avoid allocating and filtering an entire tile.
    if 0 < unresolved_count <= min(SPARSE_FLOOR_LIMIT, max(8, luminance.size // 64)):
        _resolve_floor(selected, luminance, allowed, unresolved[:unresolved_count])
    elif unresolved_count:
        floor: FloatArray = ndimage.minimum_filter(
            luminance if allowed is None else np.where(allowed, luminance, np.inf),
            size=129,
            mode="nearest",
        )
        selected &= luminance - floor >= np.float32(0.13)
    if need_coarse:
        selected &= ndimage.uniform_filter(luminance, size=3, mode="nearest") >= np.float32(0.6)
    center_list: list[tuple[int, int]] = _component_centers(selected, image, core)
    if not center_list:
        return []
    centers: IntArray = np.asarray(center_list, dtype=np.int64)
    keep: BoolArray = np.ones(len(centers), np.bool_)
    if not coarse:
        keep, uncertain = _contrast(luminance, allowed, centers)
        if uncertain.any():
            square: FloatArray
            mean: FloatArray
            mean, square, count = uniform7_at(luminance, allowed, centers[uncertain]).T
            if allowed is not None:
                mean = np.divide(mean, count, out=np.zeros_like(mean), where=count > 0)
                square = np.divide(square, count, out=np.zeros_like(square), where=count > 0)
            y, x = centers[uncertain].T
            stddev: FloatArray = np.sqrt(np.maximum(square - mean**2, 0))
            keep[uncertain] = (luminance[y, x] >= 0.75) | (luminance[y, x] - mean + stddev >= 0.08)
    return [
        (int(x) + image.origin_x, int(y) + image.origin_y)
        for (y, x), selected in zip(centers, keep)
        if selected
    ]
