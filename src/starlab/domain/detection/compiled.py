"""Compiled implementation of candidate.py; reference arithmetic and predicates.

No fastmath: float32 rounding and the native eight-lane variance sum matter.
Image-dependent results are never persisted. Numba caches compiled machine code.
"""

import math
from collections.abc import Callable, MutableSequence, Sequence
from contextlib import nullcontext
from threading import Event, Lock

import numpy as np
from numba import (  # type: ignore[attr-defined]  # Public threading_layer is absent from Numba stubs.
    get_num_threads,
    prange,
    set_num_threads,
    threading_layer,
    types,
)
from numba.extending import overload, register_jitable
from numba.typed import Dict, List
from numpy.typing import NDArray

from ..array_types import (
    RGBA,
    BoolArray,
    CandidateRecord,
    DoubleArray,
    FloatArray,
    IntArray,
    Position,
)
from ..jit import njit
from . import numeric, rays
from .candidate import Image
from .records import Buffer, CandidateRow, CompiledStats, IndexArray, SortableAttempts

Stats = CompiledStats
# These external runtime descriptors mirror the checked Python tuple/NamedTuple layouts.
KEY = types.UniTuple(types.int64, 4)  # type: ignore[no-untyped-call]
STAT_TYPE = types.NamedUniTuple(types.float64, 5, Stats)  # type: ignore[no-untyped-call]
COLOR_TYPE = types.UniTuple(types.float64, 3)  # type: ignore[no-untyped-call]
ROW_TYPE = types.UniTuple(types.float64, 8)  # type: ignore[no-untyped-call]
ATTEMPT_TYPE = types.UniTuple(types.float64, 10)  # type: ignore[no-untyped-call]
PARALLEL_MIN_PEAKS = 256
MAX_CANDIDATE_COORDS = 262144
_WORKQUEUE_LOCK = Lock()


# Numba compares overload/implementation Parameters, including annotations.
# Both value parameters must stay unannotated despite their different runtime roles.
@overload(rays.f)
def _rounded(value) -> Callable[[float], np.float64]:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Provide the float32-rounding overload for a Numba scalar type."""

    def impl(value) -> np.float64:  # type: ignore[no-untyped-def]  # noqa: ANN001
        """Round a compiled scalar to float32 and widen its storage to float64."""
        return np.float64(np.float32(value))

    return impl


for _name in (
    "add",
    "sub",
    "mul",
    "div",
    "total",
    "luma",
    "inside",
    "neighbor_mean",
    "_recenter",
    "_repair_extents",
    "repair_recenter",
):
    register_jitable(getattr(rays, _name))
for _name in (
    "quarter_position",
    "quarter_size",
    "small_size",
    "preliminary_shape",
    "final_shape",
    "large_size_step",
    "refinement_result",
):
    register_jitable(getattr(numeric, _name))
f = rays.f


def make_image(image: Image) -> Buffer:
    """Build contiguous decoded RGB/luma/mask arrays and empty per-tile typed caches."""
    allowed = (
        np.empty((0, 0), np.bool_) if image.allowed is None else np.ascontiguousarray(image.allowed)
    )
    return Buffer(
        np.ascontiguousarray(image.rgb),
        np.ascontiguousarray(image.lum),
        allowed,
        image.ox,
        image.oy,
        Dict.empty(KEY, STAT_TYPE),
        Dict.empty(KEY, COLOR_TYPE),
    )


def scan_tile(
    image: Image,
    peaks: Sequence[tuple[int, int]],
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: Sequence[CandidateRecord],
    visited: BoolArray,
    cancel: Event | None,
) -> tuple[list[CandidateRecord], int]:
    """Qualify global pixel-index peaks, optionally preparing candidates in parallel blocks."""
    from ..imaging import check_cancel

    im = make_image(image)
    old = np.array([(r["x"], r["y"], r["size"]) for r in excluded], np.float64).reshape(-1, 3)
    rows: MutableSequence[CandidateRow] = List.empty_list(ROW_TYPE)
    maximum, seeds = 0.0, 0
    positions = np.asarray(peaks, np.int64).reshape(-1, 2)
    if len(positions) >= PARALLEL_MIN_PEAKS:
        indices, coords = candidate_indices(
            im, positions, threshold, neighbor_radius, visited.shape, MAX_CANDIDATE_COORDS
        )
        set_num_threads(min(4, get_num_threads()))
        records = np.empty((len(coords), 8))
        for start in range(0, len(coords), 8192):
            check_cancel(cancel)
            # OpenMP/TBB support concurrent callers; workqueue needs serialization.
            guard = _WORKQUEUE_LOCK if threading_layer() == "workqueue" else nullcontext()
            with guard:
                records[start : start + 8192] = evaluate_candidates(
                    im, coords[start : start + 8192], coarse, old
                )
    else:
        indices = np.empty((0, 0), np.int32)
        records = np.empty((0, 8))
    for start in range(0, len(positions), 128):
        check_cancel(cancel)
        maximum, count = scan_block(
            im,
            positions[start : start + 128],
            threshold,
            neighbor_radius,
            coarse,
            old,
            visited,
            rows,
            maximum,
            indices,
            records,
        )
        seeds += count
    result: list[CandidateRecord] = [
        dict(x=r[0], y=r[1], size=r[2], brightness=r[3], rgb=list(r[4:7]), count=int(r[7]))
        for r in rows
    ]
    return result, seeds


@njit(cache=True, nogil=True)
def candidate_indices(
    im: Buffer,
    positions: IntArray,
    threshold: float,
    radius: int,
    shape: tuple[int, int],
    budget: int,
) -> tuple[IndexArray, IntArray]:
    """Index above-threshold neighboring pixel seeds within the bounded eager budget."""
    indices = np.full(im.lum.shape, -1, np.int32)
    radius = max(1, radius)
    for px, py in positions:
        for y in range(max(0, py - radius), min(shape[0], py + radius + 1)):
            for x in range(max(0, px - radius), min(shape[1], px + radius + 1)):
                ix, iy = x - im.ox, y - im.oy
                if im.rgb[iy, ix].max() >= threshold:
                    indices[iy, ix] = 0
    count = np.count_nonzero(indices >= 0)
    if count > budget:
        # Scheduling fallback only: avoid retaining unbounded eager records.
        return np.empty((0, 0), np.int32), np.empty((0, 2), np.int64)
    coords = np.empty((count, 2), np.int64)
    offset = 0
    for y in range(indices.shape[0]):
        for x in range(indices.shape[1]):
            if indices[y, x] >= 0:
                indices[y, x] = offset
                coords[offset, 0], coords[offset, 1] = x + im.ox, y + im.oy
                offset += 1
    return indices, coords


@njit(cache=True, nogil=True, parallel=True)
def evaluate_candidates(
    im: Buffer, coords: IntArray, coarse: bool, excluded: DoubleArray
) -> DoubleArray:
    """Evaluate global integer seeds into N×8 rows; rejected candidates retain NaN X."""
    records = np.full((len(coords), 8), np.nan)
    for block in prange((len(coords) + 255) // 256):
        local = Buffer(
            im.rgb,
            im.lum,
            im.allowed,
            im.ox,
            im.oy,
            Dict.empty(KEY, STAT_TYPE),
            Dict.empty(KEY, COLOR_TYPE),
        )
        for i in range(block * 256, min((block + 1) * 256, len(coords))):
            result = qualify(local, coords[i, 0] + 0.5, coords[i, 1] + 0.5, coarse, excluded)
            if result is not None:
                x, y, size, brightness, r, g, b, count = result
                records[i, 0], records[i, 1], records[i, 2], records[i, 3] = x, y, size, brightness
                records[i, 4], records[i, 5], records[i, 6], records[i, 7] = r, g, b, count
    return records


@njit(cache=True)
def prepared_candidate(
    im: Buffer,
    x: int,
    y: int,
    coarse: bool,
    excluded: DoubleArray,
    indices: IndexArray,
    records: DoubleArray,
) -> CandidateRow | None:
    """Read a prepared global seed result, or qualify it when eager preparation fell back."""
    if indices.size == 0:
        return qualify(im, x + 0.5, y + 0.5, coarse, excluded)
    index = indices[y - im.oy, x - im.ox]
    if index < 0 or np.isnan(records[index, 0]):
        return None
    row = records[index]
    return (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])


@njit(cache=True)
def merge(rows: MutableSequence[CandidateRow], new: CandidateRow, maximum: float) -> float:
    """Merge a fresh seed with the first backward overlap and retain sorted insertion."""
    threshold = f(new[1] - f(max(new[2], maximum) + 1))
    for i in range(len(rows) - 1, -1, -1):
        old = rows[i]
        if old[1] < threshold:
            break
        dx, dy = f(old[0] - new[0]), f(old[1] - new[1])
        if f(f(dx * dx) + f(dy * dy)) < max(f(old[2] ** 2), f(new[2] ** 2)):
            count = min(max(old[7], 1.0) + 1, 255.0)
            weight = f(1 / count)
            remain = f(1 - weight)
            row = tuple8(old, new, remain, weight, count)
            rows[i] = row
            return max(maximum, row[2])
    index = len(rows)
    while index and new[1] <= rows[index - 1][1]:
        index -= 1
    rows.insert(index, new)
    return max(maximum, new[2])


@njit(cache=True)
def tuple8(
    a: CandidateRow, b: CandidateRow, remain: float, weight: float, count: float
) -> CandidateRow:
    """Blend seven measured fields with float32 rounding and store the updated count."""
    values = np.empty(7)
    for i in range(7):
        values[i] = f(f(a[i] * remain) + f(b[i] * weight))
    return (values[0], values[1], values[2], values[3], values[4], values[5], values[6], count)


@njit(cache=True, nogil=True)
def _neighbor_attempts(
    im: Buffer,
    px: int,
    py: int,
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: DoubleArray,
    visited: BoolArray,
    indices: IndexArray,
    records: DoubleArray,
    attempts: SortableAttempts,
) -> int:
    """Append accepted neighborhood seeds in row order and return all evaluated seeds."""
    seeds = 0
    height, width = visited.shape
    for y in range(max(0, py - neighbor_radius), min(height, py + neighbor_radius + 1)):
        for x in range(max(0, px - neighbor_radius), min(width, px + neighbor_radius + 1)):
            if visited[y, x] or covered(x + 0.5, y + 0.5, excluded):
                continue
            if im.rgb[y - im.oy, x - im.ox].max() < threshold:
                continue
            visited[y, x] = True
            seeds += 1
            result = prepared_candidate(im, x, y, coarse, excluded, indices, records)
            if result is not None:
                sx, sy, ss, sb, sr, sg, sc, sn = result
                attempts.append((float(y), float(x), sx, sy, ss, sb, sr, sg, sc, sn))
    return seeds


@njit(cache=True, nogil=True)
def scan_block(
    im: Buffer,
    peaks: IntArray,
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: DoubleArray,
    visited: BoolArray,
    rows: MutableSequence[CandidateRow],
    maximum: float,
    indices: IndexArray,
    records: DoubleArray,
) -> tuple[float, int]:
    """Process at most one cancellation block of peaks using native attempt/merge order."""
    seeds = 0
    height, width = visited.shape
    directions = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
    for p in range(len(peaks)):
        px, py = peaks[p, 0], peaks[p, 1]
        if covered(px + 0.5, py + 0.5, excluded):
            continue
        attempts: SortableAttempts = List.empty_list(ATTEMPT_TYPE)
        accepted = False
        for direction in directions:
            x, y = px + direction[0], py + direction[1]
            if not (0 <= x < width and 0 <= y < height) or visited[y, x]:
                continue
            if im.rgb[y - im.oy, x - im.ox].max() < threshold:
                continue
            visited[y, x] = True
            seeds += 1
            result = prepared_candidate(im, x, y, coarse, excluded, indices, records)
            if result is not None:
                sx, sy, ss, sb, sr, sg, sc, sn = result
                attempts.append((float(y), float(x), sx, sy, ss, sb, sr, sg, sc, sn))
                accepted = True
                break
        if accepted:
            seeds += _neighbor_attempts(
                im,
                px,
                py,
                threshold,
                neighbor_radius,
                coarse,
                excluded,
                visited,
                indices,
                records,
                attempts,
            )
        attempts.sort()
        for row in attempts:
            maximum = merge(
                rows, (row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]), maximum
            )
    return maximum, seeds


@njit(cache=True)
def inside(im: Buffer, x: float, y: float) -> bool:
    """Test global coordinates against the compiled tile bounds."""
    width: int = im.rgb.shape[1]
    height: int = im.rgb.shape[0]
    return im.ox <= x < im.ox + width and im.oy <= y < im.oy + height


@njit(cache=True)
def sample(im: Buffer, x: float, y: float) -> RGBA:
    """Read a native-truncated RGB sample; invalid or excluded samples have zero alpha."""
    ix, iy = int(x) - im.ox, int(y) - im.oy
    if not (0 <= ix < im.rgb.shape[1] and 0 <= iy < im.rgb.shape[0]):
        return (0.0, 0.0, 0.0, 0.0)
    if im.allowed.size and not im.allowed[iy, ix]:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(im.rgb[iy, ix, 0]), float(im.rgb[iy, ix, 1]), float(im.rgb[iy, ix, 2]), 1.0)


@njit(cache=True)
def support(
    im: Buffer, x: float, y: float, index: int, ring: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Enumerate allowed disk/ring weights, luminance, and RGB in native square order."""
    ix, iy = math.floor(x), math.floor(y)
    cx, cy = (x - ix) - 0.5, (y - iy) - 0.5
    radius = index * 0.25
    extent = int(f(f(radius + cx) + f(1.65))) if ring else int(f(f(radius + f(0.6)) + cx))
    lower, upper = (-extent, extent + 1) if ring else (-extent - 1, extent + 2)
    capacity = (upper - lower) ** 2
    weights = np.empty(capacity, np.float32)
    values = np.empty(capacity, np.float32)
    colors = np.empty((capacity, 3), np.float32)
    n = 0
    for dy in range(lower, upper):
        yy = dy + iy - im.oy
        if not 0 <= yy < im.rgb.shape[0]:
            continue
        for dx in range(lower, upper):
            xx = dx + ix - im.ox
            if not 0 <= xx < im.rgb.shape[1] or (im.allowed.size and not im.allowed[yy, xx]):
                continue
            ax, ay = np.float32(dx - cx), np.float32(dy - cy)
            distance = np.float32(math.sqrt(np.float32(np.float32(ax * ax) + np.float32(ay * ay))))
            # NumPy stubs lose the concrete mixed float/float32 min/max result.
            # Keep the recovered rounding expression rather than changing its arithmetic.
            if ring:
                weight = np.float32(
                    max(  # type: ignore[arg-type]
                        0.0,
                        np.float32(
                            np.float32(1.15) - abs(np.float32(np.float32(radius) - distance))
                        )
                        * np.float32(0.8695652484893799),
                    )
                )
            else:
                weight = np.float32(
                    min(1.0, max(0.0, np.float32(np.float32(radius + f(0.6)) - distance)))  # type: ignore[arg-type]
                )
            if weight > 0:
                weights[n], values[n] = weight, im.lum[yy, xx]
                colors[n] = im.rgb[yy, xx]
                n += 1
    return weights[:n], values[:n], colors[:n]


@njit(cache=True)
def lane_sum(values: FloatArray) -> float:
    """Reduce eight float32 accumulation lanes in the recovered native combination order."""
    count = len(values) // 8 * 8
    lanes = np.zeros(8, np.float32)
    for i in range(count):
        lane = i % 8
        lanes[lane] = np.float32(lanes[lane] + values[i])
    if count:
        combined = lanes[:4] + lanes[4:]
        result = f(
            f(float(combined[2]) + float(combined[0])) + f(float(combined[3]) + float(combined[1]))
        )
    else:
        result = 0.0
    for i in range(count, len(values)):
        result = f(result + float(values[i]))
    return result


@njit(cache=True)
def stats(im: Buffer, x: float, y: float, index: int, ring: bool = False) -> CompiledStats | None:
    """Return quarter-position weighted statistics; indices outside [0,1056) are invalid."""
    if not 0 <= index < 1056:
        return None
    key = (int(math.floor(x * 4)), int(math.floor(y * 4)), index, int(ring))
    if key in im.stats:
        return im.stats[key]
    weights, values, _ = support(im, key[0] * 0.25, key[1] * 0.25, index, ring)
    total, weighted = np.float32(0), np.float32(0)
    minimum, maximum = 1.0, 0.0
    for i in range(len(weights)):
        total = np.float32(total + weights[i])
        weighted = np.float32(weighted + np.float32(values[i] * weights[i]))
        minimum, maximum = min(minimum, float(values[i])), max(maximum, float(values[i]))
    if total == 0:
        result = Stats(1.0, 0.0, 0.0, 0.0, 0.0)
    else:
        mean = f(float(weighted) / float(total))
        variance = np.empty(len(weights), np.float32)
        for i in range(len(weights)):
            delta = np.float32(values[i] - np.float32(mean))
            variance[i] = np.float32(np.float32(delta * delta) * weights[i])
        std = f(math.sqrt(f(lane_sum(variance) / lane_sum(weights))))
        result = Stats(minimum, mean, maximum, std, f(mean - std))
    if len(im.stats) >= 8192:
        im.stats.clear()
    im.stats[key] = result
    return result


@njit(cache=True)
def color(im: Buffer, x: float, y: float, index: int, ring: bool = False) -> DoubleArray:
    """Return three weighted float32 RGB channels stored as float64, or zeroes."""
    if not 0 <= index < 1056:
        return np.zeros(3)
    key = (int(math.floor(x * 4)), int(math.floor(y * 4)), index, int(ring))
    if key in im.colors:
        return np.array(im.colors[key])
    weights, _, values = support(im, key[0] * 0.25, key[1] * 0.25, index, ring)
    total = np.float32(0)
    sums = np.zeros(3, np.float32)
    for i in range(len(weights)):
        total = np.float32(total + weights[i])
        for channel in range(3):
            sums[channel] = np.float32(sums[channel] + np.float32(values[i, channel] * weights[i]))
    result = np.zeros(3)
    if total > 0:
        for channel in range(3):
            result[channel] = np.float64(np.float32(sums[channel] / total))
    if len(im.colors) >= 4096:
        im.colors.clear()
    im.colors[key] = (result[0], result[1], result[2])
    return result


@njit(cache=True)
def _extend_ray(
    im: Buffer,
    x: float,
    y: float,
    dx: int,
    dy: int,
    center: tuple[float, float, float, float],
    lum: NDArray[np.float64],
    distance: NDArray[np.float64],
    available: int,
    needed: int,
) -> int:
    """Fill ray samples through needed, stopping at bounds; return the filled count."""
    while available <= needed:
        px, py = rays.add(x, available * dx), rays.add(y, available * dy)
        if not inside(im, px, py):
            break
        value = sample(im, px, py)
        lum[available] = rays.luma(value)
        a, b, c = (
            rays.sub(center[0], value[0]),
            rays.sub(center[1], value[1]),
            rays.sub(center[2], value[2]),
        )
        distance[available] = f(
            math.sqrt(rays.add(rays.add(rays.mul(b, b), rays.mul(a, a)), rays.mul(c, c)))
        )
        available += 1
    return available


@njit(cache=True)
def initial(im: Buffer, x: float, y: float) -> tuple[bool, float, float, float]:
    """Trace native ray extents, repair outliers, and validate the recentered size estimate."""
    center = sample(im, x, y)
    center_luma = rays.luma(center)
    limit = 32 if center_luma < f(0.75) else 266
    radii = np.full(8, -1.0)
    for direction in range(8):
        dx, dy = rays.DIRECTIONS[direction]
        length = f(math.sqrt(2)) if direction % 2 else 1.0
        lum, distance = np.zeros(266), np.zeros(266)
        lum[0] = center_luma
        available, rejected = 1, False
        for k in range(1, limit):
            width = int(rays.add(rays.mul(k, 0.25), 1.0))
            start = k - width + 1
            available = _extend_ray(im, x, y, dx, dy, center, lum, distance, available, k)
            if available <= k:
                radii[direction] = rays.mul(k, length)
                rejected = True
                break
            prev = rays.total(lum[start - width : start])
            current = rays.total(lum[start : k + 1])
            prev_color = rays.mul(rays.total(distance[start - width : start]), 0.125)
            current_color = rays.mul(rays.total(distance[start : k + 1]), 0.125)
            delta = rays.sub(rays.sub(prev, prev_color), rays.sub(current, current_color))
            if delta > rays.mul(width, f(0.04)) or -delta > rays.mul(width, f(0.015)):
                end = min(k + width, limit - 1)
                available = _extend_ray(im, x, y, dx, dy, center, lum, distance, available, end)
                if available <= end:
                    radii[direction] = rays.mul(k, length)
                    rejected = True
                    break
                next_sum = rays.total(lum[k + 1 : end + 1])
                next_color = rays.mul(rays.total(distance[k + 1 : end + 1]), 0.125)
                next_delta = rays.sub(
                    rays.sub(current, current_color), rays.sub(next_sum, next_color)
                )
                if delta > rays.mul(width, f(0.04)) and rays.sub(delta, next_delta) > rays.mul(
                    width, f(0.01)
                ):
                    background = min(next_sum, current)
                    ratio = rays.div(rays.sub(current, background), rays.sub(prev, background))
                    radii[direction] = rays.mul(
                        rays.add(rays.mul(ratio, width), rays.sub(start, 0.625)), length
                    )
                    rejected = True
                    break
                if -delta > rays.mul(width, f(0.015)) and -next_delta > rays.mul(width, f(0.015)):
                    return False, x, y, 0.0
        if not rejected and direction == 0:
            return False, x, y, 0.0
    ok, position, repaired = repair((x, y), radii)
    xx, yy = position
    if not ok or abs(rays.sub(xx, x)) > 2 or abs(rays.sub(yy, y)) > 2 or not inside(im, xx, yy):
        return False, x, y, 0.0
    positives = [r for r in repaired if r > 0]
    size = rays.div(rays.total(positives), len(positives)) if len(positives) else 0.0
    return 0 < size <= 128, xx, yy, size


@njit(cache=True)
def luma_color(value: DoubleArray) -> float:
    """Combine weighted color channels using the shrink-stage operation order."""
    luminance: float = (
        value[0] * numeric.LUMA[0] + value[1] * numeric.LUMA[1] + value[2] * numeric.LUMA[2]
    )
    return luminance


@njit(cache=True, error_model="numpy")
def shrink(im: Buffer, x: float, y: float, size: float) -> float:
    """Shrink the initial radius using native disk/ring luminance thresholds."""
    index = int(size * 1.32)
    if index > 8:
        a, b = (
            luma_color(color(im, x, y, index, True)),
            luma_color(color(im, x, y, int(0.4 * size))),
        )
        ratio = a / b if b else (math.nan if not a else math.inf)
        threshold, lower = 0.475 * (a + b) * ratio, int(2 * size + 0.125)
    elif size >= 2:
        threshold, lower = 0.9 * luma_color(color(im, x, y, 4)), 6
    else:
        return size
    index = int(size * 4 + 0.125)
    while index > lower and luma_color(color(im, x, y, index, True)) < threshold:
        index -= 1
    return 0.25 * (index + 2)


@njit(cache=True)
def refine_position(im: Buffer, x: float, y: float, size: float) -> tuple[float, float, bool]:
    """Hill-climb in native direction/tie order and report whether the center moved."""
    x, y = f(x), f(y)
    index = max(1, int(size * 4))
    initial_stats = stats(im, x, y, index)
    if initial_stats is None:
        return x, y, False
    best, step, changed = initial_stats.score, max(f(size * 0.125), 0.25), False
    q = 0.7071067690849304
    directions = (
        (1.0, 0.0),
        (q, q),
        (0.0, 1.0),
        (-q, q),
        (-1.0, 0.0),
        (-q, -q),
        (0.0, -1.0),
        (q, -q),
    )
    for _ in range(6):
        selected = -1
        for i in range(8):
            dx, dy = directions[i]
            value = stats(im, f(x + f(dx * step)), f(y + f(dy * step)), index)
            if value is not None and value.score > best + 1e-6:
                best, selected = value.score, i
        if selected < 0:
            break
        dx, dy = directions[selected]
        x, y = f(x + f(dx * step)), f(y + f(dy * step))
        changed, step = True, max(f(step * f(0.8)), 0.25)
    return x, y, changed


@njit(cache=True)
def refine_size(im: Buffer, x: float, y: float, size: float) -> tuple[float, bool]:
    """Refine source radius in native iteration order and report a size change."""
    original, direction = size, 0
    for iteration in range(2, 8):
        old, step = size, max(size * 0.05 / iteration, 0.25)
        if size < 1.25:
            # Only indices outside [0,1056) yield None; fixed index10 is always valid.
            peak, background = color(im, x, y, 5).max(), stats(im, x, y, 10, True).minimum  # type: ignore[union-attr]
            if peak > background:
                size = numeric.small_size((peak - background) / (1 - background))
                direction = 1 if size > old else -1 if size < old else direction
            else:
                size, direction = size + 0.25, 1
        else:
            a, b, c = (
                stats(im, x, y, int(8 * size) + 4, True),
                stats(im, x, y, int(2.4 * size)),
                stats(im, x, y, int(4 * size) - 2, True),
            )
            size, direction, stop = numeric.large_size_step(size, step, direction, a, b, c)
            if stop:
                break
        if size == old:
            break
    return size, size != original


@njit(cache=True)
def covered(x: float, y: float, excluded: DoubleArray) -> bool:
    """Test a global point against N×3 float64 exclusion disks [x,y,size]."""
    for i in range(len(excluded)):
        if (x - excluded[i, 0]) ** 2 + (y - excluded[i, 1]) ** 2 < excluded[i, 2] ** 2:
            return True
    return False


@njit(cache=True)
def _refine_candidate(
    im: Buffer,
    x: float,
    y: float,
    size: float,
    coarse: bool,
    excluded: NDArray[np.float64],
) -> tuple[float, float, float] | None:
    """Alternate native refinement stages, including rollback and convergence limits."""
    old_x, old_y = x, y
    x, y, moved = refine_position(im, x, y, size)
    size, resized = refine_size(im, x, y, size)
    limit, maximum = int(size + 8), 1.5 * size + 2
    if not coarse:
        maximum = min(maximum, 20.0)
    converged = False
    for _ in range(limit):
        if not moved and not resized:
            converged = True
            break
        if not inside(im, x, y):
            x, y, converged = old_x, old_y, True
            break
        old_x, old_y = x, y
        x, y, moved = refine_position(im, x, y, size)
        if moved or resized:
            size, resized = refine_size(im, x, y, size)
        if size > maximum or covered(x, y, excluded):
            return None
    return numeric.refinement_result(x, y, size, converged)


@njit(cache=True)
def _accept_shape(im: Buffer, x: float, y: float, size: float) -> bool:
    """Evaluate the final shape supports in their original sampling order."""
    n = max(1, int(4 * size))
    if n < 12:
        a, b = stats(im, x, y, max(1, n - 2)), stats(im, x, y, 2 * n + 4, True)
        if a is None or b is None or not (a.score >= b.score and a.score - b.score >= 0.13):
            return False
    else:
        a, b, c, d, e = (
            stats(im, x, y, n // 2),
            stats(im, x, y, n - 3),
            stats(im, x, y, n - 3, True),
            stats(im, x, y, (6 * n) // 5 + 4, True),
            stats(im, x, y, 2 * n + 4, True),
        )
        if (
            a is None
            or b is None
            or c is None
            or d is None
            or e is None
            or not numeric.final_shape(size, a, b, c, d, e)
        ):
            return False
    return True


@njit(cache=True)
def qualify(
    im: Buffer, x: float, y: float, coarse: bool, excluded: DoubleArray
) -> CandidateRow | None:
    """Qualify a half-pixel seed and return its eight-field raw source row on success."""
    ok, x, y, size = initial(im, x, y)
    if not ok:
        return None
    size = shrink(im, x, y, size)
    n = int(4 * size)
    delta = n // 8 + 4
    if n >= delta:
        a, b = stats(im, x, y, n - delta, True), stats(im, x, y, n + delta, True)
        if a is None or b is None or not numeric.preliminary_shape(a, b, n):
            return None
    refined = _refine_candidate(im, x, y, size, coarse, excluded)
    if refined is None:
        return None
    x, y, size = refined
    x, y, size = (
        numeric.quarter_position(x),
        numeric.quarter_position(y),
        numeric.quarter_size(size),
    )
    if not _accept_shape(im, x, y, size):
        return None
    if not inside(im, x, y) or (coarse and size <= 4):
        return None
    count = 1 - int((size - 1) * -0.5)
    rgb = np.zeros(3)
    for i in range(count):
        rgb += color(im, x, y, int(4 * size) + 4 * i, True)
    rgb[1] = 0.4 * rgb[0] + 0.3 * (rgb[1] + rgb[2])
    rgb /= rgb.mean()
    source = color(im, x, y, int(4 * size) + 2)
    brightness = np.float64(
        np.float32(np.float32(np.float32(source[0] + source[1]) + source[2]) / np.float32(3))
    )
    return (x, y, size, brightness, rgb[0], rgb[1], rgb[2], 1.0)


@njit(cache=True)
def repair(position: Position, radii: DoubleArray) -> tuple[bool, Position, DoubleArray]:
    """3DA00. Repaired array is returned even on failure (native mutates it).

    Outlier marking uses the ORIGINAL adjacent values; repair uses the marked
    adjacent values, never newly repaired ones. Successful non-outliers retain
    original values. Native diagonal recenter factor really is sqrt(2).
    """
    radii = radii.copy()
    position = (f(position[0]), f(position[1]))
    if not rays._repair_extents(radii):
        return False, position, radii
    return True, rays._recenter(position, radii), radii
