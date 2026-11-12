"""Compiled implementation of candidate.py; reference arithmetic and predicates.

No fastmath: float32 rounding and the native eight-lane variance sum matter.
Image-dependent results are never persisted. Numba caches compiled machine code.
"""

import math
from _thread import LockType
from collections.abc import MutableSequence, Sequence
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
from numba.extending import register_jitable
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
from .buffers import anonymous_array
from .candidate import Image
from .constants import PREFERRED_SEED_OFFSETS
from .precision import round_to_float32
from .records import (
    CacheKey,
    CandidateRow,
    CompiledStats,
    DetectionTile,
    IndexArray,
    SortableAttempts,
)

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


for _name in (
    "add_float32",
    "subtract_float32",
    "multiply_float32",
    "divide_float32",
    "sum_float32",
    "luma",
    "inside",
    "neighbor_mean",
    "recenter",
    "repair_extents",
    "repair_recenter",
):
    register_jitable(getattr(rays, _name))
for _name in (
    "quarter_position",
    "quarter_size",
    "small_size",
    "small_size_step",
    "size_refinement_done",
    "combine_variance_lanes",
    "preliminary_shape",
    "final_shape",
    "large_size_step",
    "refinement_result",
):
    register_jitable(getattr(numeric, _name))


def make_image(image: Image) -> DetectionTile:
    """Build contiguous decoded RGB/luma/mask arrays and empty per-tile typed caches."""
    allowed: BoolArray = (
        np.empty((0, 0), np.bool_) if image.allowed is None else np.ascontiguousarray(image.allowed)
    )
    return DetectionTile(
        np.ascontiguousarray(image.rgb),
        np.ascontiguousarray(image.luminance),
        allowed,
        image.origin_x,
        image.origin_y,
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

    tile: DetectionTile = make_image(image)
    excluded_sources: DoubleArray = np.array(
        [(record["x"], record["y"], record["size"]) for record in excluded], np.float64
    ).reshape(-1, 3)
    rows: MutableSequence[CandidateRow] = List.empty_list(ROW_TYPE)
    maximum, seeds = 0.0, 0
    positions: IntArray = np.asarray(peaks, np.int64).reshape(-1, 2)
    if len(positions) >= PARALLEL_MIN_PEAKS:
        indices: IndexArray
        indices, coordinates = candidate_indices(
            tile,
            anonymous_array(tile.luminance.shape, np.int32),
            positions,
            threshold,
            neighbor_radius,
            visited.shape,
            MAX_CANDIDATE_COORDS,
        )
        set_num_threads(min(4, get_num_threads()))
        records: DoubleArray = np.empty((len(coordinates), 8))
        for start in range(0, len(coordinates), 8192):
            check_cancel(cancel)
            # OpenMP/TBB support concurrent callers; workqueue needs serialization.
            guard: LockType | nullcontext[None] = (
                _WORKQUEUE_LOCK if threading_layer() == "workqueue" else nullcontext()
            )
            with guard:
                records[start : start + 8192] = evaluate_candidates(
                    tile, coordinates[start : start + 8192], coarse, excluded_sources
                )
    else:
        indices = np.empty((0, 0), np.int32)
        records = np.empty((0, 8))
    for start in range(0, len(positions), 128):
        check_cancel(cancel)
        maximum, count = scan_block(
            tile,
            positions[start : start + 128],
            threshold,
            neighbor_radius,
            coarse,
            excluded_sources,
            visited,
            rows,
            maximum,
            indices,
            records,
        )
        seeds += count
    result: list[CandidateRecord] = [
        CandidateRecord(
            x=record[0],
            y=record[1],
            size=record[2],
            brightness=record[3],
            rgb=list(record[4:7]),
            count=int(record[7]),
        )
        for record in rows
    ]
    return result, seeds


@njit(cache=True, nogil=True)
def candidate_indices(
    tile: DetectionTile,
    indices: IndexArray,
    positions: IntArray,
    threshold: float,
    radius: int,
    shape: tuple[int, int],
    budget: int,
) -> tuple[IndexArray, IntArray]:
    """Index above-threshold neighboring pixel seeds within the bounded eager budget."""
    indices[:, :] = -1
    radius = max(1, radius)
    for peak_x, peak_y in positions:
        for y in range(max(0, peak_y - radius), min(shape[0], peak_y + radius + 1)):
            for x in range(max(0, peak_x - radius), min(shape[1], peak_x + radius + 1)):
                pixel_x, pixel_y = x - tile.origin_x, y - tile.origin_y
                if tile.rgb[pixel_y, pixel_x].max() >= threshold:
                    indices[pixel_y, pixel_x] = 0
    candidate_count: np.intp = np.count_nonzero(indices >= 0)
    if candidate_count > budget:
        # Scheduling fallback only: avoid retaining unbounded eager records.
        return np.empty((0, 0), np.int32), np.empty((0, 2), np.int64)
    coordinates: IntArray = np.empty((candidate_count, 2), np.int64)
    offset: int = 0
    for y in range(indices.shape[0]):
        for x in range(indices.shape[1]):
            if indices[y, x] >= 0:
                indices[y, x] = offset
                coordinates[offset, 0], coordinates[offset, 1] = (
                    x + tile.origin_x,
                    y + tile.origin_y,
                )
                offset += 1
    return indices, coordinates


@njit(cache=True, nogil=True, parallel=True)
def evaluate_candidates(
    tile: DetectionTile, coordinates: IntArray, coarse: bool, excluded: DoubleArray
) -> DoubleArray:
    """Evaluate global integer seeds into N×8 rows; rejected candidates retain NaN X."""
    records: DoubleArray = np.full((len(coordinates), 8), np.nan)
    for block in prange((len(coordinates) + 255) // 256):
        local_tile: DetectionTile = DetectionTile(
            tile.rgb,
            tile.luminance,
            tile.allowed,
            tile.origin_x,
            tile.origin_y,
            Dict.empty(KEY, STAT_TYPE),
            Dict.empty(KEY, COLOR_TYPE),
        )
        for candidate_index in range(block * 256, min((block + 1) * 256, len(coordinates))):
            result: CandidateRow | None = qualify(
                local_tile,
                coordinates[candidate_index, 0] + 0.5,
                coordinates[candidate_index, 1] + 0.5,
                coarse,
                excluded,
            )
            if result is not None:
                x, y, size, brightness, red, green, blue, count = result
                (
                    records[candidate_index, 0],
                    records[candidate_index, 1],
                    records[candidate_index, 2],
                    records[candidate_index, 3],
                ) = x, y, size, brightness
                (
                    records[candidate_index, 4],
                    records[candidate_index, 5],
                    records[candidate_index, 6],
                    records[candidate_index, 7],
                ) = red, green, blue, count
    return records


@njit(cache=True)
def prepared_candidate(
    tile: DetectionTile,
    x: int,
    y: int,
    coarse: bool,
    excluded: DoubleArray,
    indices: IndexArray,
    records: DoubleArray,
) -> CandidateRow | None:
    """Read a prepared global seed result, or qualify it when eager preparation fell back."""
    if indices.size == 0:
        return qualify(tile, x + 0.5, y + 0.5, coarse, excluded)
    index: np.int32 = indices[y - tile.origin_y, x - tile.origin_x]
    if index < 0 or np.isnan(records[index, 0]):
        return None
    row: DoubleArray = records[index]
    return (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])


@njit(cache=True)
def merge(
    rows: MutableSequence[CandidateRow], incoming_row: CandidateRow, maximum_size: float
) -> float:
    """Merge a fresh seed with the first backward overlap and retain sorted insertion."""
    minimum_y: float = round_to_float32(
        incoming_row[1] - round_to_float32(max(incoming_row[2], maximum_size) + 1)
    )
    for row_index in range(len(rows) - 1, -1, -1):
        existing_row: CandidateRow = rows[row_index]
        if existing_row[1] < minimum_y:
            break
        dx, dy = (
            round_to_float32(existing_row[0] - incoming_row[0]),
            round_to_float32(existing_row[1] - incoming_row[1]),
        )
        if round_to_float32(round_to_float32(dx * dx) + round_to_float32(dy * dy)) < max(
            round_to_float32(existing_row[2] ** 2), round_to_float32(incoming_row[2] ** 2)
        ):
            observation_count: float = min(max(existing_row[7], 1.0) + 1, 255.0)
            incoming_weight: float = round_to_float32(1 / observation_count)
            existing_weight: float = round_to_float32(1 - incoming_weight)
            row: CandidateRow = blend_row(
                existing_row, incoming_row, existing_weight, incoming_weight, observation_count
            )
            rows[row_index] = row
            return max(maximum_size, row[2])
    insertion_index: int = len(rows)
    while insertion_index and incoming_row[1] <= rows[insertion_index - 1][1]:
        insertion_index -= 1
    rows.insert(insertion_index, incoming_row)
    return max(maximum_size, incoming_row[2])


@njit(cache=True)
def blend_row(
    existing: CandidateRow,
    incoming: CandidateRow,
    existing_weight: float,
    incoming_weight: float,
    observation_count: float,
) -> CandidateRow:
    """Blend seven measured fields with float32 rounding and store the updated count."""
    blended_fields: DoubleArray = np.empty(7)
    for field_index in range(7):
        blended_fields[field_index] = round_to_float32(
            round_to_float32(existing[field_index] * existing_weight)
            + round_to_float32(incoming[field_index] * incoming_weight)
        )
    return (
        blended_fields[0],
        blended_fields[1],
        blended_fields[2],
        blended_fields[3],
        blended_fields[4],
        blended_fields[5],
        blended_fields[6],
        observation_count,
    )


@njit(cache=True, nogil=True)
def _neighbor_attempts(
    tile: DetectionTile,
    peak_x: int,
    peak_y: int,
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
    seeds: int = 0
    height, width = visited.shape
    for y in range(max(0, peak_y - neighbor_radius), min(height, peak_y + neighbor_radius + 1)):
        for x in range(max(0, peak_x - neighbor_radius), min(width, peak_x + neighbor_radius + 1)):
            if visited[y, x] or covered(x + 0.5, y + 0.5, excluded):
                continue
            if tile.rgb[y - tile.origin_y, x - tile.origin_x].max() < threshold:
                continue
            visited[y, x] = True
            seeds += 1
            result: CandidateRow | None = prepared_candidate(
                tile, x, y, coarse, excluded, indices, records
            )
            if result is not None:
                (
                    source_x,
                    source_y,
                    source_size,
                    source_brightness,
                    source_red,
                    source_green,
                    source_blue,
                    source_count,
                ) = result
                attempts.append(
                    (
                        float(y),
                        float(x),
                        source_x,
                        source_y,
                        source_size,
                        source_brightness,
                        source_red,
                        source_green,
                        source_blue,
                        source_count,
                    )
                )
    return seeds


@njit(cache=True, nogil=True)
def scan_block(
    tile: DetectionTile,
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
    seeds: int = 0
    height, width = visited.shape
    for peak_index in range(len(peaks)):
        peak_x, peak_y = peaks[peak_index, 0], peaks[peak_index, 1]
        if covered(peak_x + 0.5, peak_y + 0.5, excluded):
            continue
        attempts: SortableAttempts = List.empty_list(ATTEMPT_TYPE)
        accepted: bool = False
        for direction in PREFERRED_SEED_OFFSETS:
            x, y = peak_x + direction[0], peak_y + direction[1]
            if not (0 <= x < width and 0 <= y < height) or visited[y, x]:
                continue
            if tile.rgb[y - tile.origin_y, x - tile.origin_x].max() < threshold:
                continue
            visited[y, x] = True
            seeds += 1
            result: CandidateRow | None = prepared_candidate(
                tile, x, y, coarse, excluded, indices, records
            )
            if result is not None:
                (
                    source_x,
                    source_y,
                    source_size,
                    source_brightness,
                    source_red,
                    source_green,
                    source_blue,
                    source_count,
                ) = result
                attempts.append(
                    (
                        float(y),
                        float(x),
                        source_x,
                        source_y,
                        source_size,
                        source_brightness,
                        source_red,
                        source_green,
                        source_blue,
                        source_count,
                    )
                )
                accepted = True
                break
        if accepted:
            seeds += _neighbor_attempts(
                tile,
                peak_x,
                peak_y,
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
def inside(tile: DetectionTile, x: float, y: float) -> bool:
    """Test global coordinates against the compiled tile bounds."""
    width: int = tile.rgb.shape[1]
    height: int = tile.rgb.shape[0]
    return (
        tile.origin_x <= x < tile.origin_x + width and tile.origin_y <= y < tile.origin_y + height
    )


@njit(cache=True)
def sample(tile: DetectionTile, x: float, y: float) -> RGBA:
    """Read a native-truncated RGB sample; invalid or excluded samples have zero alpha."""
    pixel_x, pixel_y = int(x) - tile.origin_x, int(y) - tile.origin_y
    if not (0 <= pixel_x < tile.rgb.shape[1] and 0 <= pixel_y < tile.rgb.shape[0]):
        return 0.0, 0.0, 0.0, 0.0
    if tile.allowed.size and not tile.allowed[pixel_y, pixel_x]:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(tile.rgb[pixel_y, pixel_x, 0]),
        float(tile.rgb[pixel_y, pixel_x, 1]),
        float(tile.rgb[pixel_y, pixel_x, 2]),
        1.0,
    )


@njit(cache=True)
def support(
    tile: DetectionTile, x: float, y: float, index: int, ring: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Enumerate allowed disk/ring weights, luminance, and RGB in native square order."""
    pixel_x, pixel_y = math.floor(x), math.floor(y)
    phase_x, phase_y = (x - pixel_x) - 0.5, (y - pixel_y) - 0.5
    radius: float = index * 0.25
    extent: int = (
        int(round_to_float32(round_to_float32(radius + phase_x) + round_to_float32(1.65)))
        if ring
        else int(round_to_float32(round_to_float32(radius + round_to_float32(0.6)) + phase_x))
    )
    lower, upper = (-extent, extent + 1) if ring else (-extent - 1, extent + 2)
    capacity: int = (upper - lower) ** 2
    weights: FloatArray = np.empty(capacity, dtype=np.float32)
    luminance_samples: FloatArray = np.empty(capacity, dtype=np.float32)
    color_samples: FloatArray = np.empty((capacity, 3), dtype=np.float32)
    sample_count: int = 0
    for dy in range(lower, upper):
        sample_y: int = dy + pixel_y - tile.origin_y
        if not 0 <= sample_y < tile.rgb.shape[0]:
            continue
        for dx in range(lower, upper):
            sample_x: int = dx + pixel_x - tile.origin_x
            if not 0 <= sample_x < tile.rgb.shape[1] or (
                tile.allowed.size and not tile.allowed[sample_y, sample_x]
            ):
                continue
            offset_x, offset_y = np.float32(dx - phase_x), np.float32(dy - phase_y)
            distance: np.float32 = np.float32(
                math.sqrt(
                    np.float32(np.float32(offset_x * offset_x) + np.float32(offset_y * offset_y))
                )
            )
            # NumPy stubs lose the concrete mixed float/float32 min/max result.
            # Keep the recovered rounding expression rather than changing its arithmetic.
            if ring:
                weight: np.float32 = np.float32(
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
                    min(
                        1.0,
                        max(0.0, np.float32(np.float32(radius + round_to_float32(0.6)) - distance)),
                    )  # type: ignore[arg-type]
                )
            if weight > 0:
                weights[sample_count], luminance_samples[sample_count] = (
                    weight,
                    tile.luminance[sample_y, sample_x],
                )
                color_samples[sample_count] = tile.rgb[sample_y, sample_x]
                sample_count += 1
    return weights[:sample_count], luminance_samples[:sample_count], color_samples[:sample_count]


@njit(cache=True)
def lane_sum(values: FloatArray) -> float:
    """Reduce eight float32 accumulation lanes in the recovered native combination order."""
    lane_count: int = len(values) // 8 * 8
    lanes: FloatArray = np.zeros(8, np.float32)
    for value_index in range(lane_count):
        lane: int = value_index % 8
        lanes[lane] = np.float32(lanes[lane] + values[value_index])
    if lane_count:
        result: float = numeric.combine_variance_lanes(lanes)
    else:
        result = 0.0
    for value_index in range(lane_count, len(values)):
        result = round_to_float32(result + float(values[value_index]))
    return result


@njit(cache=True)
def stats(
    tile: DetectionTile, x: float, y: float, index: int, ring: bool = False
) -> CompiledStats | None:
    """Return quarter-position weighted statistics; indices outside [0,1056) are invalid."""
    if not 0 <= index < 1056:
        return None
    cache_key: CacheKey = (int(math.floor(x * 4)), int(math.floor(y * 4)), index, int(ring))
    if cache_key in tile.stats:
        return tile.stats[cache_key]
    weights, values, _ = support(tile, cache_key[0] * 0.25, cache_key[1] * 0.25, index, ring)
    weighted_luminance: np.float32
    total_weight: np.float32
    total_weight, weighted_luminance = np.float32(0), np.float32(0)
    minimum, maximum = 1.0, 0.0
    for sample_index in range(len(weights)):
        total_weight = np.float32(total_weight + weights[sample_index])
        weighted_luminance = np.float32(
            weighted_luminance + np.float32(values[sample_index] * weights[sample_index])
        )
        minimum, maximum = (
            min(minimum, float(values[sample_index])),
            max(maximum, float(values[sample_index])),
        )
    if total_weight == 0:
        result: CompiledStats = Stats(1.0, 0.0, 0.0, 0.0, 0.0)
    else:
        mean: float = round_to_float32(float(weighted_luminance) / float(total_weight))
        variance: FloatArray = np.empty(len(weights), np.float32)
        for sample_index in range(len(weights)):
            delta: np.float32 = np.float32(values[sample_index] - np.float32(mean))
            variance[sample_index] = np.float32(np.float32(delta * delta) * weights[sample_index])
        stddev: float = round_to_float32(
            math.sqrt(round_to_float32(lane_sum(variance) / lane_sum(weights)))
        )
        result = Stats(minimum, mean, maximum, stddev, round_to_float32(mean - stddev))
    if len(tile.stats) >= 8192:
        tile.stats.clear()
    tile.stats[cache_key] = result
    return result


@njit(cache=True)
def color(tile: DetectionTile, x: float, y: float, index: int, ring: bool = False) -> DoubleArray:
    """Return three weighted float32 RGB channels stored as float64, or zeroes."""
    if not 0 <= index < 1056:
        return np.zeros(3)
    cache_key: CacheKey = (int(math.floor(x * 4)), int(math.floor(y * 4)), index, int(ring))
    if cache_key in tile.colors:
        return np.array(tile.colors[cache_key])
    weights, _, values = support(tile, cache_key[0] * 0.25, cache_key[1] * 0.25, index, ring)
    total_weight: np.float32 = np.float32(0)
    channel_sums: FloatArray = np.zeros(3, np.float32)
    for sample_index in range(len(weights)):
        total_weight = np.float32(total_weight + weights[sample_index])
        for channel in range(3):
            channel_sums[channel] = np.float32(
                channel_sums[channel]
                + np.float32(values[sample_index, channel] * weights[sample_index])
            )
    result: DoubleArray = np.zeros(3)
    if total_weight > 0:
        for channel in range(3):
            result[channel] = np.float64(np.float32(channel_sums[channel] / total_weight))
    if len(tile.colors) >= 4096:
        tile.colors.clear()
    tile.colors[cache_key] = (result[0], result[1], result[2])
    return result


@njit(cache=True)
def _extend_ray(
    tile: DetectionTile,
    x: float,
    y: float,
    dx: int,
    dy: int,
    center: tuple[float, float, float, float],
    luminance_samples: NDArray[np.float64],
    color_distances: NDArray[np.float64],
    available: int,
    needed: int,
) -> int:
    """Fill ray samples through needed, stopping at bounds; return the filled count."""
    while available <= needed:
        sample_x, sample_y = (
            rays.add_float32(x, available * dx),
            rays.add_float32(y, available * dy),
        )
        if not inside(tile, sample_x, sample_y):
            break
        sample_rgba: RGBA = sample(tile, sample_x, sample_y)
        luminance_samples[available] = rays.luma(sample_rgba)
        red_delta, green_delta, blue_delta = (
            rays.subtract_float32(center[0], sample_rgba[0]),
            rays.subtract_float32(center[1], sample_rgba[1]),
            rays.subtract_float32(center[2], sample_rgba[2]),
        )
        color_distances[available] = round_to_float32(
            math.sqrt(
                rays.add_float32(
                    rays.add_float32(
                        rays.multiply_float32(green_delta, green_delta),
                        rays.multiply_float32(red_delta, red_delta),
                    ),
                    rays.multiply_float32(blue_delta, blue_delta),
                )
            )
        )
        available += 1
    return available


@njit(cache=True)
def initial(tile: DetectionTile, x: float, y: float) -> tuple[bool, float, float, float]:
    """Trace native ray extents, repair outliers, and validate the re-centered size estimate."""
    center: RGBA = sample(tile, x, y)
    center_luma: float = rays.luma(center)
    limit: int = 32 if center_luma < round_to_float32(0.75) else 266
    radii: DoubleArray = np.full(8, -1.0)
    for direction in range(8):
        dx, dy = rays.DIRECTIONS[direction]
        length: float = round_to_float32(math.sqrt(2)) if direction % 2 else 1.0
        luminance_samples, color_distances = np.zeros(266), np.zeros(266)
        luminance_samples[0] = center_luma
        rejected: bool
        available: int
        available, rejected = 1, False
        for sample_index in range(1, limit):
            width: int = int(rays.add_float32(rays.multiply_float32(sample_index, 0.25), 1.0))
            start: int = sample_index - width + 1
            available = _extend_ray(
                tile,
                x,
                y,
                dx,
                dy,
                center,
                luminance_samples,
                color_distances,
                available,
                sample_index,
            )
            if available <= sample_index:
                radii[direction] = rays.multiply_float32(sample_index, length)
                rejected = True
                break
            previous_luminance: float = rays.sum_float32(luminance_samples[start - width : start])
            current_luminance: float = rays.sum_float32(luminance_samples[start : sample_index + 1])
            previous_color_distance: float = rays.multiply_float32(
                rays.sum_float32(color_distances[start - width : start]), 0.125
            )
            current_color_distance: float = rays.multiply_float32(
                rays.sum_float32(color_distances[start : sample_index + 1]), 0.125
            )
            delta: float = rays.subtract_float32(
                rays.subtract_float32(previous_luminance, previous_color_distance),
                rays.subtract_float32(current_luminance, current_color_distance),
            )
            if delta > rays.multiply_float32(
                width, round_to_float32(0.04)
            ) or -delta > rays.multiply_float32(width, round_to_float32(0.015)):
                end: int = min(sample_index + width, limit - 1)
                available = _extend_ray(
                    tile, x, y, dx, dy, center, luminance_samples, color_distances, available, end
                )
                if available <= end:
                    radii[direction] = rays.multiply_float32(sample_index, length)
                    rejected = True
                    break
                next_luminance: float = rays.sum_float32(
                    luminance_samples[sample_index + 1 : end + 1]
                )
                next_color_distance: float = rays.multiply_float32(
                    rays.sum_float32(color_distances[sample_index + 1 : end + 1]), 0.125
                )
                next_delta: float = rays.subtract_float32(
                    rays.subtract_float32(current_luminance, current_color_distance),
                    rays.subtract_float32(next_luminance, next_color_distance),
                )
                if delta > rays.multiply_float32(
                    width, round_to_float32(0.04)
                ) and rays.subtract_float32(delta, next_delta) > rays.multiply_float32(
                    width, round_to_float32(0.01)
                ):
                    background: float = min(next_luminance, current_luminance)
                    ratio: float = rays.divide_float32(
                        rays.subtract_float32(current_luminance, background),
                        rays.subtract_float32(previous_luminance, background),
                    )
                    radii[direction] = rays.multiply_float32(
                        rays.add_float32(
                            rays.multiply_float32(ratio, width), rays.subtract_float32(start, 0.625)
                        ),
                        length,
                    )
                    rejected = True
                    break
                if -delta > rays.multiply_float32(
                    width, round_to_float32(0.015)
                ) and -next_delta > rays.multiply_float32(width, round_to_float32(0.015)):
                    return False, x, y, 0.0
        if not rejected and direction == 0:
            return False, x, y, 0.0
    valid_extents, position, repaired = repair((x, y), radii)
    center_x, center_y = position
    if (
        not valid_extents
        or abs(rays.subtract_float32(center_x, x)) > 2
        or abs(rays.subtract_float32(center_y, y)) > 2
        or not inside(tile, center_x, center_y)
    ):
        return False, x, y, 0.0
    positive_radii: list[float] = [radius for radius in repaired if radius > 0]
    size: float = (
        rays.divide_float32(rays.sum_float32(positive_radii), len(positive_radii))
        if len(positive_radii)
        else 0.0
    )
    return 0 < size <= 128, center_x, center_y, size


@njit(cache=True)
def luma_color(value: DoubleArray) -> float:
    """Combine weighted color channels using the shrink-stage operation order."""
    luminance: float = (
        value[0] * numeric.LUMA[0] + value[1] * numeric.LUMA[1] + value[2] * numeric.LUMA[2]
    )
    return luminance


@njit(cache=True, error_model="numpy")
def shrink(tile: DetectionTile, x: float, y: float, size: float) -> float:
    """Shrink the initial radius using native disk/ring luminance thresholds."""
    index: int = int(size * 1.32)
    if index > 8:
        ring_luminance, disk_luminance = (
            luma_color(color(tile, x, y, index, True)),
            luma_color(color(tile, x, y, int(0.4 * size))),
        )
        ratio: float = (
            ring_luminance / disk_luminance
            if disk_luminance
            else (math.nan if not ring_luminance else math.inf)
        )
        threshold, minimum_index = (
            0.475 * (ring_luminance + disk_luminance) * ratio,
            int(2 * size + 0.125),
        )
    elif size >= 2:
        threshold, minimum_index = 0.9 * luma_color(color(tile, x, y, 4)), 6
    else:
        return size
    index = int(size * 4 + 0.125)
    while index > minimum_index and luma_color(color(tile, x, y, index, True)) < threshold:
        index -= 1
    return 0.25 * (index + 2)


@njit(cache=True)
def refine_position(
    tile: DetectionTile, x: float, y: float, size: float
) -> tuple[float, float, bool]:
    """Hill-climb in native direction/tie order and report whether the center moved."""
    x, y = round_to_float32(x), round_to_float32(y)
    index: int = max(1, int(size * 4))
    initial_stats: CompiledStats | None = stats(tile, x, y, index)
    if initial_stats is None:
        return x, y, False
    best, step, changed = initial_stats.score, max(round_to_float32(size * 0.125), 0.25), False
    diagonal_step: float = 0.7071067690849304
    directions: tuple[Position, ...] = (
        (1.0, 0.0),
        (diagonal_step, diagonal_step),
        (0.0, 1.0),
        (-diagonal_step, diagonal_step),
        (-1.0, 0.0),
        (-diagonal_step, -diagonal_step),
        (0.0, -1.0),
        (diagonal_step, -diagonal_step),
    )
    for _ in range(6):
        selected: int = -1
        for direction_index in range(8):
            dx, dy = directions[direction_index]
            value: CompiledStats | None = stats(
                tile,
                round_to_float32(x + round_to_float32(dx * step)),
                round_to_float32(y + round_to_float32(dy * step)),
                index,
            )
            if value is not None and value.score > best + 1e-6:
                best, selected = value.score, direction_index
        if selected < 0:
            break
        dx, dy = directions[selected]
        x, y = (
            round_to_float32(x + round_to_float32(dx * step)),
            round_to_float32(y + round_to_float32(dy * step)),
        )
        changed, step = True, max(round_to_float32(step * round_to_float32(0.8)), 0.25)
    return x, y, changed


@njit(cache=True)
def refine_size(tile: DetectionTile, x: float, y: float, size: float) -> tuple[float, bool]:
    """Refine source radius in native iteration order and report a size change."""
    direction: int
    initial_size, direction = size, 0
    for iteration in range(2, 8):
        stop: bool = False
        previous_size, step = size, max(size * 0.05 / iteration, 0.25)
        if size < 1.25:
            # Only indices outside [0,1056) yield None; fixed index10 is always valid.
            peak, background = color(tile, x, y, 5).max(), stats(tile, x, y, 10, True).minimum  # type: ignore[union-attr]
            size, direction = numeric.small_size_step(size, direction, peak, background)
        else:
            outer_stats, inner_stats, edge_stats = (
                stats(tile, x, y, int(8 * size) + 4, True),
                stats(tile, x, y, int(2.4 * size)),
                stats(tile, x, y, int(4 * size) - 2, True),
            )
            size, direction, stop = numeric.large_size_step(
                size, step, direction, outer_stats, inner_stats, edge_stats
            )
        if numeric.size_refinement_done(size, previous_size, stop):
            break
    return size, size != initial_size


@njit(cache=True)
def covered(x: float, y: float, excluded: DoubleArray) -> bool:
    """Test a global point against N×3 float64 exclusion disks [x,y,size]."""
    for source_index in range(len(excluded)):
        if (x - excluded[source_index, 0]) ** 2 + (y - excluded[source_index, 1]) ** 2 < excluded[
            source_index, 2
        ] ** 2:
            return True
    return False


@njit(cache=True)
def _refine_candidate(
    tile: DetectionTile,
    x: float,
    y: float,
    size: float,
    coarse: bool,
    excluded: NDArray[np.float64],
) -> tuple[float, float, float] | None:
    """Alternate native refinement stages, including rollback and convergence limits."""
    old_x, old_y = x, y
    x, y, moved = refine_position(tile, x, y, size)
    size, resized = refine_size(tile, x, y, size)
    maximum: float
    limit, maximum = int(size + 8), 1.5 * size + 2
    if not coarse:
        maximum = min(maximum, 20.0)
    converged: bool = False
    for _ in range(limit):
        if not moved and not resized:
            converged = True
            break
        if not inside(tile, x, y):
            x, y, converged = old_x, old_y, True
            break
        old_x, old_y = x, y
        x, y, moved = refine_position(tile, x, y, size)
        if moved or resized:
            size, resized = refine_size(tile, x, y, size)
        if size > maximum or covered(x, y, excluded):
            return None
    return numeric.refinement_result(x, y, size, converged)


@njit(cache=True)
def _accept_shape(tile: DetectionTile, x: float, y: float, size: float) -> bool:
    """Evaluate the final shape supports in their original sampling order."""
    size_index: int = max(1, int(4 * size))
    if size_index < 12:
        inner_disk, outer_disk = (
            stats(tile, x, y, max(1, size_index - 2)),
            stats(tile, x, y, 2 * size_index + 4, True),
        )
        if (
            inner_disk is None
            or outer_disk is None
            or not (
                inner_disk.score >= outer_disk.score and inner_disk.score - outer_disk.score >= 0.13
            )
        ):
            return False
    else:
        inner_disk, outer_disk, inner_ring, middle_ring, outer_ring = (
            stats(tile, x, y, size_index // 2),
            stats(tile, x, y, size_index - 3),
            stats(tile, x, y, size_index - 3, True),
            stats(tile, x, y, (6 * size_index) // 5 + 4, True),
            stats(tile, x, y, 2 * size_index + 4, True),
        )
        if (
            inner_disk is None
            or outer_disk is None
            or inner_ring is None
            or middle_ring is None
            or outer_ring is None
            or not numeric.final_shape(
                size, inner_disk, outer_disk, inner_ring, middle_ring, outer_ring
            )
        ):
            return False
    return True


@njit(cache=True)
def qualify(
    tile: DetectionTile, x: float, y: float, coarse: bool, excluded: DoubleArray
) -> CandidateRow | None:
    """Qualify a half-pixel seed and return its eight-field raw source row on success."""
    size: float
    valid_seed, x, y, size = initial(tile, x, y)
    if not valid_seed:
        return None
    size = shrink(tile, x, y, size)
    size_index: int = int(4 * size)
    ring_offset: int = size_index // 8 + 4
    if size_index >= ring_offset:
        inner_stats, outer_stats = (
            stats(tile, x, y, size_index - ring_offset, True),
            stats(tile, x, y, size_index + ring_offset, True),
        )
        if (
            inner_stats is None
            or outer_stats is None
            or not numeric.preliminary_shape(inner_stats, outer_stats, size_index)
        ):
            return None
    refined: tuple[float, float, float] | None = _refine_candidate(
        tile, x, y, size, coarse, excluded
    )
    if refined is None:
        return None
    x, y, size = refined
    x, y, size = (
        numeric.quarter_position(x),
        numeric.quarter_position(y),
        numeric.quarter_size(size),
    )
    if not _accept_shape(tile, x, y, size):
        return None
    if not inside(tile, x, y) or (coarse and size <= 4):
        return None
    ring_count: int = 1 - int((size - 1) * -0.5)
    color_sum: DoubleArray = np.zeros(3)
    for ring_index in range(ring_count):
        color_sum += color(tile, x, y, int(4 * size) + 4 * ring_index, True)
    color_sum[1] = 0.4 * color_sum[0] + 0.3 * (color_sum[1] + color_sum[2])
    color_sum /= color_sum.mean()
    source: DoubleArray = color(tile, x, y, int(4 * size) + 2)
    brightness: np.float64 = np.float64(
        np.float32(np.float32(np.float32(source[0] + source[1]) + source[2]) / np.float32(3))
    )
    return x, y, size, brightness, color_sum[0], color_sum[1], color_sum[2], 1.0


@njit(cache=True)
def repair(position: Position, radii: DoubleArray) -> tuple[bool, Position, DoubleArray]:
    """Return repaired radii even when extent repair fails.

    Outlier marking uses the ORIGINAL adjacent values; repair uses the marked
    adjacent values, never newly repaired ones. Successful non-outliers retain
    original values. Native diagonal recenter factor really is sqrt(2).
    """
    radii = radii.copy()
    position = (round_to_float32(position[0]), round_to_float32(position[1]))
    if not rays.repair_extents(radii):
        return False, position, radii
    return True, rays.recenter(position, radii), radii
