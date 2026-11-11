"""Verified native CPU profiles; all geometry remains in full-image pixel units."""

import math
from collections import OrderedDict
from threading import RLock
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..array_types import BoolArray, Bounds, DoubleArray, FloatArray, IntArray, Position
from ..jit import njit
from .constants import SECONDARY_RAY_COUNT, SECONDARY_RAYS
from .settings import EffectKind, EffectParameters

type IndexArray = NDArray[np.int32]
type IntegerArray = IndexArray | IntArray
type CachedArray = NDArray[np.generic]
type CacheKey = tuple[object, ...]
type CacheValue = CachedArray | tuple[CachedArray, ...]
# Profile: integer x/y offsets, float32 opacity and radius in native pixels.
type ProfileArrays = tuple[IndexArray, IndexArray, FloatArray, FloatArray]
# Scene: flattened sampled pixel index, opacity, radius and source-record index.
type SceneArrays = tuple[IndexArray, FloatArray, FloatArray, IndexArray]
type RawProfile = tuple[IntegerArray, IntegerArray, DoubleArray, DoubleArray]


class ArrayCache[T: CacheValue]:
    """LRU bounded by array bytes, shared safely by cancelling preview workers."""

    def __init__(self, megabytes: int) -> None:
        """Set the byte budget; each retained value is made read-only on insertion."""
        self.limit = megabytes * 1024 * 1024
        self.max_entries = 4096
        self.entries: OrderedDict[CacheKey, tuple[T, int]] = OrderedDict()
        self.bytes = 0
        self.lock = RLock()

    def get(self, key: CacheKey) -> T | None:
        """Return and promote a retained value without copying its immutable arrays."""
        with self.lock:
            item: tuple[T, int] | None = self.entries.get(key)
            if item is not None:
                self.entries.move_to_end(key)
                return item[0]
        return None

    def put(self, key: CacheKey, value: T) -> T:
        """Freeze cacheable arrays and evict least-recently-used values within the budget."""
        arrays: tuple[CachedArray, ...] = value if isinstance(value, tuple) else (value,)
        byte_count: int = sum(array.nbytes for array in arrays)
        if byte_count > self.limit:
            return value
        for array in arrays:
            array.flags.writeable = False
        with self.lock:
            previous: tuple[T, int] | None = self.entries.pop(key, None)
            if previous is not None:
                self.bytes -= previous[1]
            while self.entries and (
                self.bytes + byte_count > self.limit or len(self.entries) >= self.max_entries
            ):
                _, (_, removed_bytes) = self.entries.popitem(last=False)
                self.bytes -= removed_bytes
            self.entries[key] = value, byte_count
            self.bytes += byte_count
        return value


PROFILE_CACHE = ArrayCache[ProfileArrays](64)


def image_diagonal(width: int, height: int) -> int:
    """Return the native float32-rounded diagonal used to scale all effect geometry."""
    return int(np.float32(math.hypot(width, height)) + np.float32(1))


def length_envelope(length: float) -> int:
    """Round CPU ray support upward so nearby length edits can reuse geometry."""
    return max(10, math.ceil(length / 10) * 10)


def ray_coefficients(parameters: EffectParameters, scale: float) -> tuple[float, float]:
    """Return transverse sharpness and radial widening in full-image pixel units."""
    sharpness_factor: float = (
        (parameters.sharpness / 90) ** 1.4
        if parameters.sharpness <= 90
        else 1 + 0.251416 * ((parameters.sharpness - 90) / 10) ** 1.4 / scale
    )
    coefficient: float = (4 * sharpness_factor + 0.08) * scale
    widening: float = 0.15 * (parameters.spread / 100) ** 2 * coefficient
    return coefficient, widening


def geometry_key(kind: EffectKind, parameters: EffectParameters) -> CacheKey:
    """Select only controls affecting reusable sparse-profile support and geometry."""
    if kind in ("primary", "secondary"):
        # Length changes reuse transverse AA. Bounded support envelopes
        # avoid allocating a whole-image maximum-length kernel for short rays.
        envelope: int = length_envelope(parameters.length)
        return (
            kind,
            envelope,
            parameters.sharpness,
            parameters.spread,
            parameters.angle,
            parameters.points if kind == "primary" else SECONDARY_RAY_COUNT,
        )
    if kind == "soft":
        return kind, parameters.size, parameters.falloff
    return (
        kind,
        parameters.size,
        parameters.inner_width,
        parameters.outer_width,
        parameters.inner_falloff,
        parameters.outer_falloff,
    )


def gain(kind: EffectKind, parameters: EffectParameters) -> float:
    """Map the original intensity slider to the group-specific emitted-light gain."""
    exponent: float = {"primary": 2, "secondary": 1.2, "soft": 3, "ring": 1.8}[kind]
    factor: float = {"primary": 128, "secondary": 128 * 0.6, "soft": 1, "ring": 1.14}[kind]
    return cast(float, factor * (parameters.intensity / 100) ** exponent)


def source_colors(colors: ArrayLike, hue: float, saturation: float) -> FloatArray:
    """Copy source RGB and apply the native hue/saturation mapping without normalization."""
    colors = np.asarray(colors, dtype=np.float32).copy()
    green: FloatArray = colors[:, 1].copy()
    colors += green[:, None] * np.array([-hue / 1500, hue / 3000, -hue / 1000])
    np.maximum(colors, 0, out=colors)
    factor: float = saturation / 6 if saturation > 0 else saturation / 150
    colors += factor * (colors - (colors.sum(axis=1, keepdims=True) - colors) / 2)
    return np.maximum(colors, 0)


def directions(kind: EffectKind, parameters: EffectParameters) -> tuple[tuple[int, float], ...]:
    """Return rotated ray angles in tenths of degrees with their relative amplitudes."""
    angle: int = round(parameters.angle * 10)
    match kind:
        case "primary":
            return tuple(
                ((int(ray_index * 3600 / parameters.points + 0.5) + angle) % 3600, 1.0)
                for ray_index in range(parameters.points)
            )
        case "secondary":
            return tuple(
                ((ray_angle + angle) % 3600, weight / 100) for ray_angle, weight in SECONDARY_RAYS
            )
        case _:
            return ((0, 1.0),)


def sine(angle: int) -> float:
    """Evaluate a tenths-degree direction with the native float32 sine rounding."""
    angle %= 3600
    return float(np.float32(math.sin(math.radians((angle % 1800) / 10)))) * (
        1 if angle < 1800 else -1
    )


def radial_extent(kind: EffectKind, parameters: EffectParameters, diagonal: int) -> float:
    """Return the current profile cutoff in full-image pixels, without CPU envelopes."""
    scaled_diagonal: int = max(diagonal, 1414)
    match kind:
        case "primary" | "secondary":
            return cast(
                float,
                min(
                    (0.000015 ** (-(parameters.length + 25) / 190) - 1) * scaled_diagonal / 800,
                    diagonal + 2,
                ),
            )
        case "soft":
            radius: float = 1 + (scaled_diagonal // 3) * (parameters.size / 100) ** 3
            power: float = 0.6 + 0.124 * parameters.falloff
            return cast(float, radius * (1 - (0.00025 / 5.7) ** (1 / power)))
        case _:
            peak: float = (1 + 0.01 * (parameters.size + 25) ** 2) * scaled_diagonal / 3768
            return peak * (1.03 + 0.02 * parameters.outer_width)


def _coordinates(extent: int, sampling: int, residue: int) -> IndexArray:
    """Build native integer offsets in one sampling residue class."""
    start: int = math.ceil((-extent - residue) / sampling) * sampling + residue
    return np.arange(start, extent + 1, sampling, dtype=np.int32)


def scene_batch_size(
    kind: EffectKind, parameters: EffectParameters, diagonal: int, sampling: int
) -> int:
    """Bound estimated sparse contributions, including long full-size exports."""
    if kind in ("primary", "secondary"):
        scale: float = 800 / max(diagonal, 1414)
        envelope: int = length_envelope(parameters.length)
        cutoff: float = min((0.000015 ** (-(envelope + 25) / 190) - 1) / scale, diagonal + 2)
        coefficient, _ = ray_coefficients(parameters, scale)
        band: float = (1 + cutoff * 0.15 * (parameters.spread / 100) ** 2 * coefficient) / (
            coefficient / math.sqrt(2)
        ) + 0.875
        per_source: float = (
            (cutoff / sampling + 2) * (2 * band / sampling + 2) * len(directions(kind, parameters))
        )
    else:
        cutoff = radial_extent(kind, parameters, diagonal)
        per_source = (2 * cutoff / sampling + 2) ** 2
    # Four arrays × four bytes; approximately16MiB per scene batch. A single
    # extreme ray can exceed this estimate, but catalogs cannot grow the batch.
    return max(1, int(1048576 / max(per_source, 1)))


@njit(cache=True, nogil=True)
def _footprint(distance: float, ratio: float) -> float:
    """Average the cubic ray falloff over 15 transverse samples of one pixel."""
    total: float = 0.0
    for sample in range(15):
        total += max(1 - abs(distance + sample / 14 - 0.5) * ratio, 0.0) ** 3
    return total / 15


@njit(cache=True, nogil=True)
def transverse_profile(
    cross: DoubleArray, coefficient: float, denominator: DoubleArray
) -> DoubleArray:
    """The native 15 × 7 quadrature: point samples far away, supersampled near the ray."""
    result: DoubleArray = np.empty(len(cross))
    for index in range(len(cross)):
        distance: float = cross[index]
        ratio: float = coefficient / denominator[index]
        if distance > 1.5:
            result[index] = max(1 - abs(distance) * coefficient / denominator[index], 0.0) ** 3
        elif distance == 1.5:
            result[index] = _footprint(distance, ratio)
        else:
            total: float = 0.0
            for step in range(7):
                shifted: float = distance + (step - 3) / 8
                total += (
                    _footprint(shifted, ratio)
                    if shifted <= 1.5
                    else max(1 - abs(shifted) * ratio, 0.0) ** 3
                )
            result[index] = total / 7
    return result


def _sampled_cross_offsets(
    width: int,
    height: int,
    origins: IndexArray,
    residues: IndexArray,
    source_indices: IntArray,
    along: IntegerArray,
    middle: DoubleArray,
    band: float,
    axis: int,
    sampling: int,
) -> tuple[IntegerArray, IntArray, IntArray]:
    """Expand clipped transverse spans, retaining each sample's source and major offset."""
    low: IntArray = np.ceil((middle - band - residues[source_indices, 1 - axis]) / sampling).astype(
        np.int64
    )
    high: IntArray = np.floor(
        (middle + band - residues[source_indices, 1 - axis]) / sampling
    ).astype(np.int64)
    low = np.maximum(
        low,
        np.ceil(
            (-origins[source_indices, 1 - axis] - residues[source_indices, 1 - axis]) / sampling
        ).astype(np.int64),
    )
    high = np.minimum(
        high,
        np.floor(
            (
                (height if axis == 0 else width)
                - 1
                - origins[source_indices, 1 - axis]
                - residues[source_indices, 1 - axis]
            )
            / sampling
        ).astype(np.int64),
    )
    counts: IntArray = np.maximum(high - low + 1, 0)
    offsets: IntArray = np.repeat(np.cumsum(counts) - counts, counts)
    across: IntArray = (np.repeat(low, counts) + np.arange(int(counts.sum())) - offsets) * sampling
    source_indices = np.repeat(source_indices, counts)
    across += residues[source_indices, 1 - axis]
    along = cast(IntegerArray, np.repeat(along, counts))
    return along, across, source_indices


def sampled_ray_scene(
    width: int,
    height: int,
    positions: tuple[Position, ...],
    parameters: EffectParameters,
    diagonal: int,
    sampling: int,
    angle: int,
) -> SceneArrays:
    """Batch many sampled sources through one exact ray rasterizer.

    Sampling creates many residue classes. Evaluating them together avoids
    thousands of tiny profile allocations during an angle or sharpness edit.
    """
    scale: float = 800 / max(diagonal, 1414)
    envelope: int = length_envelope(parameters.length)
    cutoff: float = min((0.000015 ** (-(envelope + 25) / 190) - 1) / scale, diagonal + 2)
    extent: int = math.ceil(cutoff + 1)
    coefficient, widening = ray_coefficients(parameters, scale)
    direction_sine, direction_cosine = sine(angle), sine(angle + 900)
    horizontal: bool = abs(direction_sine) >= abs(direction_cosine)
    axis: int = 0 if horizontal else 1
    cross_coefficient: float = coefficient * (
        abs(direction_sine) if horizontal else abs(direction_cosine)
    )
    slope: float = (
        -direction_cosine / direction_sine if horizontal else -direction_sine / direction_cosine
    )
    raw_positions: DoubleArray = np.asarray(positions) + 0.5
    origins: IndexArray = np.floor(raw_positions).astype(np.int32)
    centers: DoubleArray = np.floor(4 * (raw_positions - origins)) / 4 - 0.5
    residues: IndexArray = (-origins) % sampling
    grid: IntArray = np.arange(-extent // sampling * sampling, extent + sampling, sampling)
    along: IntegerArray = grid[None, :] + residues[:, axis, None]
    positive: bool = direction_sine > 0 if horizontal else direction_cosine < 0
    valid: BoolArray = (along >= 0) if positive else (along < 0)
    valid &= np.abs(along) <= extent
    world_along: IntegerArray = along + origins[:, axis, None]
    valid &= (world_along >= 0) & (world_along < (width if horizontal else height))
    source_indices: IntArray
    column_indices: IntArray
    source_indices, column_indices = np.nonzero(valid)
    along = along[source_indices, column_indices]
    middle: DoubleArray = (
        centers[source_indices, 1 - axis] + (along - centers[source_indices, axis]) * slope
    )
    band: float = (1 + cutoff * widening) / cross_coefficient + 0.875
    along, across, source_indices = _sampled_cross_offsets(
        width, height, origins, residues, source_indices, along, middle, band, axis, sampling
    )
    offset_x, offset_y = (along, across) if horizontal else (across, along)
    world_x: IntegerArray
    world_y: IntegerArray
    world_x, world_y = offset_x + origins[source_indices, 0], offset_y + origins[source_indices, 1]
    valid = (world_x >= 0) & (world_y >= 0) & (world_x < width) & (world_y < height)
    offset_x, offset_y, world_x, world_y = (
        array[valid] for array in (offset_x, offset_y, world_x, world_y)
    )
    source_indices = source_indices[valid]
    delta_x, delta_y = offset_x - centers[source_indices, 0], offset_y - centers[source_indices, 1]
    radius: DoubleArray = np.hypot(delta_x, delta_y)
    cross: DoubleArray = (
        np.abs(delta_y - delta_x * slope) if horizontal else np.abs(delta_x - delta_y * slope)
    )
    denominator: DoubleArray = 1 + radius * widening
    valid = (radius <= cutoff) & (cross < denominator / cross_coefficient + 0.875)
    radius, cross, denominator = (array[valid] for array in (radius, cross, denominator))
    world_x, world_y = (array[valid] for array in (world_x, world_y))
    source_indices = source_indices[valid]
    values: DoubleArray = transverse_profile(cross, cross_coefficient, denominator)
    pixel_indices: IntArray = (world_y // sampling) * (
        (width + sampling - 1) // sampling
    ) + world_x // sampling
    valid = values > 0
    return cast(
        SceneArrays,
        tuple(
            np.ascontiguousarray(array[valid], dtype=dtype)
            for array, dtype in (
                (pixel_indices, np.int32),
                (values, np.float32),
                (radius, np.float32),
                (source_indices, np.int32),
            )
        ),
    )


def _sparse_ray_profile(
    parameters: EffectParameters,
    diagonal: int,
    extent: int,
    cutoff: float,
    center: tuple[float, float],
    sampling: int,
    residue: tuple[int, int],
    angle: int,
) -> RawProfile:
    """Rasterize one ray into integer offsets, transverse opacity and radius."""
    center_x, center_y = center
    scale: float = 800 / max(diagonal, 1414)
    coefficient, widening = ray_coefficients(parameters, scale)
    direction_cosine: float
    direction_sine: float
    direction_sine, direction_cosine = sine(angle), sine(angle + 900)
    horizontal: bool = abs(direction_sine) >= abs(direction_cosine)
    dominant: float = abs(direction_sine) if horizontal else abs(direction_cosine)
    cross_coefficient: float = coefficient * dominant
    slope: float = (
        -direction_cosine / direction_sine if horizontal else -direction_sine / direction_cosine
    )
    axis: int = 0 if horizontal else 1
    centers: Position = (center_x, center_y)
    along: IndexArray = _coordinates(extent, sampling, residue[axis])
    positive: bool = direction_sine > 0 if horizontal else direction_cosine < 0
    along = along[along >= 0] if positive else along[along < 0]
    middle: DoubleArray = cast(DoubleArray, centers[1 - axis] + (along - centers[axis]) * slope)
    band: float = (1 + cutoff * widening) / cross_coefficient + 0.875
    low: IntArray = np.ceil((middle - band - residue[1 - axis]) / sampling).astype(np.int64)
    high: IntArray = np.floor((middle + band - residue[1 - axis]) / sampling).astype(np.int64)
    counts: IntArray = np.maximum(high - low + 1, 0)
    offsets: IntArray = np.repeat(np.cumsum(counts) - counts, counts)
    across: IntArray = (
        np.repeat(low, counts) + np.arange(int(counts.sum())) - offsets
    ) * sampling + residue[1 - axis]
    along = np.repeat(along, counts)
    offset_x: IntegerArray
    offset_y: IntegerArray
    offset_x, offset_y = (along, across) if horizontal else (across, along)
    delta_x: DoubleArray
    delta_y: DoubleArray
    delta_x, delta_y = cast(
        tuple[DoubleArray, DoubleArray], (offset_x - center_x, offset_y - center_y)
    )
    radius: DoubleArray = np.hypot(delta_x, delta_y)
    cross: DoubleArray = (
        np.abs(delta_y - delta_x * slope) if horizontal else np.abs(delta_x - delta_y * slope)
    )
    denominator: DoubleArray = 1 + radius * widening
    valid: BoolArray = (radius <= cutoff) & (cross < denominator / cross_coefficient + 0.875)
    offset_x, offset_y = offset_x[valid], offset_y[valid]
    radius, cross, denominator = (array[valid] for array in (radius, cross, denominator))

    values: DoubleArray = transverse_profile(cross, cross_coefficient, denominator)
    return offset_x, offset_y, values, radius


def _sparse_radial_profile(
    kind: EffectKind,
    parameters: EffectParameters,
    diagonal: int,
    extent: int,
    center: tuple[float, float],
    sampling: int,
    residue: tuple[int, int],
    bounds: tuple[int, int, int, int] | None,
) -> RawProfile:
    """Evaluate soft or ring opacity on the bounded sampled radial grid."""
    center_x, center_y = center
    x: IndexArray
    y: IndexArray
    x, y = (
        _coordinates(extent, sampling, residue[0]),
        _coordinates(extent, sampling, residue[1]),
    )
    if bounds is not None:
        x = x[(x >= bounds[0]) & (x < bounds[2])]
        y = y[(y >= bounds[1]) & (y < bounds[3])]
    offset_x: IndexArray
    offset_y: IndexArray
    offset_x, offset_y = np.meshgrid(x, y)
    offset_x, offset_y = offset_x.ravel(), offset_y.ravel()
    radius: DoubleArray = np.hypot(offset_x - center_x, offset_y - center_y)
    if kind == "soft":
        size: float = 1 + (max(diagonal, 1414) // 3) * (parameters.size / 100) ** 3
        values: DoubleArray = np.maximum(
            5.7 * np.maximum(1 - radius / size, 0) ** (0.6 + 0.124 * parameters.falloff) - 0.00025,
            0,
        )
    else:
        peak: float = (1 + 0.01 * (parameters.size + 25) ** 2) * max(diagonal, 1414) / 3768
        inner: float = peak * (1 - (0.03 + 0.0097 * parameters.inner_width))
        outer: float = peak * (1.03 + 0.02 * parameters.outer_width)
        values = np.where(
            radius < peak,
            np.clip((radius - inner) / (peak - inner), 0, 1)
            ** (0.08 * (parameters.inner_falloff + 10)),
            np.clip((outer - radius) / (outer - peak), 0, 1)
            ** (0.08 * (parameters.outer_falloff + 10)),
        )
    return offset_x, offset_y, values, radius


def sparse_profile(
    kind: EffectKind,
    parameters: EffectParameters,
    diagonal: int,
    phase: tuple[int, int],
    sampling: int,
    residue: tuple[int, int],
    angle: int = 0,
    bounds: Bounds | None = None,
) -> ProfileArrays:
    """One ray's integer offsets, opacity, radius; AA always uses full pixels."""
    key: CacheKey = (
        geometry_key(kind, parameters),
        diagonal,
        phase,
        sampling,
        residue,
        angle,
        bounds,
    )
    cached: ProfileArrays | None = PROFILE_CACHE.get(key)
    if cached is not None:
        return cached
    cutoff: float = radial_extent(kind, parameters, diagonal)
    if kind in ("primary", "secondary"):
        envelope: int = length_envelope(parameters.length)
        cutoff = min(
            (0.000015 ** (-(envelope + 25) / 190) - 1) * max(diagonal, 1414) / 800, diagonal + 2
        )
    extent: int = math.ceil(cutoff + 1)
    center: Position = phase[0] / 4 - 0.5, phase[1] / 4 - 0.5
    if kind in ("primary", "secondary"):
        offset_x, offset_y, values, radius = _sparse_ray_profile(
            parameters, diagonal, extent, cutoff, center, sampling, residue, angle
        )
    else:
        offset_x, offset_y, values, radius = _sparse_radial_profile(
            kind, parameters, diagonal, extent, center, sampling, residue, bounds
        )
    valid: BoolArray = values > 0
    arrays: ProfileArrays = cast(
        ProfileArrays,
        tuple(
            np.ascontiguousarray(array[valid], dtype=dtype)
            for array, dtype in (
                (offset_x, np.int32),
                (offset_y, np.int32),
                (values, np.float32),
                (radius, np.float32),
            )
        ),
    )
    return PROFILE_CACHE.put(key, arrays)
