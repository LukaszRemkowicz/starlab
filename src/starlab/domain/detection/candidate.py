"""Candidate qualification, subpixel positioning and source-size estimation.

See docs/architecture.md for the detection pipeline and its limitations."""

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal, cast

import numpy as np
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
from .catalog import blend_record
from .numeric import (
    LUMA,
    Stats,
    combine_variance_lanes,
    final_shape,
    large_size_step,
    preliminary_shape,
    quarter_position,
    quarter_size,
    refine_position,
    refinement_result,
    size_refinement_done,
    small_size_step,
)
from .precision import round_to_float32
from .rays import initial_candidate

type KernelKind = Literal["disk", "ring"]


@njit(cache=True, nogil=True)
def _luminance(rgb: FloatArray) -> FloatArray:
    """Return H×W float32 luminance from an H×W×3 decoded detector tile."""
    luminance: FloatArray = np.empty(rgb.shape[:2], np.float32)
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            luminance[y, x] = (
                rgb[y, x, 0] * np.float32(LUMA[0]) + rgb[y, x, 2] * np.float32(LUMA[2])
            ) + rgb[y, x, 1] * np.float32(LUMA[1])
    return luminance


class Image:
    """Decoded RGB tile with global origin, allowed mask, and native sampling caches."""

    def __init__(
        self, rgb: FloatArray, origin: tuple[int, int] = (0, 0), allowed: BoolArray | None = None
    ) -> None:
        """Own decoded RGB/luminance sampling state and bounded per-tile caches."""
        self.rgb = np.asarray(rgb, dtype=np.float32)
        self.height: int
        self.width: int
        self.height, self.width = self.rgb.shape[:2]
        self.origin_x, self.origin_y = origin
        self.allowed = allowed
        self.bounds = (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.width,
            self.origin_y + self.height,
        )
        self._stats_cached = lru_cache(maxsize=8192)(self._stats)
        self._color_cached = lru_cache(maxsize=4096)(self._color)
        self.luminance = _luminance(self.rgb)

    def inside(self, x: float, y: float) -> bool:
        """Test global coordinates against this tile’s half-open bounds."""
        return (
            self.origin_x <= x < self.origin_x + self.width
            and self.origin_y <= y < self.origin_y + self.height
        )

    def sample(self, x: float, y: float) -> RGBA:
        """Sample by native coordinate truncation; excluded/outside points have zero alpha."""
        pixel_x, pixel_y = int(x) - self.origin_x, int(y) - self.origin_y
        if not (0 <= pixel_x < self.width and 0 <= pixel_y < self.height):
            return 0, 0, 0, 0
        if self.allowed is not None and not self.allowed[pixel_y, pixel_x]:
            return 0, 0, 0, 0
        return cast(RGBA, (*map(float, self.rgb[pixel_y, pixel_x]), 1))

    @staticmethod
    @lru_cache(maxsize=128)
    def kernel(
        kind: KernelKind, index: int, phase_index_x: int, phase_index_y: int
    ) -> tuple[IntArray, IntArray, FloatArray]:
        """Return sparse X/Y offsets and float32 weights for quarter-phase support."""
        radius: float = index * 0.25
        phase_x: float = phase_index_x * 0.25 - 0.5
        phase_y: float = phase_index_y * 0.25 - 0.5
        # Native square enumeration uses X phase for BOTH bounds. Annular
        # supports are deliberately clipped; an ideal unbounded ring differs.
        if kind == "disk":
            extent: int = int(
                round_to_float32(round_to_float32(radius + round_to_float32(0.6)) + phase_x)
            )
            lower, upper = -extent - 1, extent + 2
        else:
            extent = int(
                round_to_float32(round_to_float32(radius + phase_x) + round_to_float32(1.65))
            )
            lower, upper = -extent, extent + 1
        grid_x: IntArray
        grid_y: IntArray
        grid_y, grid_x = np.mgrid[lower:upper, lower:upper]
        offset_x: FloatArray = (grid_x - phase_x).astype(np.float32)
        offset_y: FloatArray = (grid_y - phase_y).astype(np.float32)
        distance: FloatArray = np.sqrt(offset_x * offset_x + offset_y * offset_y)
        weights: FloatArray = (
            np.clip(np.float32(radius + np.float32(0.6)) - distance, 0, 1)
            if kind == "disk"
            else np.maximum(
                0,
                (np.float32(1.15) - np.abs(np.float32(radius) - distance))
                * np.float32(0.8695652484893799),
            )
        )
        positive_weights: BoolArray = weights > 0
        return (
            grid_x[positive_weights],
            grid_y[positive_weights],
            weights[positive_weights].astype(np.float32),
        )

    def values(
        self, x: float, y: float, index: int, kind: KernelKind
    ) -> tuple[IntArray, IntArray, FloatArray] | None:
        """Clip sparse support to allowed tile pixels; invalid table indices return None."""
        if not 0 <= index < 1056:
            return None
        pixel_x, pixel_y = math.floor(x), math.floor(y)
        offset_x, offset_y, weights = self.kernel(
            kind, index, int(4 * (x - pixel_x)), int(4 * (y - pixel_y))
        )
        sample_x, sample_y = offset_x + pixel_x - self.origin_x, offset_y + pixel_y - self.origin_y
        valid: BoolArray = (
            (sample_x >= 0) & (sample_y >= 0) & (sample_x < self.width) & (sample_y < self.height)
        )
        if self.allowed is not None:
            valid_indices: NDArray[np.intp] = np.flatnonzero(valid)
            valid[valid_indices] &= self.allowed[sample_y[valid_indices], sample_x[valid_indices]]
        return sample_x[valid], sample_y[valid], weights[valid]

    def stats(self, x: float, y: float, index: int, kind: KernelKind = "disk") -> Stats | None:
        """Read cached weighted statistics at floor-quantized quarter coordinates."""
        return self._stats_cached(math.floor(x * 4) * 0.25, math.floor(y * 4) * 0.25, index, kind)

    def _stats(self, x: float, y: float, index: int, kind: KernelKind = "disk") -> Stats | None:
        """Compute native weighted statistics, including the eight-lane variance reduction."""
        values: tuple[IntArray, IntArray, FloatArray] | None = self.values(x, y, index, kind)
        if values is None:
            return None
        sample_x, sample_y, weights = values
        luminance_samples: FloatArray = self.luminance[sample_y, sample_x]
        total_weight: float = float(np.cumsum(weights, dtype=np.float32)[-1]) if len(weights) else 0
        if not total_weight:
            return Stats(1, 0, 0, 0, 0)
        mean: float = round_to_float32(
            float(np.cumsum(luminance_samples * weights, dtype=np.float32)[-1]) / total_weight
        )
        delta: FloatArray = luminance_samples - np.float32(mean)
        variance: FloatArray = delta * delta * weights

        def native_sum(values: FloatArray) -> float:
            """Reduce eight float32 lanes in native combination order, then accumulate the tail."""
            lane_count: int = len(values) // 8 * 8
            if lane_count:
                lanes: FloatArray = np.cumsum(
                    values[:lane_count].reshape(-1, 8), axis=0, dtype=np.float32
                )[-1]
                result: float = combine_variance_lanes(lanes)
            else:
                result = 0.0
            for value in values[lane_count:]:
                result = round_to_float32(result + float(value))
            return result

        stddev: float = round_to_float32(
            math.sqrt(round_to_float32(native_sum(variance) / native_sum(weights)))
        )
        return Stats(
            min(1, float(luminance_samples.min())),
            mean,
            max(0, float(luminance_samples.max())),
            stddev,
            round_to_float32(mean - stddev),
        )

    def color(
        self, x: float, y: float, index: int, kind: KernelKind = "disk"
    ) -> FloatArray | DoubleArray:
        """Read cached weighted RGB at floor-quantized quarter coordinates."""
        return self._color_cached(math.floor(x * 4) * 0.25, math.floor(y * 4) * 0.25, index, kind)

    def _color(
        self, x: float, y: float, index: int, kind: KernelKind = "disk"
    ) -> FloatArray | DoubleArray:
        """Compute sparse weighted RGB; empty or invalid supports return three zeroes."""
        values: tuple[IntArray, IntArray, FloatArray] | None = self.values(x, y, index, kind)
        if values is None:
            return np.zeros(3)
        sample_x, sample_y, weights = values
        if weights.sum() <= 0:
            return np.zeros(3)
        result: FloatArray = (
            np.cumsum(self.rgb[sample_y, sample_x] * weights[:, None], axis=0, dtype=np.float32)[-1]
            / np.cumsum(weights, dtype=np.float32)[-1]
        )
        return result

    def clear_cache(self) -> None:
        """Dispose cache wrappers and their bound-method ownership cycles after tile use."""
        self._stats_cached.cache_clear()
        self._color_cached.cache_clear()
        # Cached bound methods own this tile; break that cycle immediately so
        # empty tiles cannot retain their RGB arrays until a later GC sweep.
        del self._stats_cached
        del self._color_cached


def shrink_initial(image: Image, x: float, y: float, size: float) -> float:
    """Shrink the ray estimate using native annular/disk luminance comparisons."""
    index: int = int(size * 1.32)
    if index > 8:
        ring_luminance: float = float(image.color(x, y, index, "ring") @ LUMA)
        disk_luminance: float = float(image.color(x, y, int(0.4 * size)) @ LUMA)
        ratio: float = (
            ring_luminance / disk_luminance
            if disk_luminance
            else (math.nan if not ring_luminance else math.inf)
        )
        threshold: float = 0.475 * (ring_luminance + disk_luminance) * ratio
        minimum_index: int = int(2 * size + 0.125)
    elif size >= 2:
        threshold = 0.9 * float(image.color(x, y, 4) @ LUMA)
        minimum_index = 6
    else:
        return size
    index = int(size * 4 + 0.125)
    while index > minimum_index and float(image.color(x, y, index, "ring") @ LUMA) < threshold:
        index -= 1
    return 0.25 * (index + 2)


def refine_size(image: Image, x: float, y: float, size: float) -> tuple[float, bool]:
    """Refine radius in native step order and report whether it changed."""
    initial_size: float = size
    direction: int = 0
    for iteration in range(2, 8):
        stop: bool = False
        previous_size: float = size
        step: float = max(size * 0.05 / iteration, 0.25)
        if size < 1.25:
            peak: float = float(image.color(x, y, 5).max())
            background: float = cast(Stats, image.stats(x, y, 10, "ring")).minimum
            size, direction = small_size_step(size, direction, peak, background)
        else:
            outer_stats: Stats | None = image.stats(x, y, int(8 * size) + 4, "ring")
            inner_stats: Stats | None = image.stats(x, y, int(2.4 * size))
            edge_stats: Stats | None = image.stats(x, y, int(4 * size) - 2, "ring")
            size, direction, stop = large_size_step(
                size, step, direction, outer_stats, inner_stats, edge_stats
            )
        if size_refinement_done(size, previous_size, stop):
            break
    return size, size != initial_size


def _refine_candidate(
    image: Image, x: float, y: float, size: float, coarse: bool, excluded: Sequence[CandidateRecord]
) -> tuple[float, float, float] | None:
    """Alternate position and size refinement, preserving native rollback and limits."""
    previous: Position = (x, y)
    (x, y), moved = refine_position((x, y), size, image.stats)
    size, resized = refine_size(image, x, y, size)
    limit: int = int(size + 8)
    maximum: float = 1.5 * size + 2
    if not coarse:
        maximum = min(maximum, 20)
    converged: bool = False
    for _ in range(limit):
        if not moved and not resized:
            converged = True
            break
        # 96EE5 restores the position preceding the previous move and jumps
        # straight to quantization, including for sizes greater than two.
        if not image.inside(x, y):
            x, y = previous
            converged = True
            break
        previous = (x, y)
        (x, y), moved = refine_position((x, y), size, image.stats)
        if moved or resized:
            size, resized = refine_size(image, x, y, size)
        if size > maximum or covered(x, y, excluded):
            return None
    return refinement_result(x, y, size, converged)


def _accept_shape(image: Image, x: float, y: float, size: float) -> bool:
    """Sample the final disk/ring supports and apply the recovered shape predicate."""
    size_index: int = max(1, int(4 * size))
    support_stats: tuple[Stats | None, ...]
    if size_index < 12:
        inner_stats: Stats | None = image.stats(x, y, max(1, size_index - 2))
        outer_stats: Stats | None = image.stats(x, y, 2 * size_index + 4, "ring")
        support_stats = (inner_stats, outer_stats)
    else:
        support_stats = (
            image.stats(x, y, size_index // 2),
            image.stats(x, y, size_index - 3),
            image.stats(x, y, size_index - 3, "ring"),
            image.stats(x, y, (6 * size_index) // 5 + 4, "ring"),
            image.stats(x, y, 2 * size_index + 4, "ring"),
        )
    return not any(sampled_stats is None for sampled_stats in support_stats) and final_shape(
        size, *cast(tuple[Stats, ...], support_stats)
    )


def candidate(
    image: Image, x: float, y: float, coarse: bool = False, excluded: Sequence[CandidateRecord] = ()
) -> CandidateRecord | None:
    """Qualify one global half-pixel seed and return raw source measurements on success."""
    size: float
    valid_seed, (x, y), size = initial_candidate((x, y), image.bounds, image.sample)
    if not valid_seed:
        return None
    size = shrink_initial(image, x, y, size)
    size_index: int = int(4 * size)
    ring_offset: int = size_index // 8 + 4
    if size_index >= ring_offset:
        inner_stats: Stats | None = image.stats(x, y, size_index - ring_offset, "ring")
        outer_stats: Stats | None = image.stats(x, y, size_index + ring_offset, "ring")
        if (
            inner_stats is None
            or outer_stats is None
            or not preliminary_shape(inner_stats, outer_stats, size_index)
        ):
            return None
    refined: tuple[float, float, float] | None = _refine_candidate(
        image, x, y, size, coarse, excluded
    )
    if refined is None:
        return None
    x, y, size = refined
    x, y = quarter_position(x), quarter_position(y)
    size = quarter_size(size)
    if not _accept_shape(image, x, y, size):
        return None
    if not image.inside(x, y) or (coarse and size <= 4):
        return None
    # 96EF0: sum annular colors at size, size+1, ...; disk brightness.
    ring_count: int = 1 - int((size - 1) * -0.5)
    color: DoubleArray = cast(
        DoubleArray,
        sum(
            (
                image.color(x, y, int(4 * size) + 4 * ring_index, "ring")
                for ring_index in range(ring_count)
            ),
            start=np.zeros(3),
        ),
    )
    # 9712B followed by PSHUFD E1 writes GREEN, not red.
    color[1] = 0.4 * color[0] + 0.3 * (color[1] + color[2])
    color /= color.mean()
    brightness: float = float(image.color(x, y, int(4 * size) + 2).mean())
    return CandidateRecord(x=x, y=y, size=size, rgb=color.tolist(), brightness=brightness, count=1)


def covered(x: float, y: float, records: Sequence[CandidateRecord]) -> bool:
    """Test a global point against existing source exclusion disks."""
    return any(
        (x - candidate_record["x"]) ** 2 + (y - candidate_record["y"]) ** 2
        < candidate_record["size"] ** 2
        for candidate_record in records
    )


def merge_candidate(
    rows: list[CandidateRecord], incoming_record: CandidateRecord, maximum_size: float
) -> float:
    """Merge a fresh source with a bounded backward search and Y-sorted insertion."""
    minimum_y: float = round_to_float32(
        incoming_record["y"] - round_to_float32(max(incoming_record["size"], maximum_size) + 1)
    )
    for existing_record in reversed(rows):
        if existing_record["y"] < minimum_y:
            break
        dx: float = round_to_float32(existing_record["x"] - incoming_record["x"])
        dy: float = round_to_float32(existing_record["y"] - incoming_record["y"])
        if round_to_float32(round_to_float32(dx * dx) + round_to_float32(dy * dy)) < max(
            round_to_float32(existing_record["size"] ** 2),
            round_to_float32(incoming_record["size"] ** 2),
        ):
            count: int = min(max(existing_record["count"], 1) + 1, 255)
            blend_record(existing_record, incoming_record, count, round_to_float32(1 / count))
            return max(maximum_size, existing_record["size"])
    insertion_index: int = len(rows)
    while insertion_index and incoming_record["y"] <= rows[insertion_index - 1]["y"]:
        insertion_index -= 1
    rows.insert(insertion_index, incoming_record)
    return max(maximum_size, incoming_record["size"])
