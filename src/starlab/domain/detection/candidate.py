"""Candidate qualification, subpixel positioning and source-size estimation.

See docs/architecture.md for the detection pipeline and its limitations."""

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal, cast

import numpy as np

from ..array_types import (
    RGBA,
    BoolArray,
    CandidateRecord,
    DoubleArray,
    FloatArray,
    IntArray,
)
from ..jit import njit
from .catalog import blend_record
from .numeric import (
    LUMA,
    Stats,
    final_shape,
    large_size_step,
    preliminary_shape,
    quarter_position,
    quarter_size,
    refine_position,
    refinement_result,
    small_size,
)
from .rays import f, initial_candidate

type KernelKind = Literal["disk", "ring"]


@njit(cache=True, nogil=True)
def _luminance(rgb: FloatArray) -> FloatArray:
    """Return H×W float32 luminance from an H×W×3 decoded detector tile."""
    result = np.empty(rgb.shape[:2], np.float32)
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            result[y, x] = (
                rgb[y, x, 0] * np.float32(LUMA[0]) + rgb[y, x, 2] * np.float32(LUMA[2])
            ) + rgb[y, x, 1] * np.float32(LUMA[1])
    return result


class Image:
    """Decoded RGB tile with global origin, allowed mask, and native sampling caches."""

    def __init__(
        self, rgb: FloatArray, origin: tuple[int, int] = (0, 0), allowed: BoolArray | None = None
    ) -> None:
        """Own decoded RGB/luminance sampling state and bounded per-tile caches."""
        self.rgb = np.asarray(rgb, dtype=np.float32)
        self.h: int
        self.w: int
        self.h, self.w = self.rgb.shape[:2]
        self.ox, self.oy = origin
        self.allowed = allowed
        self.bounds = (self.ox, self.oy, self.ox + self.w, self.oy + self.h)
        self._stats_cached = lru_cache(maxsize=8192)(self._stats)
        self._color_cached = lru_cache(maxsize=4096)(self._color)
        self.lum = _luminance(self.rgb)

    def inside(self, x: float, y: float) -> bool:
        """Test global coordinates against this tile’s half-open bounds."""
        return self.ox <= x < self.ox + self.w and self.oy <= y < self.oy + self.h

    def sample(self, x: float, y: float) -> RGBA:
        """Sample by native coordinate truncation; excluded/outside points have zero alpha."""
        ix, iy = int(x) - self.ox, int(y) - self.oy
        if not (0 <= ix < self.w and 0 <= iy < self.h):
            return (0, 0, 0, 0)
        if self.allowed is not None and not self.allowed[iy, ix]:
            return (0, 0, 0, 0)
        return cast(RGBA, (*map(float, self.rgb[iy, ix]), 1))

    @staticmethod
    @lru_cache(maxsize=128)
    def kernel(
        kind: KernelKind, index: int, qx: int, qy: int
    ) -> tuple[IntArray, IntArray, FloatArray]:
        """Return sparse X/Y offsets and float32 weights for quarter-phase support."""
        r = index * 0.25
        cx = qx * 0.25 - 0.5
        cy = qy * 0.25 - 0.5
        # Native square enumeration uses X phase for BOTH bounds. Annular
        # supports are deliberately clipped; an ideal unbounded ring differs.
        if kind == "disk":
            extent = int(f(f(r + f(0.6)) + cx))
            lower, upper = -extent - 1, extent + 2
        else:
            extent = int(f(f(r + cx) + f(1.65)))
            lower, upper = -extent, extent + 1
        yy, xx = np.mgrid[lower:upper, lower:upper]
        dx = (xx - cx).astype(np.float32)
        dy = (yy - cy).astype(np.float32)
        distance = np.sqrt(dx * dx + dy * dy)
        weights = (
            np.clip(np.float32(r + np.float32(0.6)) - distance, 0, 1)
            if kind == "disk"
            else np.maximum(
                0,
                (np.float32(1.15) - np.abs(np.float32(r) - distance))
                * np.float32(0.8695652484893799),
            )
        )
        keep = weights > 0
        return xx[keep], yy[keep], weights[keep].astype(np.float32)

    def values(
        self, x: float, y: float, index: int, kind: KernelKind
    ) -> tuple[IntArray, IntArray, FloatArray] | None:
        """Clip sparse support to allowed tile pixels; invalid table indices return None."""
        if not 0 <= index < 1056:
            return None
        ix, iy = math.floor(x), math.floor(y)
        dx, dy, weight = self.kernel(kind, index, int(4 * (x - ix)), int(4 * (y - iy)))
        xx, yy = dx + ix - self.ox, dy + iy - self.oy
        valid = (xx >= 0) & (yy >= 0) & (xx < self.w) & (yy < self.h)
        if self.allowed is not None:
            indices = np.flatnonzero(valid)
            valid[indices] &= self.allowed[yy[indices], xx[indices]]
        return xx[valid], yy[valid], weight[valid]

    def stats(self, x: float, y: float, index: int, kind: KernelKind = "disk") -> Stats | None:
        """Read cached weighted statistics at floor-quantized quarter coordinates."""
        return self._stats_cached(math.floor(x * 4) * 0.25, math.floor(y * 4) * 0.25, index, kind)

    def _stats(self, x: float, y: float, index: int, kind: KernelKind = "disk") -> Stats | None:
        """Compute native weighted statistics, including the eight-lane variance reduction."""
        values = self.values(x, y, index, kind)
        if values is None:
            return None
        xx, yy, weight = values
        v = self.lum[yy, xx]
        total = float(np.cumsum(weight, dtype=np.float32)[-1]) if len(weight) else 0
        if not total:
            return Stats(1, 0, 0, 0, 0)
        mean = f(float(np.cumsum(v * weight, dtype=np.float32)[-1]) / total)
        delta = v - np.float32(mean)
        variance = delta * delta * weight

        def native_sum(values: FloatArray) -> float:
            """Reduce eight float32 lanes in native combination order, then accumulate the tail."""
            count = len(values) // 8 * 8
            if count:
                lanes = np.cumsum(values[:count].reshape(-1, 8), axis=0, dtype=np.float32)[-1]
                combined = lanes[:4] + lanes[4:]
                result = f(
                    f(float(combined[2]) + float(combined[0]))
                    + f(float(combined[3]) + float(combined[1]))
                )
            else:
                result = 0.0
            for value in values[count:]:
                result = f(result + float(value))
            return result

        std = f(math.sqrt(f(native_sum(variance) / native_sum(weight))))
        return Stats(min(1, float(v.min())), mean, max(0, float(v.max())), std, f(mean - std))

    def color(
        self, x: float, y: float, index: int, kind: KernelKind = "disk"
    ) -> FloatArray | DoubleArray:
        """Read cached weighted RGB at floor-quantized quarter coordinates."""
        return self._color_cached(math.floor(x * 4) * 0.25, math.floor(y * 4) * 0.25, index, kind)

    def _color(
        self, x: float, y: float, index: int, kind: KernelKind = "disk"
    ) -> FloatArray | DoubleArray:
        """Compute sparse weighted RGB; empty or invalid supports return three zeroes."""
        values = self.values(x, y, index, kind)
        if values is None:
            return np.zeros(3)
        xx, yy, weight = values
        if weight.sum() <= 0:
            return np.zeros(3)
        result: FloatArray = (
            np.cumsum(self.rgb[yy, xx] * weight[:, None], axis=0, dtype=np.float32)[-1]
            / np.cumsum(weight, dtype=np.float32)[-1]
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
    index = int(size * 1.32)
    if index > 8:
        a = float(image.color(x, y, index, "ring") @ LUMA)
        b = float(image.color(x, y, int(0.4 * size)) @ LUMA)
        ratio = a / b if b else (math.nan if not a else math.inf)
        threshold = 0.475 * (a + b) * ratio
        lower = int(2 * size + 0.125)
    elif size >= 2:
        threshold = 0.9 * float(image.color(x, y, 4) @ LUMA)
        lower = 6
    else:
        return size
    index = int(size * 4 + 0.125)
    while index > lower and float(image.color(x, y, index, "ring") @ LUMA) < threshold:
        index -= 1
    return 0.25 * (index + 2)


def refine_size(image: Image, x: float, y: float, size: float) -> tuple[float, bool]:
    """Refine radius in native step order and report whether it changed."""
    original = size
    direction = 0
    for iteration in range(2, 8):
        old = size
        step = max(size * 0.05 / iteration, 0.25)
        if size < 1.25:
            peak = float(image.color(x, y, 5).max())
            background = cast(Stats, image.stats(x, y, 10, "ring")).minimum
            if peak > background:
                size = small_size((peak - background) / (1 - background))
                direction = 1 if size > old else -1 if size < old else direction
            else:
                size += 0.25
                direction = 1
        else:
            a = image.stats(x, y, int(8 * size) + 4, "ring")
            b = image.stats(x, y, int(2.4 * size))
            c = image.stats(x, y, int(4 * size) - 2, "ring")
            size, direction, stop = large_size_step(size, step, direction, a, b, c)
            if stop:
                break
        if size == old:
            break
    return size, size != original


def _refine_candidate(
    image: Image, x: float, y: float, size: float, coarse: bool, excluded: Sequence[CandidateRecord]
) -> tuple[float, float, float] | None:
    """Alternate position and size refinement, preserving native rollback and limits."""
    previous = (x, y)
    (x, y), moved = refine_position((x, y), size, image.stats)
    size, resized = refine_size(image, x, y, size)
    limit = int(size + 8)
    maximum = 1.5 * size + 2
    if not coarse:
        maximum = min(maximum, 20)
    converged = False
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
    n = max(1, int(4 * size))
    args: tuple[Stats | None, ...]
    if n < 12:
        a = image.stats(x, y, max(1, n - 2))
        b = image.stats(x, y, 2 * n + 4, "ring")
        args = (a, b)
    else:
        args = (
            image.stats(x, y, n // 2),
            image.stats(x, y, n - 3),
            image.stats(x, y, n - 3, "ring"),
            image.stats(x, y, (6 * n) // 5 + 4, "ring"),
            image.stats(x, y, 2 * n + 4, "ring"),
        )
    return not any(v is None for v in args) and final_shape(size, *cast(tuple[Stats, ...], args))


def candidate(
    image: Image, x: float, y: float, coarse: bool = False, excluded: Sequence[CandidateRecord] = ()
) -> CandidateRecord | None:
    """Qualify one global half-pixel seed and return raw source measurements on success."""
    ok, (x, y), size = initial_candidate((x, y), image.bounds, image.sample)
    if not ok:
        return None
    size = shrink_initial(image, x, y, size)
    n = int(4 * size)
    delta = n // 8 + 4
    if n >= delta:
        a = image.stats(x, y, n - delta, "ring")
        b = image.stats(x, y, n + delta, "ring")
        if a is None or b is None or not preliminary_shape(a, b, n):
            return None
    refined = _refine_candidate(image, x, y, size, coarse, excluded)
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
    count = 1 - int((size - 1) * -0.5)
    color = cast(
        DoubleArray,
        sum(
            (image.color(x, y, int(4 * size) + 4 * i, "ring") for i in range(count)),
            start=np.zeros(3),
        ),
    )
    # 9712B followed by PSHUFD E1 writes GREEN, not red.
    color[1] = 0.4 * color[0] + 0.3 * (color[1] + color[2])
    color /= color.mean()
    brightness = float(image.color(x, y, int(4 * size) + 2).mean())
    return CandidateRecord(x=x, y=y, size=size, rgb=color.tolist(), brightness=brightness, count=1)


def covered(x: float, y: float, records: Sequence[CandidateRecord]) -> bool:
    """Test a global point against existing source exclusion disks."""
    return any((x - r["x"]) ** 2 + (y - r["y"]) ** 2 < r["size"] ** 2 for r in records)


def merge_candidate(
    rows: list[CandidateRecord], new: CandidateRecord, maximum_size: float
) -> float:
    """3E5E0 fresh automatic records: y insertion, bounded backward search."""
    threshold = f(new["y"] - f(max(new["size"], maximum_size) + 1))
    for old in reversed(rows):
        if old["y"] < threshold:
            break
        dx = f(old["x"] - new["x"])
        dy = f(old["y"] - new["y"])
        if f(f(dx * dx) + f(dy * dy)) < max(f(old["size"] ** 2), f(new["size"] ** 2)):
            count = min(max(old["count"], 1) + 1, 255)
            blend_record(old, new, count, f(1 / count))
            return max(maximum_size, old["size"])
    index = len(rows)
    while index and new["y"] <= rows[index - 1]["y"]:
        index -= 1
    rows.insert(index, new)
    return max(maximum_size, new["size"])
