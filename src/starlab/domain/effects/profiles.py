"""Verified native CPU profiles; all geometry remains in full-image pixel units."""

import math
from collections import OrderedDict
from threading import RLock

import numpy as np

from .settings import EffectParameters

SECONDARY_RAYS = (
    (40, 27),
    (75, 9),
    (110, 7),
    (160, 43),
    (214, 22),
    (240, 7),
    (270, 9),
    (303, 11),
    (326, 9),
    (340, 14),
    (388, 25),
    (426, 17),
    (454, 8),
    (489, 30),
    (540, 22),
    (610, 18),
    (670, 31),
    (720, 22),
    (790, 31),
    (840, 10),
    (890, 33),
    (940, 15),
    (986, 11),
    (1013, 36),
    (1040, 21),
    (1090, 34),
    (1120, 9),
    (1176, 14),
    (1224, 29),
    (1270, 21),
    (1330, 13),
    (1350, 12),
    (1410, 17),
    (1460, 29),
    (1492, 7),
    (1520, 11),
    (1522, 9),
    (1580, 22),
    (1630, 10),
    (1642, 8),
    (1676, 21),
    (1721, 6),
    (1749, 13),
    (1780, 14),
    (1830, 11),
    (1890, 23),
    (1953, 9),
    (1990, 15),
    (2033, 7),
    (2070, 21),
    (2150, 15),
    (2200, 36),
    (2240, 14),
    (2273, 5),
    (2310, 6),
    (2351, 20),
    (2380, 22),
    (2444, 13),
    (2500, 27),
    (2560, 9),
    (2604, 16),
    (2670, 32),
    (2721, 14),
    (2751, 18),
    (2770, 8),
    (2829, 19),
    (2860, 6),
    (2910, 16),
    (2960, 22),
    (3020, 8),
    (3070, 13),
    (3103, 24),
    (3130, 6),
    (3150, 14),
    (3200, 5),
    (3232, 19),
    (3310, 8),
    (3328, 19),
    (3380, 11),
    (3404, 18),
    (3470, 22),
    (3520, 10),
    (3558, 21),
    (3584, 10),
)


class ArrayCache:
    """LRU bounded by array bytes, shared safely by cancelling preview workers."""

    def __init__(self, megabytes):
        self.limit = megabytes * 1024 * 1024
        self.max_entries = 4096
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            item = self.entries.get(key)
            if item is not None:
                self.entries.move_to_end(key)
                return item[0]
        return None

    def put(self, key, value):
        arrays = value if isinstance(value, tuple) else (value,)
        size = sum(a.nbytes for a in arrays)
        if size > self.limit:
            return value
        for array in arrays:
            array.flags.writeable = False
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous is not None:
                self.bytes -= previous[1]
            while self.entries and (
                self.bytes + size > self.limit or len(self.entries) >= self.max_entries
            ):
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
            self.entries[key] = value, size
            self.bytes += size
        return value


PROFILE_CACHE = ArrayCache(64)


def image_diagonal(width, height):
    return int(np.float32(math.hypot(width, height)) + np.float32(1))


def length_envelope(length: float) -> int:
    """Round CPU ray support upward so nearby length edits can reuse geometry."""
    return max(10, math.ceil(length / 10) * 10)


def ray_coefficients(p: EffectParameters, scale: float) -> tuple[float, float]:
    """Return transverse sharpness and radial widening in full-image pixel units."""
    sharp = (
        (p.sharpness / 90) ** 1.4
        if p.sharpness <= 90
        else 1 + 0.251416 * ((p.sharpness - 90) / 10) ** 1.4 / scale
    )
    coefficient = (4 * sharp + 0.08) * scale
    widening = 0.15 * (p.spread / 100) ** 2 * coefficient
    return coefficient, widening


def geometry_key(kind, p):
    if kind in ("primary", "secondary"):
        # Length changes reuse transverse AA. Bounded support envelopes
        # avoid allocating a whole-image maximum-length kernel for short rays.
        envelope = length_envelope(p.length)
        return (
            kind,
            envelope,
            p.sharpness,
            p.spread,
            p.angle,
            p.points if kind == "primary" else 84,
        )
    if kind == "soft":
        return kind, p.size, p.falloff
    return kind, p.size, p.inner_width, p.outer_width, p.inner_falloff, p.outer_falloff


def gain(kind, p):
    exponent = {"primary": 2, "secondary": 1.2, "soft": 3, "ring": 1.8}[kind]
    factor = {"primary": 128, "secondary": 128 * 0.6, "soft": 1, "ring": 1.14}[kind]
    return factor * (p.intensity / 100) ** exponent


def source_colors(colors, hue, saturation):
    colors = np.asarray(colors, dtype=np.float32).copy()
    green = colors[:, 1].copy()
    colors += green[:, None] * np.array([-hue / 1500, hue / 3000, -hue / 1000])
    np.maximum(colors, 0, out=colors)
    factor = saturation / 6 if saturation > 0 else saturation / 150
    colors += factor * (colors - (colors.sum(axis=1, keepdims=True) - colors) / 2)
    return np.maximum(colors, 0)


def directions(kind, p):
    angle = round(p.angle * 10)
    match kind:
        case "primary":
            return tuple(
                ((int(i * 3600 / p.points + 0.5) + angle) % 3600, 1.0) for i in range(p.points)
            )
        case "secondary":
            return tuple(((a + angle) % 3600, weight / 100) for a, weight in SECONDARY_RAYS)
        case _:
            return ((0, 1.0),)


def sine(angle):
    angle %= 3600
    return float(np.float32(math.sin(math.radians((angle % 1800) / 10)))) * (
        1 if angle < 1800 else -1
    )


def radial_extent(kind, p, diagonal):
    d = max(diagonal, 1414)
    match kind:
        case "primary" | "secondary":
            return min((0.000015 ** (-(p.length + 25) / 190) - 1) * d / 800, diagonal + 2)
        case "soft":
            radius = 1 + (d // 3) * (p.size / 100) ** 3
            power = 0.6 + 0.124 * p.falloff
            return radius * (1 - (0.00025 / 5.7) ** (1 / power))
        case _:
            peak = (1 + 0.01 * (p.size + 25) ** 2) * d / 3768
            return peak * (1.03 + 0.02 * p.outer_width)


def _coordinates(extent, sampling, residue):
    start = math.ceil((-extent - residue) / sampling) * sampling + residue
    return np.arange(start, extent + 1, sampling, dtype=np.int32)


def scene_batch_size(kind, p, diagonal, sampling):
    """Bound estimated sparse contributions, including long full-size exports."""
    if kind in ("primary", "secondary"):
        scale = 800 / max(diagonal, 1414)
        envelope = length_envelope(p.length)
        cutoff = min((0.000015 ** (-(envelope + 25) / 190) - 1) / scale, diagonal + 2)
        coefficient, _ = ray_coefficients(p, scale)
        band = (1 + cutoff * 0.15 * (p.spread / 100) ** 2 * coefficient) / (
            coefficient / math.sqrt(2)
        ) + 0.875
        per_source = (cutoff / sampling + 2) * (2 * band / sampling + 2) * len(directions(kind, p))
    else:
        cutoff = radial_extent(kind, p, diagonal)
        per_source = (2 * cutoff / sampling + 2) ** 2
    # Four arrays × four bytes; approximately16MiB per scene batch. A single
    # extreme ray can exceed this estimate, but catalogs cannot grow the batch.
    return max(1, int(1048576 / max(per_source, 1)))


def transverse_profile(cross, coefficient, denominator):
    """The native 15 × 7 quadrature, vectorized in bounded blocks."""
    result = np.maximum(1 - np.abs(cross) * coefficient / denominator, 0) ** 3
    near = np.flatnonzero(cross <= 1.5)
    offsets = np.arange(15) / 14 - 0.5
    shifts = np.arange(-3, 4) / 8
    for start in range(0, len(near), 4096):
        index = near[start : start + 4096]
        distance = cross[index, None]
        ratio = coefficient / denominator[index, None]
        result[index] = (np.maximum(1 - np.abs(distance + offsets) * ratio, 0) ** 3).mean(axis=1)
        index = index[cross[index] < 1.5]
        if not len(index):
            continue
        shifted = cross[index, None] + shifts
        ratio = coefficient / denominator[index, None]
        point = np.maximum(1 - np.abs(shifted) * ratio, 0) ** 3
        samples = np.maximum(1 - np.abs(shifted[..., None] + offsets) * ratio[..., None], 0) ** 3
        result[index] = np.where(shifted <= 1.5, samples.mean(axis=2), point).mean(axis=1)
    return result


def sampled_ray_scene(width, height, positions, p, diagonal, sampling, angle):
    """Batch many sampled sources through one exact ray rasterizer.

    Sampling creates many residue classes. Evaluating them together avoids
    thousands of tiny profile allocations during an angle or sharpness edit.
    """
    scale = 800 / max(diagonal, 1414)
    envelope = length_envelope(p.length)
    cutoff = min((0.000015 ** (-(envelope + 25) / 190) - 1) / scale, diagonal + 2)
    extent = math.ceil(cutoff + 1)
    coefficient, widening = ray_coefficients(p, scale)
    sin, cos = sine(angle), sine(angle + 900)
    horizontal = abs(sin) >= abs(cos)
    axis = 0 if horizontal else 1
    cross_coefficient = coefficient * (abs(sin) if horizontal else abs(cos))
    slope = -cos / sin if horizontal else -sin / cos
    raw = np.asarray(positions) + 0.5
    origins = np.floor(raw).astype(np.int32)
    centers = np.floor(4 * (raw - origins)) / 4 - 0.5
    residues = (-origins) % sampling
    grid = np.arange(-extent // sampling * sampling, extent + sampling, sampling)
    along = grid[None, :] + residues[:, axis, None]
    positive = sin > 0 if horizontal else cos < 0
    valid = (along >= 0) if positive else (along < 0)
    valid &= np.abs(along) <= extent
    world_along = along + origins[:, axis, None]
    valid &= (world_along >= 0) & (world_along < (width if horizontal else height))
    source, column = np.nonzero(valid)
    along = along[source, column]
    middle = centers[source, 1 - axis] + (along - centers[source, axis]) * slope
    band = (1 + cutoff * widening) / cross_coefficient + 0.875
    low = np.ceil((middle - band - residues[source, 1 - axis]) / sampling).astype(np.int64)
    high = np.floor((middle + band - residues[source, 1 - axis]) / sampling).astype(np.int64)
    low = np.maximum(
        low,
        np.ceil((-origins[source, 1 - axis] - residues[source, 1 - axis]) / sampling).astype(
            np.int64
        ),
    )
    high = np.minimum(
        high,
        np.floor(
            (
                (height if horizontal else width)
                - 1
                - origins[source, 1 - axis]
                - residues[source, 1 - axis]
            )
            / sampling
        ).astype(np.int64),
    )
    counts = np.maximum(high - low + 1, 0)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    across = (np.repeat(low, counts) + np.arange(int(counts.sum())) - offsets) * sampling
    source = np.repeat(source, counts)
    across += residues[source, 1 - axis]
    along = np.repeat(along, counts)
    xx, yy = (along, across) if horizontal else (across, along)
    world_x, world_y = xx + origins[source, 0], yy + origins[source, 1]
    valid = (world_x >= 0) & (world_y >= 0) & (world_x < width) & (world_y < height)
    xx, yy, source, world_x, world_y = (v[valid] for v in (xx, yy, source, world_x, world_y))
    dx, dy = xx - centers[source, 0], yy - centers[source, 1]
    radius = np.hypot(dx, dy)
    cross = np.abs(dy - dx * slope) if horizontal else np.abs(dx - dy * slope)
    denominator = 1 + radius * widening
    valid = (radius <= cutoff) & (cross < denominator / cross_coefficient + 0.875)
    radius, cross, denominator, source, world_x, world_y = (
        v[valid] for v in (radius, cross, denominator, source, world_x, world_y)
    )
    values = transverse_profile(cross, cross_coefficient, denominator)
    index = (world_y // sampling) * ((width + sampling - 1) // sampling) + world_x // sampling
    valid = values > 0
    return tuple(
        np.ascontiguousarray(v[valid], dtype=dtype)
        for v, dtype in (
            (index, np.int32),
            (values, np.float32),
            (radius, np.float32),
            (source, np.int32),
        )
    )


def sparse_profile(kind, p, diagonal, phase, sampling, residue, angle=0, bounds=None):
    """One ray's integer offsets, opacity, radius; AA always uses full pixels."""
    key = geometry_key(kind, p), diagonal, phase, sampling, residue, angle, bounds
    cached = PROFILE_CACHE.get(key)
    if cached is not None:
        return cached
    cutoff = radial_extent(kind, p, diagonal)
    if kind in ("primary", "secondary"):
        envelope = length_envelope(p.length)
        cutoff = min(
            (0.000015 ** (-(envelope + 25) / 190) - 1) * max(diagonal, 1414) / 800, diagonal + 2
        )
    extent = math.ceil(cutoff + 1)
    center_x, center_y = phase[0] / 4 - 0.5, phase[1] / 4 - 0.5
    if kind in ("primary", "secondary"):
        scale = 800 / max(diagonal, 1414)
        coefficient, widening = ray_coefficients(p, scale)
        sin, cos = sine(angle), sine(angle + 900)
        horizontal = abs(sin) >= abs(cos)
        dominant = abs(sin) if horizontal else abs(cos)
        cross_coefficient = coefficient * dominant
        slope = -cos / sin if horizontal else -sin / cos
        axis = 0 if horizontal else 1
        centers = (center_x, center_y)
        along = _coordinates(extent, sampling, residue[axis])
        positive = sin > 0 if horizontal else cos < 0
        along = along[along >= 0] if positive else along[along < 0]
        middle = centers[1 - axis] + (along - centers[axis]) * slope
        band = (1 + cutoff * widening) / cross_coefficient + 0.875
        low = np.ceil((middle - band - residue[1 - axis]) / sampling).astype(np.int64)
        high = np.floor((middle + band - residue[1 - axis]) / sampling).astype(np.int64)
        counts = np.maximum(high - low + 1, 0)
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        across = (
            np.repeat(low, counts) + np.arange(int(counts.sum())) - offsets
        ) * sampling + residue[1 - axis]
        along = np.repeat(along, counts)
        xx, yy = (along, across) if horizontal else (across, along)
        dx, dy = xx - center_x, yy - center_y
        radius = np.hypot(dx, dy)
        cross = np.abs(dy - dx * slope) if horizontal else np.abs(dx - dy * slope)
        denominator = 1 + radius * widening
        valid = (radius <= cutoff) & (cross < denominator / cross_coefficient + 0.875)
        xx, yy, radius, cross, denominator = (
            v[valid] for v in (xx, yy, radius, cross, denominator)
        )

        values = transverse_profile(cross, cross_coefficient, denominator)
    else:
        x, y = (
            _coordinates(extent, sampling, residue[0]),
            _coordinates(extent, sampling, residue[1]),
        )
        if bounds is not None:
            x = x[(x >= bounds[0]) & (x < bounds[2])]
            y = y[(y >= bounds[1]) & (y < bounds[3])]
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx.ravel(), yy.ravel()
        radius = np.hypot(xx - center_x, yy - center_y)
        if kind == "soft":
            size = 1 + (max(diagonal, 1414) // 3) * (p.size / 100) ** 3
            values = np.maximum(
                5.7 * np.maximum(1 - radius / size, 0) ** (0.6 + 0.124 * p.falloff) - 0.00025, 0
            )
        else:
            peak = (1 + 0.01 * (p.size + 25) ** 2) * max(diagonal, 1414) / 3768
            inner = peak * (1 - (0.03 + 0.0097 * p.inner_width))
            outer = peak * (1.03 + 0.02 * p.outer_width)
            values = np.where(
                radius < peak,
                np.clip((radius - inner) / (peak - inner), 0, 1) ** (0.08 * (p.inner_falloff + 10)),
                np.clip((outer - radius) / (outer - peak), 0, 1) ** (0.08 * (p.outer_falloff + 10)),
            )
    valid = values > 0
    arrays = tuple(
        np.ascontiguousarray(v[valid], dtype=dtype)
        for v, dtype in ((xx, np.int32), (yy, np.int32), (values, np.float32), (radius, np.float32))
    )
    return PROFILE_CACHE.put(key, arrays)
