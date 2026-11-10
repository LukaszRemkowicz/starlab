"""Numerical helpers for weighted star-source analysis.

Float32 compatibility rounding is used where quantized results require it."""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from ..array_types import Position
from .rays import f
from .records import CompiledStats

LUMA = (0.2126390039920807, 0.7151686549186707, 0.07219231128692627)


@dataclass(frozen=True)
class Stats:
    """Weighted source statistics; score is the native mean-minus-standard-deviation value."""

    minimum: float
    mean: float
    maximum: float
    stddev: float
    score: float


def statistics(samples: Iterable[tuple[float, float]]) -> Stats:
    """F1E10: (luminance, kernel weight), valid in-bounds samples only."""
    samples = list(samples)
    total = sum(w for v, w in samples)
    mean = sum(v * w for v, w in samples) / total if total > 0 else 0.0
    std = math.sqrt(sum(w * (v - mean) ** 2 for v, w in samples) / total) if total > 0 else 0.0
    return Stats(
        min([1.0] + [v for v, w in samples]),
        mean,
        max([0.0] + [v for v, w in samples]),
        std,
        mean - std,
    )


def disk_weight(radius: float, distance: float) -> float:
    """C92C0. Radius is table index/4; disk has a one-pixel edge ramp."""
    return min(1.0, max(0.0, radius + 0.6 - distance))


def annulus_weight(radius: float, distance: float) -> float:
    """EBC30. A triangular radial profile with half-width 1.15 pixels."""
    return max(0.0, (1.15 - abs(radius - distance)) / 1.15)


def quarter_position(value: float) -> float:
    """96C78: deliberately NOT standard nearest-quarter rounding."""
    return math.floor(4 * value + 0.25) * 0.25


def quarter_size(value: float) -> float:
    """Quantize native size using truncation after the half-quarter bias."""
    return math.trunc(4 * value + 0.5) * 0.25


def small_size(contrast: float) -> float:
    """97D9D: proposed size when current size <1.25, contrast=(peak-bg)/(1-bg)."""
    for limit, size in [(0.12, 0.25), (0.26, 0.5), (0.38, 0.75), (0.57, 1.0)]:
        if contrast < limit:
            return size
    return 1.25


def preliminary_shape(
    inner: Stats | CompiledStats, outer: Stats | CompiledStats, index: int
) -> bool:
    """98230, after successful samples and image/mask checks; index=trunc(4*s)."""
    if index < math.trunc(index / 8) + 4:
        return True
    # COMISS/JB at983EE accepts unordered 0*infinity for a saturated ring.
    if outer.minimum == 1:
        return True
    scale = 1 / (1 - outer.minimum)
    return (1 - inner.mean) * scale < 0.64 * (1 - outer.mean) * scale or (
        index < 10 and (inner.mean - outer.mean) * scale > 0.04
    )


def final_shape(
    size: float,
    a: Stats | CompiledStats,
    b: Stats | CompiledStats,
    c: Stats | CompiledStats | None = None,
    d: Stats | CompiledStats | None = None,
    e: Stats | CompiledStats | None = None,
) -> bool:
    """98440, successful stats only. A/B are disks for large candidates.

    n=max(1,trunc(4*size)). n<12: A=disk(max(1,n-2)),B=ring(2*n+4).
    n>=12: A=disk(n//2), B=disk(n-3), C=ring(n-3),
    D=ring(floor(1.2*n)+4), E=ring(2*n+4). Arguments in table indices.
    """
    if int(4 * size) < 12:
        return a.score >= b.score and a.score - b.score >= 0.13
    # Large candidates require all five successful supports; both callers enforce this.
    background = min(d.minimum, e.minimum)  # type: ignore[union-attr]
    if background >= 1:
        return False
    scale = 1 / (1 - background)
    aa, bb, cc, dd = [(s.score - background) * scale for s in (a, b, c, d)]  # type: ignore[union-attr]
    return (
        a.maximum >= 0.75
        and aa >= 0.67
        and a.mean >= 0.95 * max(b.maximum, d.maximum, e.maximum)  # type: ignore[union-attr]
        and bb >= 0.62
        and cc - dd >= 0.7 * (aa - cc)
        and aa >= 0.985 * bb
        and bb >= cc
        and 0.9 * cc >= dd
    )


def large_size_step(
    size: float,
    step: float,
    direction: int,
    outer: Stats | CompiledStats | None,
    inner: Stats | CompiledStats | None,
    edge_sample: Stats | CompiledStats | None,
) -> tuple[float, int, bool]:
    """Apply one sampled radius adjustment, including growth-reversal termination."""
    if outer is None or inner is None or edge_sample is None:
        return size, direction, True
    denominator = 1 - outer.minimum
    center = (inner.score - outer.minimum) / denominator if denominator else math.nan
    edge = (edge_sample.score - outer.minimum) / denominator if denominator else math.nan
    if center < 0.7 or edge < 0.4 * center:
        return size - step, -1, False
    if center > 0.97 and edge > 0.98 * center and size + step <= 128:
        if direction == -1:
            return size + step, direction, True
        return size + step, 1, False
    return size, direction, False


def refinement_result(
    x: float, y: float, size: float, converged: bool
) -> tuple[float, float, float] | None:
    """Accept a converged fit or a small source whose bounded refinement expired."""
    if not converged and size > 2:
        return None
    return x, y, size


def refine_position(
    position: Position, size: float, sample: Callable[[float, float, int], Stats | None]
) -> tuple[Position, bool]:
    """97680 hill climb. sample(x,y,index) returns Stats or None.

    Direction order is the native jump table97B3C: E,SE,S,SW,W,NW,N,NE.
    """

    x, y = map(f, position)
    index = max(1, int(size * 4))
    initial = sample(x, y, index)
    if initial is None:
        return position, False
    best = initial.score
    step = max(f(size * 0.125), 0.25)
    changed = False
    q = 0.7071067690849304
    directions = ((1, 0), (q, q), (0, 1), (-q, q), (-1, 0), (-q, -q), (0, -1), (q, -q))
    for _ in range(6):
        selected = None
        for dx, dy in directions:
            stats = sample(f(x + f(dx * step)), f(y + f(dy * step)), index)
            if stats is not None and stats.score > best + 1e-6:
                best = stats.score
                selected = (dx, dy)
        if selected is None:
            break
        dx, dy = selected
        x = f(x + f(dx * step))
        y = f(y + f(dy * step))
        changed = True
        step = max(f(step * f(0.8)), 0.25)
    return (x, y), changed


def normalize_catalog(
    records: Iterable[tuple[Sequence[float], float, float]],
) -> tuple[list[tuple[float, ...]], list[float], float]:
    """3E210 for unprocessed automatic records, all active.

    Inputs: (rgb, size_field_30, brightness_field_38).
    Does not model flags, cancellation or records already marked processed.
    """
    scaled = [tuple(channel * b * s * s / 16384 for channel in rgb) for rgb, s, b in records]
    peak = max((max(rgb) for rgb in scaled), default=0.0)
    multiplier = 0.87 / peak if peak else 0.87
    colors = [tuple(v * multiplier for v in rgb) for rgb in scaled]
    brightness = [sum(c * w for c, w in zip(rgb, LUMA)) for rgb in colors]
    return colors, brightness, max(brightness, default=0.0)
