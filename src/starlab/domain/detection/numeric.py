"""Numerical helpers for weighted star-source analysis.

Float32 compatibility rounding is used where quantized results require it."""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from ..array_types import FloatArray, Position
from .precision import round_to_float32
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
    """Compute weighted luminance statistics from valid in-bounds samples."""
    samples = list(samples)
    total: float = sum(weight for luminance, weight in samples)
    mean: float = (
        sum(luminance * weight for luminance, weight in samples) / total if total > 0 else 0.0
    )
    stddev: float = (
        math.sqrt(sum(weight * (luminance - mean) ** 2 for luminance, weight in samples) / total)
        if total > 0
        else 0.0
    )
    return Stats(
        min([1.0] + [luminance for luminance, weight in samples]),
        mean,
        max([0.0] + [luminance for luminance, weight in samples]),
        stddev,
        mean - stddev,
    )


def disk_weight(radius: float, distance: float) -> float:
    """Weight a disk with a one-pixel edge ramp; radius is table index/4."""
    return min(1.0, max(0.0, radius + 0.6 - distance))


def annulus_weight(radius: float, distance: float) -> float:
    """Weight a triangular annular profile with half-width 1.15 pixels."""
    return max(0.0, (1.15 - abs(radius - distance)) / 1.15)


def quarter_position(value: float) -> float:
    """Quantize position with the recovered asymmetric quarter-pixel bias."""
    return math.floor(4 * value + 0.25) * 0.25


def quarter_size(value: float) -> float:
    """Quantize native size using truncation after the half-quarter bias."""
    return math.trunc(4 * value + 0.5) * 0.25


def small_size(contrast: float) -> float:
    """Select a size below 1.25 pixels from background-normalized contrast."""
    for limit, size in [(0.12, 0.25), (0.26, 0.5), (0.38, 0.75), (0.57, 1.0)]:
        if contrast < limit:
            return size
    return 1.25


def combine_variance_lanes(lanes: FloatArray) -> float:
    """Combine eight float32 variance lanes in the required pairwise rounding order."""
    combined_lanes: FloatArray = lanes[:4] + lanes[4:]
    return round_to_float32(
        round_to_float32(float(combined_lanes[2]) + float(combined_lanes[0]))
        + round_to_float32(float(combined_lanes[3]) + float(combined_lanes[1]))
    )


def small_size_step(
    size: float, direction: int, peak: float, background: float
) -> tuple[float, int]:
    """Select a subpixel radius from contrast and preserve its growth direction."""
    if peak > background:
        proposed_size: float = small_size((peak - background) / (1 - background))
        return (
            proposed_size,
            1 if proposed_size > size else -1 if proposed_size < size else direction,
        )
    return size + 0.25, 1


def preliminary_shape(
    inner: Stats | CompiledStats, outer: Stats | CompiledStats, index: int
) -> bool:
    """Screen inner/outer contrast at table index trunc(4*size)."""
    if index < math.trunc(index / 8) + 4:
        return True
    # COMISS/JB at983EE accepts unordered 0*infinity for a saturated ring.
    if outer.minimum == 1:
        return True
    scale: float = 1 / (1 - outer.minimum)
    return (1 - inner.mean) * scale < 0.64 * (1 - outer.mean) * scale or (
        index < 10 and (inner.mean - outer.mean) * scale > 0.04
    )


def final_shape(
    size: float,
    inner_disk: Stats | CompiledStats,
    outer_disk: Stats | CompiledStats,
    inner_ring: Stats | CompiledStats | None = None,
    middle_ring: Stats | CompiledStats | None = None,
    outer_ring: Stats | CompiledStats | None = None,
) -> bool:
    """Validate source shape from successful disk and ring statistics.

    n=max(1,trunc(4*size)). n<12: A=disk(max(1,n-2)),B=ring(2*n+4).
    n>=12: A=disk(n//2), B=disk(n-3), C=ring(n-3),
    D=ring(floor(1.2*n)+4), E=ring(2*n+4). Arguments in table indices.
    """
    if int(4 * size) < 12:
        return inner_disk.score >= outer_disk.score and inner_disk.score - outer_disk.score >= 0.13
    # Large candidates require all five successful supports; both callers enforce this.
    background: float = min(middle_ring.minimum, outer_ring.minimum)  # type: ignore[union-attr]
    if background >= 1:
        return False
    scale: float = 1 / (1 - background)
    inner_disk_score, outer_disk_score, inner_ring_score, middle_ring_score = [
        (sampled_stats.score - background) * scale  # type: ignore[union-attr]
        for sampled_stats in (inner_disk, outer_disk, inner_ring, middle_ring)
    ]
    return (
        inner_disk.maximum >= 0.75
        and inner_disk_score >= 0.67
        and inner_disk.mean
        >= 0.95 * max(outer_disk.maximum, middle_ring.maximum, outer_ring.maximum)  # type: ignore[union-attr]
        and outer_disk_score >= 0.62
        and inner_ring_score - middle_ring_score >= 0.7 * (inner_disk_score - inner_ring_score)
        and inner_disk_score >= 0.985 * outer_disk_score
        and outer_disk_score >= inner_ring_score
        and 0.9 * inner_ring_score >= middle_ring_score
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
    denominator: float = 1 - outer.minimum
    center: float = (inner.score - outer.minimum) / denominator if denominator else math.nan
    edge: float = (edge_sample.score - outer.minimum) / denominator if denominator else math.nan
    if center < 0.7 or edge < 0.4 * center:
        return size - step, -1, False
    if center > 0.97 and edge > 0.98 * center and size + step <= 128:
        if direction == -1:
            return size + step, direction, True
        return size + step, 1, False
    return size, direction, False


def size_refinement_done(size: float, previous_size: float, stop: bool) -> bool:
    """Stop radius refinement on a sampling limit or an unchanged size."""
    return stop or size == previous_size


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
    """Hill-climb weighted luminance; sample(x,y,index) returns Stats or None.

    Direction order is the native jump table97B3C: E,SE,S,SW,W,NW,N,NE.
    """

    y: float
    x: float
    x, y = map(round_to_float32, position)
    index: int = max(1, int(size * 4))
    initial: Stats | None = sample(x, y, index)
    if initial is None:
        return position, False
    best: float = initial.score
    step: float = max(round_to_float32(size * 0.125), 0.25)
    changed: bool = False
    diagonal_step: float = 0.7071067690849304
    directions: tuple[Position, ...] = (
        (1, 0),
        (diagonal_step, diagonal_step),
        (0, 1),
        (-diagonal_step, diagonal_step),
        (-1, 0),
        (-diagonal_step, -diagonal_step),
        (0, -1),
        (diagonal_step, -diagonal_step),
    )
    for _ in range(6):
        selected: Position | None = None
        for dx, dy in directions:
            stats: Stats | None = sample(
                round_to_float32(x + round_to_float32(dx * step)),
                round_to_float32(y + round_to_float32(dy * step)),
                index,
            )
            if stats is not None and stats.score > best + 1e-6:
                best = stats.score
                selected = (dx, dy)
        if selected is None:
            break
        dx, dy = selected
        x = round_to_float32(x + round_to_float32(dx * step))
        y = round_to_float32(y + round_to_float32(dy * step))
        changed = True
        step = max(round_to_float32(step * round_to_float32(0.8)), 0.25)
    return (x, y), changed


def normalize_catalog(
    records: Iterable[tuple[Sequence[float], float, float]],
) -> tuple[list[tuple[float, ...]], list[float], float]:
    """Normalize active, unprocessed source colors and luminance globally.

    Inputs: (rgb, size_field_30, brightness_field_38).
    Does not model flags, cancellation or records already marked processed.
    """
    scaled: list[tuple[float, ...]] = [
        tuple(channel * source_brightness * source_size * source_size / 16384 for channel in rgb)
        for rgb, source_size, source_brightness in records
    ]
    peak: float = max((max(rgb) for rgb in scaled), default=0.0)
    multiplier: float = 0.87 / peak if peak else 0.87
    colors: list[tuple[float, ...]] = [
        tuple(channel * multiplier for channel in rgb) for rgb in scaled
    ]
    brightness: list[float] = [
        sum(channel * weight for channel, weight in zip(rgb, LUMA)) for rgb in colors
    ]
    return colors, brightness, max(brightness, default=0.0)
