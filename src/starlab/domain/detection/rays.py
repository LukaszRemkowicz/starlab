"""Recovered RVA 98880 / 3DA00 / 97190; independent research reference.

Input sampler supplies RGBA float32 at the exact requested float position.
This module does not implement that callback (interpolation/image access).
All finite RGB inputs, finite bounds, and native float32 operation ordering.
"""

import math
import struct
from collections.abc import Callable, Iterable, Sequence
from typing import cast

from ..array_types import RGBA, Bounds, Position

DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
LUMA = (0.2126390039920807, 0.7151686549186707, 0.07219231128692627)


def f(x: float) -> float:
    """Round one intermediate through IEEE float32 and return its Python scalar."""
    return cast(float, struct.unpack("<f", struct.pack("<f", x))[0])


def add(a: float, b: float) -> float:
    """Add with one float32 rounding step."""
    return f(a + b)


def sub(a: float, b: float) -> float:
    """Subtract with one float32 rounding step."""
    return f(a - b)


def mul(a: float, b: float) -> float:
    """Multiply with one float32 rounding step."""
    return f(a * b)


def div(a: float, b: float) -> float:
    """Divide with native zero/NaN handling and float32 rounding."""
    if b == 0:
        return math.nan if a == 0 else math.copysign(math.inf, a * b)
    return f(a / b)


def total(values: Iterable[float]) -> float:
    """Accumulate in input order, rounding each addition to float32."""
    value = 0.0
    for item in values:
        value = add(value, item)
    return value


def luma(rgb: Sequence[float]) -> float:
    """Combine the first three channels in the recovered B/R/G operation order."""
    return add(add(mul(rgb[2], LUMA[2]), mul(rgb[0], LUMA[0])), mul(rgb[1], LUMA[1]))


def inside(position: Position, bounds: Bounds) -> bool:
    """Test global coordinates against half-open image bounds."""
    x, y = position
    left, top, right, bottom = bounds
    return x >= left and y >= top and x < right and y < bottom


def _extend_ray(
    position: tuple[float, float],
    bounds: Bounds,
    sample: Callable[[float, float], tuple[float, ...]],
    center: tuple[float, ...],
    dx: int,
    dy: int,
    lum: list[float],
    distance: list[float],
    needed: int,
) -> bool:
    """Append reference ray samples through needed, retaining native float32 order."""
    while len(lum) <= needed:
        i = len(lum)
        p = (add(position[0], i * dx), add(position[1], i * dy))
        if not inside(p, bounds):
            return False
        color = tuple(map(f, sample(*p)))
        lum.append(luma(color))
        differences = [mul(sub(a, b), sub(a, b)) for a, b in zip(center[:3], color[:3])]
        distance.append(f(math.sqrt(add(add(differences[1], differences[0]), differences[2]))))
    return True


def initial_extents(
    position: Position, bounds: Bounds, sample: Callable[[float, float], RGBA]
) -> tuple[bool, list[float]]:
    """98880: returns (success, 8 radii); unset radii retain caller's -1.

    The first ray reaching its limit rejects the entire candidate. Subsequent
    rays reaching their limit remain -1. Any ray with sustained outward rise
    rejects the entire candidate. Out-of-bounds sampling sets radius to current
    main-loop k, including if a look-ahead sample crossed the boundary.
    """
    center = tuple(map(f, sample(*position)))
    center_luma = luma(center)
    limit = 32 if center_luma < f(0.75) else 266
    radii = [-1.0] * 8
    for direction, (dx, dy) in enumerate(DIRECTIONS):
        length = f(math.sqrt(2)) if direction % 2 else 1.0
        lum = [center_luma]
        distance = [0.0]

        for k in range(1, limit):
            if not _extend_ray(position, bounds, sample, center, dx, dy, lum, distance, k):
                radii[direction] = mul(k, length)
                break
            width = int(add(mul(k, 0.25), 1.0))
            start = k - width + 1
            prev = total(lum[start - width : start])
            current = total(lum[start : k + 1])
            prev_color = mul(total(distance[start - width : start]), 0.125)
            current_color = mul(total(distance[start : k + 1]), 0.125)
            delta = sub(sub(prev, prev_color), sub(current, current_color))
            if delta > mul(width, f(0.04)) or -delta > mul(width, f(0.015)):
                end = min(k + width, limit - 1)
                if not _extend_ray(position, bounds, sample, center, dx, dy, lum, distance, end):
                    radii[direction] = mul(k, length)
                    break
                next_sum = total(lum[k + 1 : end + 1])
                next_color = mul(total(distance[k + 1 : end + 1]), 0.125)
                next_delta = sub(sub(current, current_color), sub(next_sum, next_color))
                if delta > mul(width, f(0.04)) and sub(delta, next_delta) > mul(width, f(0.01)):
                    background = min(next_sum, current)
                    ratio = div(sub(current, background), sub(prev, background))
                    radii[direction] = mul(add(mul(ratio, width), sub(start, 0.625)), length)
                    break
                if -delta > mul(width, f(0.015)) and -next_delta > mul(width, f(0.015)):
                    return False, radii
        else:
            if direction == 0:
                return False, radii
    return True, radii


def neighbor_mean(a: float, b: float) -> float:
    """Average valid adjacent radii, retaining an available single neighbor."""
    if a < 0:
        return b
    if b < 0:
        return a
    return mul(add(a, b), 0.5)


def repair_recenter(
    position: Position, radii: Sequence[float]
) -> tuple[bool, Position, list[float]]:
    """3DA00. Repaired array is returned even on failure (native mutates it).

    Outlier marking uses the ORIGINAL adjacent values; repair uses the marked
    adjacent values, never newly repaired ones. Successful non-outliers retain
    original values. Native diagonal recenter factor really is sqrt(2).
    """
    radii = list(map(f, radii))
    marked = radii.copy()
    position = cast(Position, tuple(map(f, position)))
    for i, value in enumerate(radii):
        neighbor = neighbor_mean(radii[(i - 1) % 8], radii[(i + 1) % 8])
        if neighbor >= 0:
            ratio = div(
                abs(sub(value, neighbor)), add(mul(value, f(0.333)), mul(neighbor, f(0.67)))
            )
            if ratio > 1:
                marked[i] = -1.0
    success = True
    for i, value in enumerate(marked):
        if value < 0:
            replacement = neighbor_mean(marked[(i - 1) % 8], marked[(i + 1) % 8])
            if replacement < 0:
                success = False
            else:
                radii[i] = replacement
    if not success:
        return False, position, radii
    return True, _recenter(position, radii), radii


def _recenter(position: tuple[float, float], radii: list[float]) -> tuple[float, float]:
    """Accumulate valid radii in native direction order and shift the center."""
    # Preserve native accumulation order, including skipped invalid radii.
    x = y = nx = ny = 0.0
    q = f(math.sqrt(2))
    if radii[6] >= 0:
        y = -radii[6]
        ny = 1.0
    if radii[7] >= 0:
        x = mul(radii[7], q)
        y = sub(y, x)
        nx = 1.0
        ny = add(ny, 1)
    if radii[0] >= 0:
        x = add(x, radii[0])
        nx = add(nx, 1)
    if radii[1] >= 0:
        v = mul(radii[1], q)
        x = add(x, v)
        y = add(y, v)
        nx = add(nx, 1)
        ny = add(ny, 1)
    if radii[2] >= 0:
        y = add(y, radii[2])
        ny = add(ny, 1)
    if radii[3] >= 0:
        v = mul(radii[3], q)
        x = sub(x, v)
        y = add(y, v)
        nx = add(nx, 1)
        ny = add(ny, 1)
    if radii[4] >= 0:
        x = sub(x, radii[4])
        nx = add(nx, 1)
    if radii[5] >= 0:
        v = mul(radii[5], q)
        x = sub(x, v)
        y = sub(y, v)
        nx = add(nx, 1)
        ny = add(ny, 1)
    return add(position[0], div(x, nx)), add(position[1], div(y, ny))


def initial_candidate(
    position: Position, bounds: Bounds, sample: Callable[[float, float], RGBA]
) -> tuple[bool, Position, float]:
    """97190 wrapper: commits center/size only on complete success."""
    success, radii = initial_extents(position, bounds, sample)
    if not success:
        return False, position, 0.0
    success, center, radii = repair_recenter(position, radii)
    if (
        not success
        or any(abs(sub(a, b)) > 2 for a, b in zip(center, position))
        or not inside(center, bounds)
    ):
        return False, position, 0.0
    positives = [r for r in radii if r > 0]
    size = div(total(positives), len(positives)) if positives else 0.0
    if not 0 < size <= 128:
        return False, position, 0.0
    return True, center, size
