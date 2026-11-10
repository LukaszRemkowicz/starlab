"""Reference source-ray tracing, outlier repair, and center estimation.

Input sampler supplies RGBA float32 at the exact requested float position.
This module does not implement that callback (interpolation/image access).
All finite RGB inputs, finite bounds, and native float32 operation ordering.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from typing import cast

from ..array_types import RGBA, Bounds, DoubleArray, Position
from .precision import round_to_float32

DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
LUMA = (0.2126390039920807, 0.7151686549186707, 0.07219231128692627)


def add_float32(left: float, right: float) -> float:
    """Add with one float32 rounding step."""
    return round_to_float32(left + right)


def subtract_float32(left: float, right: float) -> float:
    """Subtract with one float32 rounding step."""
    return round_to_float32(left - right)


def multiply_float32(left: float, right: float) -> float:
    """Multiply with one float32 rounding step."""
    return round_to_float32(left * right)


def divide_float32(numerator: float, denominator: float) -> float:
    """Divide with native zero/NaN handling and float32 rounding."""
    if denominator == 0:
        return math.nan if numerator == 0 else math.copysign(math.inf, numerator * denominator)
    return round_to_float32(numerator / denominator)


def sum_float32(values: Iterable[float]) -> float:
    """Accumulate in input order, rounding each addition to float32."""
    value: float = 0.0
    for item in values:
        value = add_float32(value, item)
    return value


def luma(rgb: Sequence[float]) -> float:
    """Combine the first three channels in the recovered B/R/G operation order."""
    return add_float32(
        add_float32(multiply_float32(rgb[2], LUMA[2]), multiply_float32(rgb[0], LUMA[0])),
        multiply_float32(rgb[1], LUMA[1]),
    )


def inside(position: Position, bounds: Bounds) -> bool:
    """Test global coordinates against half-open image bounds."""
    x, y = position
    left, top, right, bottom = bounds
    return left <= x < right and top <= y < bottom


def _extend_ray(
    position: tuple[float, float],
    bounds: Bounds,
    sample: Callable[[float, float], tuple[float, ...]],
    center: tuple[float, ...],
    dx: int,
    dy: int,
    luminance_samples: list[float],
    color_distances: list[float],
    needed: int,
) -> bool:
    """Append reference ray samples through needed, retaining native float32 order."""
    while len(luminance_samples) <= needed:
        sample_index: int = len(luminance_samples)
        sample_position: Position = (
            add_float32(position[0], sample_index * dx),
            add_float32(position[1], sample_index * dy),
        )
        if not inside(sample_position, bounds):
            return False
        color: tuple[float, ...] = tuple(map(round_to_float32, sample(*sample_position)))
        luminance_samples.append(luma(color))
        differences: list[float] = [
            multiply_float32(
                subtract_float32(center_channel, sample_channel),
                subtract_float32(center_channel, sample_channel),
            )
            for center_channel, sample_channel in zip(center[:3], color[:3])
        ]
        color_distances.append(
            round_to_float32(
                math.sqrt(add_float32(add_float32(differences[1], differences[0]), differences[2]))
            )
        )
    return True


def initial_extents(
    position: Position, bounds: Bounds, sample: Callable[[float, float], RGBA]
) -> tuple[bool, list[float]]:
    """Trace eight ray extents; unset radii retain the caller's -1.

    The first ray reaching its limit rejects the entire candidate. Subsequent
    rays reaching their limit remain -1. Any ray with sustained outward rise
    rejects the entire candidate. Out-of-bounds sampling sets radius to current
    main-loop k, including if a look-ahead sample crossed the boundary.
    """
    center: tuple[float, ...] = tuple(map(round_to_float32, sample(*position)))
    center_luma: float = luma(center)
    limit: int = 32 if center_luma < round_to_float32(0.75) else 266
    radii: list[float] = [-1.0] * 8
    for direction, (dx, dy) in enumerate(DIRECTIONS):
        length: float = round_to_float32(math.sqrt(2)) if direction % 2 else 1.0
        luminance_samples: list[float] = [center_luma]
        color_distances: list[float] = [0.0]

        for sample_index in range(1, limit):
            if not _extend_ray(
                position,
                bounds,
                sample,
                center,
                dx,
                dy,
                luminance_samples,
                color_distances,
                sample_index,
            ):
                radii[direction] = multiply_float32(sample_index, length)
                break
            width: int = int(add_float32(multiply_float32(sample_index, 0.25), 1.0))
            start: int = sample_index - width + 1
            previous_luminance: float = sum_float32(luminance_samples[start - width : start])
            current_luminance: float = sum_float32(luminance_samples[start : sample_index + 1])
            previous_color_distance: float = multiply_float32(
                sum_float32(color_distances[start - width : start]), 0.125
            )
            current_color_distance: float = multiply_float32(
                sum_float32(color_distances[start : sample_index + 1]), 0.125
            )
            delta: float = subtract_float32(
                subtract_float32(previous_luminance, previous_color_distance),
                subtract_float32(current_luminance, current_color_distance),
            )
            if delta > multiply_float32(width, round_to_float32(0.04)) or -delta > multiply_float32(
                width, round_to_float32(0.015)
            ):
                end: int = min(sample_index + width, limit - 1)
                if not _extend_ray(
                    position,
                    bounds,
                    sample,
                    center,
                    dx,
                    dy,
                    luminance_samples,
                    color_distances,
                    end,
                ):
                    radii[direction] = multiply_float32(sample_index, length)
                    break
                next_luminance: float = sum_float32(luminance_samples[sample_index + 1 : end + 1])
                next_color_distance: float = multiply_float32(
                    sum_float32(color_distances[sample_index + 1 : end + 1]), 0.125
                )
                next_delta: float = subtract_float32(
                    subtract_float32(current_luminance, current_color_distance),
                    subtract_float32(next_luminance, next_color_distance),
                )
                if delta > multiply_float32(width, round_to_float32(0.04)) and subtract_float32(
                    delta, next_delta
                ) > multiply_float32(width, round_to_float32(0.01)):
                    background: float = min(next_luminance, current_luminance)
                    ratio: float = divide_float32(
                        subtract_float32(current_luminance, background),
                        subtract_float32(previous_luminance, background),
                    )
                    radii[direction] = multiply_float32(
                        add_float32(multiply_float32(ratio, width), subtract_float32(start, 0.625)),
                        length,
                    )
                    break
                if -delta > multiply_float32(
                    width, round_to_float32(0.015)
                ) and -next_delta > multiply_float32(width, round_to_float32(0.015)):
                    return False, radii
        else:
            if direction == 0:
                return False, radii
    return True, radii


def neighbor_mean(previous_radius: float, next_radius: float) -> float:
    """Average valid adjacent radii, retaining an available single neighbor."""
    if previous_radius < 0:
        return next_radius
    if next_radius < 0:
        return previous_radius
    return multiply_float32(add_float32(previous_radius, next_radius), 0.5)


def repair_recenter(
    position: Position, radii: Sequence[float]
) -> tuple[bool, Position, list[float]]:
    """Return repaired radii even when extent repair fails.

    Outlier marking uses the ORIGINAL adjacent values; repair uses the marked
    adjacent values, never newly repaired ones. Successful non-outliers retain
    original values. Native diagonal recenter factor really is sqrt(2).
    """
    radii = list(map(round_to_float32, radii))
    position = cast(Position, tuple(map(round_to_float32, position)))
    if not repair_extents(radii):
        return False, position, radii
    return True, recenter(position, radii), radii


def repair_extents(radii: list[float] | DoubleArray) -> bool:
    """Repair marked outlier radii in place using a frozen neighbor snapshot."""
    marked: list[float] | DoubleArray = radii.copy()
    for radius_index, value in enumerate(radii):
        neighbor: float = neighbor_mean(
            radii[(radius_index - 1) % 8], radii[(radius_index + 1) % 8]
        )
        if neighbor >= 0:
            ratio: float = divide_float32(
                abs(subtract_float32(value, neighbor)),
                add_float32(
                    multiply_float32(value, round_to_float32(0.333)),
                    multiply_float32(neighbor, round_to_float32(0.67)),
                ),
            )
            if ratio > 1:
                marked[radius_index] = -1.0
    success: bool = True
    for radius_index, value in enumerate(marked):
        if value < 0:
            replacement: float = neighbor_mean(
                marked[(radius_index - 1) % 8], marked[(radius_index + 1) % 8]
            )
            if replacement < 0:
                success = False
            else:
                radii[radius_index] = replacement
    return success


def recenter(
    position: tuple[float, float], radii: list[float] | DoubleArray
) -> tuple[float, float]:
    """Accumulate valid radii in native direction order and shift the center."""
    # Preserve native accumulation order, including skipped invalid radii.
    x: float = 0.0
    y: float = 0.0
    horizontal_count: float = 0.0
    vertical_count: float = 0.0
    diagonal_scale: float = round_to_float32(math.sqrt(2))
    if radii[6] >= 0:
        y = -radii[6]
        vertical_count = 1.0
    if radii[7] >= 0:
        x = multiply_float32(radii[7], diagonal_scale)
        y = subtract_float32(y, x)
        horizontal_count = 1.0
        vertical_count = add_float32(vertical_count, 1)
    if radii[0] >= 0:
        x = add_float32(x, radii[0])
        horizontal_count = add_float32(horizontal_count, 1)
    if radii[1] >= 0:
        diagonal_offset: float = multiply_float32(radii[1], diagonal_scale)
        x = add_float32(x, diagonal_offset)
        y = add_float32(y, diagonal_offset)
        horizontal_count = add_float32(horizontal_count, 1)
        vertical_count = add_float32(vertical_count, 1)
    if radii[2] >= 0:
        y = add_float32(y, radii[2])
        vertical_count = add_float32(vertical_count, 1)
    if radii[3] >= 0:
        diagonal_offset = multiply_float32(radii[3], diagonal_scale)
        x = subtract_float32(x, diagonal_offset)
        y = add_float32(y, diagonal_offset)
        horizontal_count = add_float32(horizontal_count, 1)
        vertical_count = add_float32(vertical_count, 1)
    if radii[4] >= 0:
        x = subtract_float32(x, radii[4])
        horizontal_count = add_float32(horizontal_count, 1)
    if radii[5] >= 0:
        diagonal_offset = multiply_float32(radii[5], diagonal_scale)
        x = subtract_float32(x, diagonal_offset)
        y = subtract_float32(y, diagonal_offset)
        horizontal_count = add_float32(horizontal_count, 1)
        vertical_count = add_float32(vertical_count, 1)
    return add_float32(position[0], divide_float32(x, horizontal_count)), add_float32(
        position[1], divide_float32(y, vertical_count)
    )


def initial_candidate(
    position: Position, bounds: Bounds, sample: Callable[[float, float], RGBA]
) -> tuple[bool, Position, float]:
    """Accept a center and radius only after every ray-validation step succeeds."""
    success, radii = initial_extents(position, bounds, sample)
    if not success:
        return False, position, 0.0
    success, center, radii = repair_recenter(position, radii)
    if (
        not success
        or any(
            abs(subtract_float32(center_coordinate, seed_coordinate)) > 2
            for center_coordinate, seed_coordinate in zip(center, position)
        )
        or not inside(center, bounds)
    ):
        return False, position, 0.0
    positive_radii: list[float] = [radius for radius in radii if radius > 0]
    size: float = (
        divide_float32(sum_float32(positive_radii), len(positive_radii)) if positive_radii else 0.0
    )
    if not 0 < size <= 128:
        return False, position, 0.0
    return True, center, size
