"""Local linear-light aperture photometry on original image pixels.

Neighbour ownership partitions crowded apertures. The measured width is an
approximation; no full PSF fitting or Gaussian deblending is performed."""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from time import perf_counter
from typing import cast

import numpy as np
from scipy.spatial import cKDTree

from .array_types import RGB, BoolArray, ByteArray, DoubleArray, FloatArray, IntArray
from .detection.numeric import LUMA
from .imaging import ImageFrame, check_cancel
from .jit import njit
from .sky_mask import SkyMask
from .stars import Star

logger = logging.getLogger(__name__)
DIRECTIONS = np.array(
    [
        (math.cos(direction_angle), math.sin(direction_angle))
        for direction_angle in np.arange(16) * (2 * math.pi / 16)
    ]
)


@njit(cache=True)
def _sample_luminance(
    pixels: FloatArray, classes: ByteArray, cell_size: int, x: float, y: float
) -> float:
    """Sample rounded native pixels as luminance, returning NaN outside supported sky."""
    pixel_x: int
    pixel_y: int
    pixel_x, pixel_y = int(round(x)), int(round(y))
    if not (0 <= pixel_x < pixels.shape[1] and 0 <= pixel_y < pixels.shape[0]):
        return math.nan
    if classes.size and classes[pixel_y // cell_size, pixel_x // cell_size] == 0:
        return math.nan
    if pixels.shape[2] == 1:
        return float(pixels[pixel_y, pixel_x, 0])
    return (
        float(pixels[pixel_y, pixel_x, 0]) * LUMA[0]
        + float(pixels[pixel_y, pixel_x, 1]) * LUMA[1]
        + float(pixels[pixel_y, pixel_x, 2]) * LUMA[2]
    )


@njit(cache=True)
def _background_samples(
    pixels: FloatArray,
    classes: ByteArray,
    cell_size: int,
    x: float,
    y: float,
    inner: float,
    outer: float,
) -> DoubleArray:
    """Collect finite luminance samples in the original annulus scan order."""
    step: int = max(1, math.ceil(outer / 12))
    extent: int = math.ceil(outer)
    samples: DoubleArray = np.empty(((2 * extent // step + 2) ** 2, 3))
    sample_count: int = 0
    for delta_y in range(-extent, extent + 1, step):
        for delta_x in range(-extent, extent + 1, step):
            if inner**2 <= delta_x * delta_x + delta_y * delta_y <= outer**2:
                value: float = _sample_luminance(
                    pixels, classes, cell_size, x + delta_x, y + delta_y
                )
                if math.isfinite(value):
                    # Samples are read at integer pixels, even for a subpixel centre.
                    samples[sample_count, 0] = round(x + delta_x) - x
                    samples[sample_count, 1] = round(y + delta_y) - y
                    samples[sample_count, 2] = value
                    sample_count += 1
    return samples[:sample_count]


@njit(cache=True)
def _background(
    pixels: FloatArray,
    classes: ByteArray,
    cell_size: int,
    x: float,
    y: float,
    inner: float,
    outer: float,
) -> tuple[float, float, float]:
    """Fit a local luminance plane with three median-residual clipping iterations."""
    samples: DoubleArray = _background_samples(pixels, classes, cell_size, x, y, inner, outer)
    sample_count: int = len(samples)
    if sample_count < 8:
        return math.nan, 0.0, 0.0
    keep: BoolArray = np.ones(sample_count, np.bool_)
    residual: DoubleArray = np.empty(sample_count)
    background: float = 0.0
    gradient_x: float = 0.0
    gradient_y: float = 0.0
    for iteration in range(3):
        retained_count: float = 0.0
        sum_x: float = 0.0
        sum_y: float = 0.0
        sum_xx: float = 0.0
        sum_xy: float = 0.0
        sum_yy: float = 0.0
        sum_value: float = 0.0
        sum_x_value: float = 0.0
        sum_y_value: float = 0.0
        for sample_index in range(sample_count):
            if keep[sample_index]:
                delta_x: np.float64
                delta_y: np.float64
                value: np.float64
                delta_x, delta_y, value = samples[sample_index]
                retained_count += 1
                sum_x += delta_x
                sum_y += delta_y
                sum_xx += delta_x * delta_x
                sum_xy += delta_x * delta_y
                sum_yy += delta_y * delta_y
                sum_value += value
                sum_x_value += delta_x * value
                sum_y_value += delta_y * value
        if retained_count < 8:
            break
        centered_xx: float
        centered_xy: float
        centered_yy: float
        centered_xx, centered_yy, centered_xy = (
            sum_xx - sum_x * sum_x / retained_count,
            sum_yy - sum_y * sum_y / retained_count,
            sum_xy - sum_x * sum_y / retained_count,
        )
        centered_x_value: float
        centered_y_value: float
        centered_x_value, centered_y_value = (
            sum_x_value - sum_x * sum_value / retained_count,
            sum_y_value - sum_y * sum_value / retained_count,
        )
        determinant: float = centered_xx * centered_yy - centered_xy * centered_xy
        if abs(determinant) > 1e-10:
            gradient_x, gradient_y = (
                (centered_x_value * centered_yy - centered_y_value * centered_xy) / determinant,
                (centered_y_value * centered_xx - centered_x_value * centered_xy) / determinant,
            )
        background = (sum_value - gradient_x * sum_x - gradient_y * sum_y) / retained_count
        for sample_index in range(sample_count):
            residual[sample_index] = (
                samples[sample_index, 2]
                - background
                - gradient_x * samples[sample_index, 0]
                - gradient_y * samples[sample_index, 1]
            )
        center: np.float64 = np.median(residual)
        noise: float = max(1e-7, 1.4826 * np.median(np.abs(residual - center)))
        for sample_index in range(sample_count):
            keep[sample_index] = abs(residual[sample_index] - center) <= 3 * noise
    return background, gradient_x, gradient_y


@njit(cache=True)
def _aperture(
    pixels: FloatArray,
    classes: ByteArray,
    cell_size: int,
    x: float,
    y: float,
    background: float,
    gradient_x: float,
    gradient_y: float,
) -> tuple[int, float]:
    """Estimate half-height width and bounded signed-flux aperture from radial samples."""
    peak: float = 0.0
    for scan_y in range(-1, 2):
        for scan_x in range(-1, 2):
            value: float | np.float64 = _sample_luminance(
                pixels, classes, cell_size, x + scan_x, y + scan_y
            )
            if math.isfinite(value):
                signal: float = (
                    value
                    - background
                    - gradient_x * (round(x + scan_x) - x)
                    - gradient_y * (round(y + scan_y) - y)
                )
                peak = max(peak, signal)
    if peak <= 1e-8:
        return 4, 0.0
    ring: DoubleArray = np.empty(len(DIRECTIONS))
    previous: float | np.float64 = peak
    for radius in range(1, 33):
        sample_count: int = 0
        for direction in DIRECTIONS:
            delta_x: float
            delta_y: float
            delta_x, delta_y = radius * direction[0], radius * direction[1]
            value = _sample_luminance(pixels, classes, cell_size, x + delta_x, y + delta_y)
            if math.isfinite(value):
                ring[sample_count] = (
                    value
                    - background
                    - gradient_x * (round(x + delta_x) - x)
                    - gradient_y * (round(y + delta_y) - y)
                )
                sample_count += 1
        if sample_count < 4:
            return max(4, radius), 0.0
        value = np.median(ring[:sample_count])
        if value <= peak * 0.5:
            half: float = radius - 1 + (previous - peak * 0.5) / max(previous - value, 1e-12)
            # Half maximum estimates width only. The signal is the signed
            # aperture sum, not a Gaussian fit or a peak*width^2 substitute.
            return min(64, max(4, math.ceil(3.5 * half / 1.1774100225))), 2 * half
        previous = value
    return 64, 0.0


@njit(cache=True)
def _background_color_samples(
    pixels: FloatArray,
    classes: ByteArray,
    cell_size: int,
    x: float,
    y: float,
    inner: float,
    outer: float,
    background: float,
    gradient_x: float,
    gradient_y: float,
) -> tuple[DoubleArray, DoubleArray]:
    """Collect RGB planes and their luminance residuals without changing sample order."""
    extent: int
    step: int
    step, extent = max(1, math.ceil(outer / 12)), math.ceil(outer)
    samples: DoubleArray = np.empty(((2 * extent // step + 2) ** 2, 5))
    residuals: DoubleArray = np.empty(len(samples))
    sample_count: int = 0
    for delta_y in range(-extent, extent + 1, step):
        for delta_x in range(-extent, extent + 1, step):
            if inner**2 <= delta_x * delta_x + delta_y * delta_y <= outer**2:
                value: float = _sample_luminance(
                    pixels, classes, cell_size, x + delta_x, y + delta_y
                )
                if math.isfinite(value):
                    pixel_x: int
                    pixel_y: int
                    pixel_x, pixel_y = int(round(x + delta_x)), int(round(y + delta_y))
                    sample_delta_x: float
                    sample_delta_y: float
                    sample_delta_x, sample_delta_y = pixel_x - x, pixel_y - y
                    samples[sample_count, 0], samples[sample_count, 1] = (
                        sample_delta_x,
                        sample_delta_y,
                    )
                    for channel in range(3):
                        samples[sample_count, channel + 2] = pixels[
                            pixel_y, pixel_x, 0 if pixels.shape[2] == 1 else channel
                        ]
                    residuals[sample_count] = (
                        value
                        - background
                        - gradient_x * sample_delta_x
                        - gradient_y * sample_delta_y
                    )
                    sample_count += 1
    return samples[:sample_count], residuals[:sample_count]


@njit(cache=True)
def _background_colors(
    pixels: FloatArray,
    classes: ByteArray,
    cell_size: int,
    x: float,
    y: float,
    inner: float,
    outer: float,
    background: float,
    gradient_x: float,
    gradient_y: float,
) -> DoubleArray:
    """Fit RGB background planes using the luminance fit to reject bright sources."""
    samples: DoubleArray
    residuals: DoubleArray
    samples, residuals = _background_color_samples(
        pixels, classes, cell_size, x, y, inner, outer, background, gradient_x, gradient_y
    )
    sample_count: int = len(samples)
    planes: DoubleArray = np.full((3, 3), np.nan)
    if sample_count < 8:
        return planes
    residuals = residuals[:sample_count]
    center: np.float64 = np.median(residuals)
    noise: float = max(1e-7, 1.4826 * np.median(np.abs(residuals - center)))
    keep: BoolArray = np.abs(residuals - center) <= 3 * noise
    if keep.sum() < 8:
        keep[:] = True
    retained_count: float = 0.0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_xy: float = 0.0
    sum_yy: float = 0.0
    totals: DoubleArray = np.zeros((3, 3))
    for sample_index in range(sample_count):
        if keep[sample_index]:
            delta_x: np.float64
            delta_y: np.float64
            delta_x, delta_y = samples[sample_index, 0], samples[sample_index, 1]
            retained_count += 1
            sum_x += delta_x
            sum_y += delta_y
            sum_xx += delta_x * delta_x
            sum_xy += delta_x * delta_y
            sum_yy += delta_y * delta_y
            for channel in range(3):
                value: np.float64 = samples[sample_index, channel + 2]
                totals[channel, 0] += value
                totals[channel, 1] += delta_x * value
                totals[channel, 2] += delta_y * value
    centered_xx: float
    centered_xy: float
    centered_yy: float
    centered_xx, centered_yy, centered_xy = (
        sum_xx - sum_x * sum_x / retained_count,
        sum_yy - sum_y * sum_y / retained_count,
        sum_xy - sum_x * sum_y / retained_count,
    )
    determinant: float = centered_xx * centered_yy - centered_xy * centered_xy
    for channel in range(3):
        sum_value, sum_x_value, sum_y_value = totals[channel]
        centered_x_value: float
        centered_y_value: float
        centered_x_value, centered_y_value = (
            sum_x_value - sum_x * sum_value / retained_count,
            sum_y_value - sum_y * sum_value / retained_count,
        )
        color_gradient_x: float = 0.0
        color_gradient_y: float = 0.0
        if abs(determinant) > 1e-10:
            color_gradient_x, color_gradient_y = (
                (centered_x_value * centered_yy - centered_y_value * centered_xy) / determinant,
                (centered_y_value * centered_xx - centered_x_value * centered_xy) / determinant,
            )
        planes[channel, 0] = (
            sum_value - color_gradient_x * sum_x - color_gradient_y * sum_y
        ) / retained_count
        planes[channel, 1], planes[channel, 2] = color_gradient_x, color_gradient_y
    return planes


@njit(cache=True)
def _source_aperture(
    pixels: FloatArray, classes: ByteArray, cell_size: int, x: float, y: float
) -> tuple[int, float, float, float, float, float, float] | None:
    """Refine a source aperture and background, rejecting unsupported measurements."""
    if not math.isfinite(_sample_luminance(pixels, classes, cell_size, x, y)):
        return None
    background, gradient_x, gradient_y = _background(pixels, classes, cell_size, x, y, 12, 16)
    if not math.isfinite(background):
        return None
    radius: int
    width: float
    radius, width = _aperture(pixels, classes, cell_size, x, y, background, gradient_x, gradient_y)
    inner: float
    outer: float
    inner, outer = max(radius + 3, 1.5 * radius), 2 * radius
    background, gradient_x, gradient_y = _background(pixels, classes, cell_size, x, y, inner, outer)
    if not math.isfinite(background):
        return None
    updated, width = _aperture(pixels, classes, cell_size, x, y, background, gradient_x, gradient_y)
    if updated > radius + 2:
        radius = updated
        inner, outer = max(radius + 3, 1.5 * radius), 2 * radius
        background, gradient_x, gradient_y = _background(
            pixels, classes, cell_size, x, y, inner, outer
        )
        if not math.isfinite(background):
            return None
    return radius, width, inner, outer, background, gradient_x, gradient_y


@njit(cache=True)
def _owned(
    pixel_x: int,
    pixel_y: int,
    distance: float,
    positions: DoubleArray,
    neighbours: IntArray,
    index: int,
) -> float:
    """Return this source's share of a pixel: 0 if a neighbour is nearer, split evenly on ties."""
    tied: int = 1
    for other in neighbours[index]:
        if other < len(positions) and other != index:
            neighbour_x: np.float64
            neighbour_y: np.float64
            neighbour_x, neighbour_y = positions[other]
            other_distance: float = (pixel_x - neighbour_x) ** 2 + (pixel_y - neighbour_y) ** 2
            if other_distance < distance:
                return 0.0
            if other_distance == distance:
                tied += 1
    return 1.0 / tied


@njit(cache=True)
def _mirrored(
    pixel_x: int, pixel_y: int, x: float, y: float, width: int, height: int
) -> tuple[int, int]:
    """Map an aperture pixel outside the image onto a symmetric pixel inside, or (-1, -1).

    Try the point reflection through the source first, then mirroring one axis, which
    keeps sources in image corners measurable under a symmetric PSF.
    """
    for candidate_x, candidate_y in (
        (pixel_x, pixel_y),
        (round(2 * x - pixel_x), round(2 * y - pixel_y)),
        (round(2 * x - pixel_x), pixel_y),
        (pixel_x, round(2 * y - pixel_y)),
    ):
        if 0 <= candidate_x < width and 0 <= candidate_y < height:
            return candidate_x, candidate_y
    return -1, -1


@njit(cache=True)
def _integrate_owned_aperture(
    pixels: FloatArray,
    classes: ByteArray,
    cell_size: int,
    x: float,
    y: float,
    radius: int,
    background: tuple[float, float, float],
    planes: DoubleArray,
    positions: DoubleArray,
    neighbours: IntArray,
    index: int,
) -> tuple[float, DoubleArray]:
    """Sum signed source flux only over finite pixels owned by this aperture.

    Aperture pixels outside the image are replaced by their point reflection through the
    source center, assuming a symmetric PSF, so border sources are not under-measured.
    """
    background_level, gradient_x, gradient_y = background
    height, width = pixels.shape[0], pixels.shape[1]
    flux: float = 0.0
    rgb: DoubleArray = np.zeros(3)
    for pixel_y in range(math.ceil(y - radius), math.floor(y + radius) + 1):
        for pixel_x in range(math.ceil(x - radius), math.floor(x + radius) + 1):
            distance: float = (pixel_x - x) ** 2 + (pixel_y - y) ** 2
            if distance > radius * radius:
                continue
            share: float = _owned(pixel_x, pixel_y, distance, positions, neighbours, index)
            if share == 0:
                continue
            sample_x, sample_y = _mirrored(pixel_x, pixel_y, x, y, width, height)
            if sample_x < 0:
                continue
            delta_x: float = sample_x - x
            delta_y: float = sample_y - y
            value: float = _sample_luminance(pixels, classes, cell_size, sample_x, sample_y)
            if math.isfinite(value):
                flux += share * (
                    value - background_level - gradient_x * delta_x - gradient_y * delta_y
                )
                for channel in range(3):
                    rgb[channel] += share * (
                        pixels[sample_y, sample_x, 0 if pixels.shape[2] == 1 else channel]
                        - planes[channel, 0]
                        - planes[channel, 1] * delta_x
                        - planes[channel, 2] * delta_y
                    )
    return flux, rgb


@njit(cache=True, nogil=True)
def _measure_batch(
    pixels: FloatArray,
    classes: ByteArray,
    cell_size: int,
    positions: DoubleArray,
    neighbours: IntArray,
    start: int,
    stop: int,
) -> DoubleArray:
    """Return 17-column flux, RGB, width, background-plane and peak measurements."""
    result: DoubleArray = np.zeros((stop - start, 17))
    for index in range(start, stop):
        x: np.float64
        y: np.float64
        x, y = positions[index]
        aperture: tuple[int, float, float, float, float, float, float] | None = _source_aperture(
            pixels, classes, cell_size, x, y
        )
        if aperture is None:
            continue
        radius, width, inner, outer, background, gradient_x, gradient_y = aperture
        planes: DoubleArray = _background_colors(
            pixels, classes, cell_size, x, y, inner, outer, background, gradient_x, gradient_y
        )
        if not np.isfinite(planes).all():
            continue
        flux, rgb = _integrate_owned_aperture(
            pixels,
            classes,
            cell_size,
            x,
            y,
            radius,
            (background, gradient_x, gradient_y),
            planes,
            positions,
            neighbours,
            index,
        )
        result[index - start, 0] = max(0, flux)
        result[index - start, 1:4] = rgb
        result[index - start, 4] = width
        result[index - start, 5:14] = planes.T.flatten()
        peak: float = max(
            0,
            _sample_luminance(pixels, classes, cell_size, x, y)
            - background
            - gradient_x * (round(x) - x)
            - gradient_y * (round(y) - y),
        )
        positive_rgb: DoubleArray = np.maximum(rgb, 0)
        source_luma: np.float64 = positive_rgb @ np.array(LUMA)
        if source_luma > 0:
            result[index - start, 14:17] = positive_rgb * (peak / source_luma)
    return result


def _measure_sources(
    frame: ImageFrame,
    stars: tuple[Star, ...],
    *,
    sky_mask: SkyMask | None = None,
    cancel: Event | None = None,
) -> DoubleArray:
    """Partition immutable sources across bounded, cancellation-aware measurement batches."""
    check_cancel(cancel)
    if not stars:
        return np.empty((0, 17))
    positions: DoubleArray = np.array([(star.x, star.y) for star in stars])
    neighbours: IntArray = cKDTree(positions).query(positions, k=min(9, len(stars)))[1]
    neighbours = np.asarray(neighbours).reshape(len(stars), -1)
    classes: ByteArray = np.empty((0, 0), np.uint8) if sky_mask is None else sky_mask.classes
    cell_size: int = 1 if sky_mask is None else sky_mask.cell_px
    signals: DoubleArray = np.empty((len(stars), 17))

    def measure(start: int) -> tuple[int, DoubleArray]:
        """Measure at most 256 sources after checking the shared cancellation token."""
        check_cancel(cancel)
        stop: int = min(len(stars), start + 256)
        return start, _measure_batch(
            frame.pixels, classes, cell_size, positions, neighbours, start, stop
        )

    starts: range = range(0, len(stars), 256)
    if len(stars) < 1024:
        batches: map[tuple[int, DoubleArray]] = map(measure, starts)
        for start, values in batches:
            signals[start : start + len(values)] = values
    else:
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            for start, values in pool.map(measure, starts):
                check_cancel(cancel)
                signals[start : start + len(values)] = values
    check_cancel(cancel)
    return signals


def measure_source_signals(
    frame: ImageFrame,
    stars: tuple[Star, ...],
    *,
    sky_mask: SkyMask | None = None,
    cancel: Event | None = None,
) -> DoubleArray:
    """Return signed-aperture stellar signals, clamped nonnegative after integration."""
    return _measure_sources(frame, stars, sky_mask=sky_mask, cancel=cancel)[:, 0]


def calibrate_sources(
    frame: ImageFrame,
    stars: tuple[Star, ...],
    *,
    sky_mask: SkyMask | None = None,
    cancel: Event | None = None,
) -> tuple[Star, ...]:
    """Measure original pixels and normalize catalogue colors, strengths and filter profiles."""
    started: float = perf_counter()
    measured: DoubleArray = _measure_sources(frame, stars, sky_mask=sky_mask, cancel=cancel)
    if not stars:
        return stars
    signals: DoubleArray = measured[:, 0]
    colors: DoubleArray = np.maximum(measured[:, 1:4], 0)
    luminance: DoubleArray = colors @ np.array(LUMA)
    colors = np.divide(
        colors, luminance[:, None], out=np.ones_like(colors), where=luminance[:, None] > 0
    )
    colors *= signals[:, None]
    peak: float = float(colors.max(initial=0))
    colors *= 0.87 / peak if peak > 0 else 0
    brightness: DoubleArray = colors @ np.array(LUMA)
    valid: BoolArray = (signals > 0) & (measured[:, 4] > 0)
    weights: DoubleArray = np.full(len(stars), 0.5)
    if valid.any():
        score: DoubleArray = (
            np.sqrt(signals[valid] / np.median(signals[valid]))
            * measured[valid, 4]
            / np.median(measured[valid, 4])
        )
        weights[valid] = score**2 / (score**2 + 16)
    result: tuple[Star, ...] = tuple(
        replace(
            star,
            brightness=float(calibrated_brightness),
            color=cast(RGB, tuple(calibrated_color)),
            source_profile=tuple(measurement[4:14]) if measured_valid else (),
            source_peak=cast(RGB, tuple(measurement[14:17])) if measured_valid else (0.0, 0.0, 0.0),
            soft_weight=float(source_weight),
        )
        for star, calibrated_brightness, calibrated_color, measurement, source_weight, measured_valid in zip(
            stars, brightness, colors, measured, weights, valid
        )
    )
    logger.info("Photometry: %d sources measured in %.3fs", len(stars), perf_counter() - started)
    return result
