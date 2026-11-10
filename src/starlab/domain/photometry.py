"""Local linear-light aperture photometry on original image pixels.

Neighbour ownership partitions crowded apertures. The measured width is an
approximation; no full PSF fitting or Gaussian deblending is performed."""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from time import perf_counter

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from .detection.numeric import LUMA
from .imaging import check_cancel

logger = logging.getLogger(__name__)
DIRECTIONS = np.array([(math.cos(a), math.sin(a)) for a in np.arange(16) * (2 * math.pi / 16)])


@njit(cache=True)
def _sample(pixels, classes, cell, x, y):
    px, py = int(round(x)), int(round(y))
    if not (0 <= px < pixels.shape[1] and 0 <= py < pixels.shape[0]):
        return math.nan
    if classes.size and classes[py // cell, px // cell] == 0:
        return math.nan
    if pixels.shape[2] == 1:
        return float(pixels[py, px, 0])
    return (
        float(pixels[py, px, 0]) * LUMA[0]
        + float(pixels[py, px, 1]) * LUMA[1]
        + float(pixels[py, px, 2]) * LUMA[2]
    )


@njit(cache=True)
def _background_samples(
    pixels: np.ndarray,
    classes: np.ndarray,
    cell: int,
    x: float,
    y: float,
    inner: float,
    outer: float,
) -> np.ndarray:
    """Collect finite luminance samples in the original annulus scan order."""
    step = max(1, math.ceil(outer / 12))
    extent = math.ceil(outer)
    samples = np.empty(((2 * extent // step + 2) ** 2, 3))
    n = 0
    for dy in range(-extent, extent + 1, step):
        for dx in range(-extent, extent + 1, step):
            if inner**2 <= dx * dx + dy * dy <= outer**2:
                value = _sample(pixels, classes, cell, x + dx, y + dy)
                if math.isfinite(value):
                    # Samples are read at integer pixels, even for a subpixel centre.
                    samples[n, 0] = round(x + dx) - x
                    samples[n, 1] = round(y + dy) - y
                    samples[n, 2] = value
                    n += 1
    return samples[:n]


@njit(cache=True)
def _background(pixels, classes, cell, x, y, inner, outer):
    samples = _background_samples(pixels, classes, cell, x, y, inner, outer)
    n = len(samples)
    if n < 8:
        return math.nan, 0.0, 0.0
    keep = np.ones(n, np.bool_)
    residual = np.empty(n)
    background = gx = gy = 0.0
    for iteration in range(3):
        count = sx = sy = sxx = sxy = syy = sv = sxv = syv = 0.0
        for j in range(n):
            if keep[j]:
                dx, dy, value = samples[j]
                count += 1
                sx += dx
                sy += dy
                sxx += dx * dx
                sxy += dx * dy
                syy += dy * dy
                sv += value
                sxv += dx * value
                syv += dy * value
        if count < 8:
            break
        xx, yy, xy = sxx - sx * sx / count, syy - sy * sy / count, sxy - sx * sy / count
        xv, yv = sxv - sx * sv / count, syv - sy * sv / count
        determinant = xx * yy - xy * xy
        if abs(determinant) > 1e-10:
            gx, gy = (xv * yy - yv * xy) / determinant, (yv * xx - xv * xy) / determinant
        background = (sv - gx * sx - gy * sy) / count
        for j in range(n):
            residual[j] = samples[j, 2] - background - gx * samples[j, 0] - gy * samples[j, 1]
        center = np.median(residual)
        noise = max(1e-7, 1.4826 * np.median(np.abs(residual - center)))
        for j in range(n):
            keep[j] = abs(residual[j] - center) <= 3 * noise
    return background, gx, gy


@njit(cache=True)
def _aperture(pixels, classes, cell, x, y, background, gx, gy):
    peak = 0.0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            value = _sample(pixels, classes, cell, x + dx, y + dy)
            if math.isfinite(value):
                signal = value - background - gx * (round(x + dx) - x) - gy * (round(y + dy) - y)
                peak = max(peak, signal)
    if peak <= 1e-8:
        return 4, 0.0
    ring = np.empty(len(DIRECTIONS))
    previous = peak
    for radius in range(1, 33):
        n = 0
        for direction in DIRECTIONS:
            dx, dy = radius * direction[0], radius * direction[1]
            value = _sample(pixels, classes, cell, x + dx, y + dy)
            if math.isfinite(value):
                ring[n] = value - background - gx * (round(x + dx) - x) - gy * (round(y + dy) - y)
                n += 1
        if n < 4:
            return max(4, radius), 0.0
        value = np.median(ring[:n])
        if value <= peak * 0.5:
            half = radius - 1 + (previous - peak * 0.5) / max(previous - value, 1e-12)
            # Half maximum estimates width only. The signal is the signed
            # aperture sum, not a Gaussian fit or a peak*width^2 substitute.
            return min(64, max(4, math.ceil(3.5 * half / 1.1774100225))), 2 * half
        previous = value
    return 64, 0.0


@njit(cache=True)
def _background_color_samples(
    pixels: np.ndarray,
    classes: np.ndarray,
    cell: int,
    x: float,
    y: float,
    inner: float,
    outer: float,
    background: float,
    gx: float,
    gy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect RGB planes and their luminance residuals without changing sample order."""
    step, extent = max(1, math.ceil(outer / 12)), math.ceil(outer)
    samples = np.empty(((2 * extent // step + 2) ** 2, 5))
    residuals = np.empty(len(samples))
    n = 0
    for dy in range(-extent, extent + 1, step):
        for dx in range(-extent, extent + 1, step):
            if inner**2 <= dx * dx + dy * dy <= outer**2:
                value = _sample(pixels, classes, cell, x + dx, y + dy)
                if math.isfinite(value):
                    px, py = int(round(x + dx)), int(round(y + dy))
                    sx, sy = px - x, py - y
                    samples[n, 0], samples[n, 1] = sx, sy
                    for channel in range(3):
                        samples[n, channel + 2] = pixels[
                            py, px, 0 if pixels.shape[2] == 1 else channel
                        ]
                    residuals[n] = value - background - gx * sx - gy * sy
                    n += 1
    return samples[:n], residuals[:n]


@njit(cache=True)
def _background_colors(pixels, classes, cell, x, y, inner, outer, background, gx, gy):
    """Fit RGB background planes using the luminance fit to reject bright sources."""
    samples, residuals = _background_color_samples(
        pixels, classes, cell, x, y, inner, outer, background, gx, gy
    )
    n = len(samples)
    planes = np.full((3, 3), np.nan)
    if n < 8:
        return planes
    residuals = residuals[:n]
    center = np.median(residuals)
    noise = max(1e-7, 1.4826 * np.median(np.abs(residuals - center)))
    keep = np.abs(residuals - center) <= 3 * noise
    if keep.sum() < 8:
        keep[:] = True
    count = sx = sy = sxx = sxy = syy = 0.0
    totals = np.zeros((3, 3))
    for i in range(n):
        if keep[i]:
            dx, dy = samples[i, 0], samples[i, 1]
            count += 1
            sx += dx
            sy += dy
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
            for channel in range(3):
                value = samples[i, channel + 2]
                totals[channel, 0] += value
                totals[channel, 1] += dx * value
                totals[channel, 2] += dy * value
    xx, yy, xy = sxx - sx * sx / count, syy - sy * sy / count, sxy - sx * sy / count
    determinant = xx * yy - xy * xy
    for channel in range(3):
        sv, sxv, syv = totals[channel]
        xv, yv = sxv - sx * sv / count, syv - sy * sv / count
        cx = cy = 0.0
        if abs(determinant) > 1e-10:
            cx, cy = (xv * yy - yv * xy) / determinant, (yv * xx - xv * xy) / determinant
        planes[channel, 0] = (sv - cx * sx - cy * sy) / count
        planes[channel, 1], planes[channel, 2] = cx, cy
    return planes


@njit(cache=True)
def _source_aperture(
    pixels: np.ndarray, classes: np.ndarray, cell: int, x: float, y: float
) -> tuple[int, float, float, float, float, float, float] | None:
    """Refine a source aperture and background, rejecting unsupported measurements."""
    if not math.isfinite(_sample(pixels, classes, cell, x, y)):
        return None
    bg, gx, gy = _background(pixels, classes, cell, x, y, 12, 16)
    if not math.isfinite(bg):
        return None
    radius, width = _aperture(pixels, classes, cell, x, y, bg, gx, gy)
    inner, outer = max(radius + 3, 1.5 * radius), 2 * radius
    bg, gx, gy = _background(pixels, classes, cell, x, y, inner, outer)
    if not math.isfinite(bg):
        return None
    updated, width = _aperture(pixels, classes, cell, x, y, bg, gx, gy)
    if updated > radius + 2:
        radius = updated
        inner, outer = max(radius + 3, 1.5 * radius), 2 * radius
        bg, gx, gy = _background(pixels, classes, cell, x, y, inner, outer)
        if not math.isfinite(bg):
            return None
    return radius, width, inner, outer, bg, gx, gy


@njit(cache=True)
def _integrate_owned_aperture(
    pixels: np.ndarray,
    classes: np.ndarray,
    cell: int,
    x: float,
    y: float,
    radius: int,
    background: tuple[float, float, float],
    planes: np.ndarray,
    positions: np.ndarray,
    neighbours: np.ndarray,
    index: int,
) -> tuple[float, np.ndarray]:
    """Sum signed source flux only over finite pixels owned by this aperture."""
    bg, gx, gy = background
    flux = 0.0
    rgb = np.zeros(3)
    for py in range(
        max(0, math.ceil(y - radius)), min(pixels.shape[0], math.floor(y + radius) + 1)
    ):
        for px in range(
            max(0, math.ceil(x - radius)), min(pixels.shape[1], math.floor(x + radius) + 1)
        ):
            dx, dy = px - x, py - y
            distance = dx * dx + dy * dy
            if distance > radius * radius:
                continue
            owned = True
            for other in neighbours[index]:
                if other < len(positions) and other != index:
                    nx, ny = positions[other]
                    if (px - nx) ** 2 + (py - ny) ** 2 < distance:
                        owned = False
                        break
            if owned:
                value = _sample(pixels, classes, cell, px, py)
                if math.isfinite(value):
                    flux += value - bg - gx * dx - gy * dy
                    for channel in range(3):
                        rgb[channel] += (
                            pixels[py, px, 0 if pixels.shape[2] == 1 else channel]
                            - planes[channel, 0]
                            - planes[channel, 1] * dx
                            - planes[channel, 2] * dy
                        )
    return flux, rgb


@njit(cache=True, nogil=True)
def _measure_batch(pixels, classes, cell, positions, neighbours, start, stop):
    result = np.zeros((stop - start, 17))
    for index in range(start, stop):
        x, y = positions[index]
        aperture = _source_aperture(pixels, classes, cell, x, y)
        if aperture is None:
            continue
        radius, width, inner, outer, bg, gx, gy = aperture
        planes = _background_colors(pixels, classes, cell, x, y, inner, outer, bg, gx, gy)
        if not np.isfinite(planes).all():
            continue
        flux, rgb = _integrate_owned_aperture(
            pixels, classes, cell, x, y, radius, (bg, gx, gy), planes, positions, neighbours, index
        )
        result[index - start, 0] = max(0, flux)
        result[index - start, 1:4] = rgb
        result[index - start, 4] = width
        result[index - start, 5:14] = planes.T.flatten()
        peak = max(
            0, _sample(pixels, classes, cell, x, y) - bg - gx * (round(x) - x) - gy * (round(y) - y)
        )
        positive_rgb = np.maximum(rgb, 0)
        source_luma = positive_rgb @ np.array(LUMA)
        if source_luma > 0:
            result[index - start, 14:17] = positive_rgb * (peak / source_luma)
    return result


def _measure_sources(frame, stars, *, sky_mask=None, cancel=None):
    check_cancel(cancel)
    if not stars:
        return np.empty((0, 17))
    positions = np.array([(s.x, s.y) for s in stars])
    neighbours = cKDTree(positions).query(positions, k=min(9, len(stars)))[1]
    neighbours = np.asarray(neighbours).reshape(len(stars), -1)
    classes = np.empty((0, 0), np.uint8) if sky_mask is None else sky_mask.classes
    cell = 1 if sky_mask is None else sky_mask.cell_px
    signals = np.empty((len(stars), 17))

    def measure(start):
        check_cancel(cancel)
        stop = min(len(stars), start + 256)
        return start, _measure_batch(
            frame.pixels, classes, cell, positions, neighbours, start, stop
        )

    starts = range(0, len(stars), 256)
    if len(stars) < 1024:
        batches = map(measure, starts)
        for start, values in batches:
            signals[start : start + len(values)] = values
    else:
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            for start, values in pool.map(measure, starts):
                check_cancel(cancel)
                signals[start : start + len(values)] = values
    check_cancel(cancel)
    return signals


def measure_source_signals(frame, stars, *, sky_mask=None, cancel=None):
    return _measure_sources(frame, stars, sky_mask=sky_mask, cancel=cancel)[:, 0]


def calibrate_sources(frame, stars, *, sky_mask=None, cancel=None):
    started = perf_counter()
    measured = _measure_sources(frame, stars, sky_mask=sky_mask, cancel=cancel)
    if not stars:
        return stars
    signals = measured[:, 0]
    colors = np.maximum(measured[:, 1:4], 0)
    luminance = colors @ np.array(LUMA)
    colors = np.divide(
        colors, luminance[:, None], out=np.ones_like(colors), where=luminance[:, None] > 0
    )
    colors *= signals[:, None]
    peak = float(colors.max(initial=0))
    colors *= 0.87 / peak if peak > 0 else 0
    brightness = colors @ np.array(LUMA)
    valid = (signals > 0) & (measured[:, 4] > 0)
    weights = np.full(len(stars), 0.5)
    if valid.any():
        score = (
            np.sqrt(signals[valid] / np.median(signals[valid]))
            * measured[valid, 4]
            / np.median(measured[valid, 4])
        )
        weights[valid] = score**2 / (score**2 + 16)
    result = tuple(
        replace(
            s,
            brightness=float(b),
            color=tuple(c),
            source_profile=tuple(row[4:14]) if measured_valid else (),
            source_peak=tuple(row[14:17]) if measured_valid else (0.0, 0.0, 0.0),
            soft_weight=float(w),
        )
        for s, b, c, row, w, measured_valid in zip(
            stars, brightness, colors, measured, weights, valid
        )
    )
    logger.info("Photometry: %d sources measured in %.3fs", len(stars), perf_counter() - started)
    return result
