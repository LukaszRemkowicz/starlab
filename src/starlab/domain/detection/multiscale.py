"""Multiscale source detection against locally estimated sky background.

This optional detector measures compact structure at several scales.
Float image allocations stay tile-sized; seeds and catalogues are global."""

import logging
import math
import time
from collections.abc import Iterator, Sequence
from threading import Event
from typing import cast

import numpy as np
from scipy import ndimage

from ..array_types import BoolArray, Bounds, Diagnostics, DoubleArray, FloatArray
from ..imaging import ImageFrame, check_cancel
from .numeric import LUMA
from .records import (
    IndexArray,
    MeasurementMask,
    MultiscaleRecord,
    PointRecord,
    Seed,
    TileKey,
    TileThreshold,
)

logger = logging.getLogger(__name__)
SCALES = (0.8, 1.6, 3.2, 6.4, 12.8)
FLOOR = 4 / 65535


def _plane(
    patch: FloatArray | DoubleArray, xx: DoubleArray, yy: DoubleArray, annulus: BoolArray
) -> tuple[DoubleArray, float]:
    """Fit a clipped annular plane without using excluded foreground pixels."""
    values = patch[annulus]
    design = np.column_stack((np.ones(values.size), xx[annulus], yy[annulus]))
    keep = np.ones(values.size, dtype=bool)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    for _ in range(3):
        residual = values - design @ coefficients
        center = np.median(residual[keep])
        noise = float(1.4826 * np.median(np.abs(residual[keep] - center)))
        if noise < 1e-8:
            break
        updated = np.abs(residual - center) <= 3.5 * noise
        if updated.sum() < 8 or np.array_equal(updated, keep):
            break
        keep = updated
        coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
    residual = values[keep] - design[keep] @ coefficients
    noise = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    return coefficients[0] + coefficients[1] * xx + coefficients[2] * yy, noise


def _measure(
    frame: ImageFrame,
    x: float,
    y: float,
    scale: float,
    sky_mask: MeasurementMask | None,
    threshold_sigma: float,
) -> MultiscaleRecord | None:
    """Measure local-background source moments, color, and morphology around a pixel-index seed."""
    sigma = math.sqrt(2) * scale
    radius = max(3.0, 3 * sigma)
    outer = math.ceil(max(8.0, 5 * sigma))
    px, py = round(x), round(y)
    left, top = max(0, px - outer), max(0, py - outer)
    right, bottom = min(frame.width, px + outer + 1), min(frame.height, py + outer + 1)
    channels = frame.pixels[top:bottom, left:right]
    luminance = channels[..., 0] if channels.shape[2] == 1 else channels @ np.asarray(LUMA)
    yy, xx = np.mgrid[top:bottom, left:right]
    xx, yy = xx - x, yy - y
    distance2 = xx**2 + yy**2
    support = (
        np.ones(luminance.shape, dtype=bool)
        if sky_mask is None
        else sky_mask.region((left, top, right, bottom))
    )
    annulus = (distance2 >= max(radius + 1, 4 * sigma) ** 2) & support
    annulus &= distance2 <= outer**2
    if annulus.sum() < 8:
        return None
    background, noise = _plane(luminance, xx, yy, annulus)
    signal = luminance - background
    core = (distance2 <= radius**2) & support
    # Positive noise otherwise biases moments outward in a faint-star aperture.
    weights = np.where(core & (signal > max(2 * noise, 1e-8)), signal, 0)
    flux = float(weights.sum())
    if flux <= 0 or np.count_nonzero(weights) < 3:
        return None
    cx, cy = float((weights * xx).sum() / flux), float((weights * yy).sum() / flux)
    dx, dy = xx - cx, yy - cy
    covariance = (
        np.array(
            [
                [(weights * dx**2).sum(), (weights * dx * dy).sum()],
                [(weights * dx * dy).sum(), (weights * dy**2).sum()],
            ]
        )
        / flux
    )
    minor, major = np.maximum(np.linalg.eigvalsh(covariance), 0)
    fwhm = float(2.354820045 * np.sqrt((minor + major) / 2))
    ellipticity = float(1 - np.sqrt(minor / max(major, 1e-12)))
    # Reject isolated pixels and elongated ridges/trails. A circular nebular
    # knot can still look like a PSF: morphology cannot establish identity.
    peak = float(np.max(signal[core], initial=0))
    if fwhm < 0.9 or ellipticity > 0.55 or peak < max(FLOOR, threshold_sigma * noise):
        return None
    if flux / peak < 1.8:
        return None
    rim = core & (distance2 >= (0.8 * radius) ** 2)
    # A ridge cut by a circular aperture can have deceptively round moments.
    # Demand radial decay in almost all directions, including its long axis.
    if rim.sum() >= 8 and np.percentile(signal[rim], 90) > 0.6 * peak:
        return None
    if math.hypot(cx, cy) > radius * 0.6:
        return None
    global_x, global_y = x + cx, y + cy
    if sky_mask is not None and not sky_mask.allowed_at(global_x, global_y):
        return None
    background_rgb = np.median(channels[annulus], axis=0)
    rgb = np.maximum(channels - background_rgb, 0)
    rgb = np.sum(rgb * (weights > 0)[..., None], axis=(0, 1))
    if rgb.size == 1:
        rgb = np.repeat(rgb, 3)
    snr = flux / (max(noise, FLOOR / 5) * math.sqrt(np.count_nonzero(core)))
    return dict(
        x=global_x + 0.5,
        y=global_y + 0.5,
        size=max(0.25, 2 * math.sqrt((minor + major) / 2)),
        rgb=tuple(float(v) for v in rgb),
        brightness=peak,
        flux=flux,
        peak=peak,
        snr=snr,
        fwhm=fwhm,
        ellipticity=ellipticity,
        saturated=bool(np.any(channels[weights > 0] >= 0.9995)),
        scale=scale,
    )


def _nearby(
    row: PointRecord,
    grid: dict[TileKey, list[MultiscaleRecord]],
    cell_size: float,
    min_distance: float,
) -> bool:
    """Search adjacent grid cells for a previous source within the suppression radius."""
    cell = (math.floor(row["x"] / cell_size), math.floor(row["y"] / cell_size))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for previous in grid.get((cell[0] + dx, cell[1] + dy), ()):
                radius = max(min_distance, 0.9 * max(row["size"], previous["size"]))
                if (row["x"] - previous["x"]) ** 2 + (row["y"] - previous["y"]) ** 2 < radius**2:
                    return True
    return False


def _compact_peak(response: FloatArray, x: float, y: float) -> bool:
    """Require curvature in both directions; a noisy ridge is only a 1D peak."""
    x, y = round(x), round(y)
    left, right = max(0, x - 1), min(response.shape[1] - 1, x + 1)
    top, bottom = max(0, y - 1), min(response.shape[0] - 1, y + 1)
    value = response[y, x]
    cross = (
        response[top, left]
        + response[bottom, right]
        - response[top, right]
        - response[bottom, left]
    ) / 4
    curvature = -np.array(
        [
            [response[y, left] + response[y, right] - 2 * value, cross],
            [cross, response[top, x] + response[bottom, x] - 2 * value],
        ]
    )
    minor, major = np.linalg.eigvalsh(curvature)
    return cast(bool, minor > 0.1 * major and major > 0)


def _component_centers(labels: IndexArray) -> Iterator[tuple[float, float]]:
    """Yield connected-component centroids in label order, retaining subpixel means."""
    for number, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        sy, sx = np.nonzero(labels[region] == number)
        py, px = sy.mean() + region[0].start, sx.mean() + region[1].start
        yield px, py


def _tile_seeds(
    frame: ImageFrame,
    bounds: Bounds,
    core_bounds: Bounds,
    scales: tuple[float, ...],
    threshold_sigma: float,
    sky_mask: MeasurementMask | None,
    cancel: Event | None,
) -> tuple[list[Seed], list[TileThreshold]] | None:
    """Find scale-space and saturated-plateau seeds for one halo/core tile."""
    left, top, right, bottom = core_bounds
    seeds: list[Seed] = []
    thresholds: list[TileThreshold] = []
    bx, by, ex, ey = bounds
    channels = frame.pixels[by:ey, bx:ex]
    luminance = (
        channels[..., 0].copy()
        if channels.shape[2] == 1
        else np.asarray(channels @ np.asarray(LUMA, dtype=np.float32), dtype=np.float32)
    )
    support = np.ones(luminance.shape, dtype=bool) if sky_mask is None else sky_mask.region(bounds)
    core = (slice(top - by, bottom - by), slice(left - bx, right - bx))
    if not support[core].any():
        return None
    if not support.all():
        # Fill from nearest allowed sky, so bright excluded foreground
        # cannot leak through a Gaussian kernel across the mask edge.
        indices = ndimage.distance_transform_edt(
            ~support, return_distances=False, return_indices=True
        )
        luminance[~support] = luminance[tuple(indices[:, ~support])]
    saturated = (channels.max(axis=2) >= 0.9995) & support
    sat_labels, _ = ndimage.label(saturated, structure=np.ones((3, 3)))
    plateaus = []
    for px, py in _component_centers(sat_labels):
        px, py = float(px), float(py)
        if left <= px + bx < right and top <= py + by < bottom:
            plateaus.append((px, py))
    previous_broad, previous_sigma = None, None
    for scale in scales:
        check_cancel(cancel)
        # Adjacent geometric scales share exactly the same smoothing.
        response = (
            previous_broad
            if scale == previous_sigma
            else ndimage.gaussian_filter(luminance, scale, mode="nearest")
        )
        previous_broad = ndimage.gaussian_filter(luminance, 2 * scale, mode="nearest")
        previous_sigma = 2 * scale
        response -= previous_broad
        sample = response[core][support[core]]
        center = float(np.median(sample))
        noise = float(1.4826 * np.median(np.abs(sample - center)))
        threshold = max(FLOOR, center + threshold_sigma * noise)
        thresholds.append((left, top, scale, threshold))
        selected = (
            (response == ndimage.maximum_filter(response, size=3, mode="nearest"))
            & (response > threshold)
            & support
            & ~saturated
        )
        labels, _ = ndimage.label(selected, structure=np.ones((3, 3)))
        for px, py in _component_centers(labels):
            if (
                left <= px + bx < right
                and top <= py + by < bottom
                and _compact_peak(response, px, py)
            ):
                score = float(response[round(py), round(px)])
                seeds.append((score, px + bx, py + by, scale))
        for px, py in plateaus:
            score = float(response[round(py), round(px)])
            if score > threshold and _compact_peak(response, px, py):
                seeds.append((score, px + bx, py + by, scale))
    return seeds, thresholds


def _measure_seeds(
    frame: ImageFrame,
    seeds: list[Seed],
    scales: tuple[float, ...],
    min_distance: float,
    threshold_sigma: float,
    sky_mask: MeasurementMask | None,
    cancel: Event | None,
) -> list[MultiscaleRecord]:
    """Measure strongest seeds first, suppress overlaps globally, then normalize colors."""
    # Strongest DoG contrast chooses a source width, rather than the highest
    # signal/noise (which can favor a very broad kernel on nearly noiseless sky).
    seeds.sort(key=lambda s: (-s[0], s[2], s[1], s[3]))
    rows: list[MultiscaleRecord] = []
    grid: dict[TileKey, list[MultiscaleRecord]] = {}
    cell_size = max(min_distance, 4 * max(scales))
    for score, x, y, scale in seeds:
        check_cancel(cancel)
        point: PointRecord = dict(x=x + 0.5, y=y + 0.5, size=0)
        if _nearby(point, grid, cell_size, min_distance):
            continue
        row = _measure(frame, x, y, scale, sky_mask, threshold_sigma)
        if row is None or _nearby(row, grid, cell_size, min_distance):
            continue
        rows.append(row)
        key = (math.floor(row["x"] / cell_size), math.floor(row["y"] / cell_size))
        grid.setdefault(key, []).append(row)
    peak = max((max(row["rgb"]) for row in rows), default=0)
    multiplier = 0.87 / peak if peak > 0 else 0
    for row in rows:
        row["normalized_rgb"] = tuple(v * multiplier for v in row["rgb"])
        row["normalized_brightness"] = float(np.dot(row["normalized_rgb"], LUMA))
    rows.sort(key=lambda row: (row["y"], row["x"]))
    return rows


def analyze(
    frame: ImageFrame,
    threshold_sigma: float = 5.0,
    min_distance: float = 2.0,
    cancel: Event | None = None,
    *,
    sky_mask: MeasurementMask | None = None,
    tile_size: int = 768,
    scales: Sequence[float] = SCALES,
) -> tuple[list[MultiscaleRecord], Diagnostics]:
    """Find compact sources at multiple widths over every image tile.

    ``sky_mask`` may supply ``region(bounds)`` (True means allowed) and
    ``allowed_at(x,y)`` in pixel-index coordinates. Masked values never enter
    response/noise or annular statistics. A sigma threshold is statistical in
    this method, unlike the legacy native proxy cutoff. There is no top-N cap.
    Largest default scale targets PSFs around FWHM 43 pixels; widths are sampled
    geometrically and sources larger than this range need explicit scales.
    """
    check_cancel(cancel)
    scales = tuple(sorted(set(scales)))
    if (
        not isinstance(tile_size, int)
        or tile_size < 32
        or not scales
        or any(not math.isfinite(s) or s <= 0 for s in scales)
        or not math.isfinite(threshold_sigma)
        or threshold_sigma <= 0
        or not math.isfinite(min_distance)
        or min_distance <= 0
    ):
        raise ValueError("Use tile_size >=32 and finite positive scales, sigma and distance.")
    started = time.monotonic()
    halo = math.ceil(max(8 * max(scales), 5 * math.sqrt(2) * max(scales))) + 2
    seeds: list[Seed] = []
    thresholds: list[TileThreshold] = []
    for top in range(0, frame.height, tile_size):
        for left in range(0, frame.width, tile_size):
            check_cancel(cancel)
            right, bottom = min(frame.width, left + tile_size), min(frame.height, top + tile_size)
            bounds = (
                max(0, left - halo),
                max(0, top - halo),
                min(frame.width, right + halo),
                min(frame.height, bottom + halo),
            )
            found = _tile_seeds(
                frame, bounds, (left, top, right, bottom), scales, threshold_sigma, sky_mask, cancel
            )
            if found is None:
                continue
            local_seeds, local_thresholds = found
            seeds.extend(local_seeds)
            thresholds.extend(local_thresholds)
            logger.info("Multiscale detector tile (%d,%d): %d total seeds", left, top, len(seeds))
    rows = _measure_seeds(frame, seeds, scales, min_distance, threshold_sigma, sky_mask, cancel)
    return rows, dict(
        method="Multiscale local-background DoG (Stars-Fill inspired; experimental)",
        count=len(rows),
        candidate_seeds=len(seeds),
        seconds=time.monotonic() - started,
        scales=list(scales),
        tile_size=tile_size,
        halo=halo,
        threshold_sigma=threshold_sigma,
        threshold_floor=FLOOR,
        tile_thresholds=thresholds,
        sky_mask=sky_mask is not None,
    )
