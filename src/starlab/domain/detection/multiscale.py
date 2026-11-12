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
from numpy.typing import NDArray
from scipy import ndimage

from ..array_types import (
    RGBA,
    BoolArray,
    Bounds,
    Diagnostics,
    DoubleArray,
    FloatArray,
    IntArray,
    Position,
)
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
    values: FloatArray | DoubleArray = patch[annulus]
    design: DoubleArray = np.column_stack((np.ones(values.size), xx[annulus], yy[annulus]))
    keep: BoolArray = np.ones(values.size, dtype=bool)
    coefficients: DoubleArray = np.linalg.lstsq(design, values, rcond=None)[0]
    for _ in range(3):
        residual: FloatArray | DoubleArray = values - design @ coefficients
        center: np.float32 | np.float64 = np.median(residual[keep])
        noise: float = float(1.4826 * np.median(np.abs(residual[keep] - center)))
        if noise < 1e-8:
            break
        updated: BoolArray = np.abs(residual - center) <= 3.5 * noise
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
    sigma: float = math.sqrt(2) * scale
    radius: float = max(3.0, 3 * sigma)
    outer: int = math.ceil(max(8.0, 5 * sigma))
    peak_x, peak_y = round(x), round(y)
    left, top = max(0, peak_x - outer), max(0, peak_y - outer)
    right, bottom = min(frame.width, peak_x + outer + 1), min(frame.height, peak_y + outer + 1)
    channels: FloatArray = frame.pixels[top:bottom, left:right]
    luminance: FloatArray | DoubleArray = (
        channels[..., 0] if channels.shape[2] == 1 else channels @ np.asarray(LUMA)
    )
    grid_x: IntArray
    grid_y: IntArray
    grid_y, grid_x = np.mgrid[top:bottom, left:right]
    offset_x: DoubleArray = np.subtract(grid_x, x)
    offset_y: DoubleArray = np.subtract(grid_y, y)
    distance2: DoubleArray = offset_x**2 + offset_y**2
    support: BoolArray = (
        np.ones(luminance.shape, dtype=bool)
        if sky_mask is None
        else sky_mask.region((left, top, right, bottom))
    )
    annulus: BoolArray = (distance2 >= max(radius + 1, 4 * sigma) ** 2) & support
    annulus &= distance2 <= outer**2
    if annulus.sum() < 8:
        return None
    background, noise = _plane(luminance, offset_x, offset_y, annulus)
    signal: DoubleArray = luminance - background
    core: BoolArray = (distance2 <= radius**2) & support
    # Positive noise otherwise biases moments outward in a faint-star aperture.
    weights: DoubleArray = np.where(core & (signal > max(2 * noise, 1e-8)), signal, 0)
    flux: float = float(weights.sum())
    if flux <= 0 or np.count_nonzero(weights) < 3:
        return None
    centroid_x, centroid_y = (
        float((weights * offset_x).sum() / flux),
        float((weights * offset_y).sum() / flux),
    )
    centered_x, centered_y = offset_x - centroid_x, offset_y - centroid_y
    covariance: DoubleArray = (
        np.array(
            [
                [(weights * centered_x**2).sum(), (weights * centered_x * centered_y).sum()],
                [(weights * centered_x * centered_y).sum(), (weights * centered_y**2).sum()],
            ]
        )
        / flux
    )
    minor, major = np.maximum(np.linalg.eigvalsh(covariance), 0)
    fwhm: float = float(2.354820045 * np.sqrt((minor + major) / 2))
    ellipticity: float = float(1 - np.sqrt(minor / max(major, 1e-12)))
    # Reject isolated pixels and elongated ridges/trails. A circular nebular
    # knot can still look like a PSF: morphology cannot establish identity.
    peak: float = float(np.max(signal[core], initial=0))
    if fwhm < 0.9 or ellipticity > 0.55 or peak < max(FLOOR, threshold_sigma * noise):
        return None
    if flux / peak < 1.8:
        return None
    rim: BoolArray = core & (distance2 >= (0.8 * radius) ** 2)
    # A ridge cut by a circular aperture can have deceptively round moments.
    # Demand radial decay in almost all directions, including its long axis.
    if rim.sum() >= 8 and np.percentile(signal[rim], 90) > 0.6 * peak:
        return None
    if math.hypot(centroid_x, centroid_y) > radius * 0.6:
        return None
    global_x, global_y = x + centroid_x, y + centroid_y
    if sky_mask is not None and not sky_mask.allowed_at(global_x, global_y):
        return None
    background_rgb: FloatArray = np.median(channels[annulus], axis=0)
    rgb: FloatArray = np.maximum(channels - background_rgb, 0)
    rgb = np.sum(rgb * (weights > 0)[..., None], axis=(0, 1))
    if rgb.size == 1:
        rgb = np.repeat(rgb, 3)
    snr: float = flux / (max(noise, FLOOR / 5) * math.sqrt(np.count_nonzero(core)))
    return MultiscaleRecord(
        x=global_x + 0.5,
        y=global_y + 0.5,
        size=max(0.25, 2 * math.sqrt((minor + major) / 2)),
        rgb=tuple(float(channel) for channel in rgb),
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
    cell: tuple[int, int] = (math.floor(row["x"] / cell_size), math.floor(row["y"] / cell_size))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for previous in grid.get((cell[0] + dx, cell[1] + dy), ()):
                radius: float = max(min_distance, 0.9 * max(row["size"], previous["size"]))
                if (row["x"] - previous["x"]) ** 2 + (row["y"] - previous["y"]) ** 2 < radius**2:
                    return True
    return False


def _compact_peak(response: FloatArray, x: float, y: float) -> bool:
    """Require curvature in both directions; a noisy ridge is only a 1D peak."""
    x, y = round(x), round(y)
    left, right = max(0, x - 1), min(response.shape[1] - 1, x + 1)
    top, bottom = max(0, y - 1), min(response.shape[0] - 1, y + 1)
    value: np.float32 = response[y, x]
    cross: np.float32 = (
        response[top, left]
        + response[bottom, right]
        - response[top, right]
        - response[bottom, left]
    ) / 4
    curvature: FloatArray = -np.array(
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
        component_x: NDArray[np.intp]
        component_y: NDArray[np.intp]
        component_y, component_x = np.nonzero(labels[region] == number)
        center_y, center_x = (
            component_y.mean() + region[0].start,
            component_x.mean() + region[1].start,
        )
        yield center_x, center_y


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
    bounds_left, bounds_top, bounds_right, bounds_bottom = bounds
    channels: FloatArray = frame.pixels[bounds_top:bounds_bottom, bounds_left:bounds_right]
    luminance: FloatArray = (
        channels[..., 0].copy()
        if channels.shape[2] == 1
        else np.asarray(channels @ np.asarray(LUMA, dtype=np.float32), dtype=np.float32)
    )
    support: BoolArray = (
        np.ones(luminance.shape, dtype=bool) if sky_mask is None else sky_mask.region(bounds)
    )
    core: tuple[slice, slice] = (
        slice(top - bounds_top, bottom - bounds_top),
        slice(left - bounds_left, right - bounds_left),
    )
    if not support[core].any():
        return None
    if not support.all():
        # Fill from nearest allowed sky, so bright excluded foreground
        # cannot leak through a Gaussian kernel across the mask edge.
        indices: IndexArray = ndimage.distance_transform_edt(
            ~support, return_distances=False, return_indices=True
        )
        luminance[~support] = luminance[tuple(indices[:, ~support])]
    saturated: BoolArray = (channels.max(axis=2) >= 0.9995) & support
    saturation_labels: IndexArray
    saturation_labels, _ = ndimage.label(saturated, structure=np.ones((3, 3)))
    plateaus: list[Position] = []
    for peak_x, peak_y in _component_centers(saturation_labels):
        peak_x, peak_y = float(peak_x), float(peak_y)
        if left <= peak_x + bounds_left < right and top <= peak_y + bounds_top < bottom:
            plateaus.append((peak_x, peak_y))
    previous_sigma: float | None
    previous_broad: FloatArray | None
    previous_broad, previous_sigma = None, None
    for scale in scales:
        check_cancel(cancel)
        # Adjacent geometric scales share exactly the same smoothing.
        response: FloatArray = (
            previous_broad
            if scale == previous_sigma and previous_broad is not None
            else ndimage.gaussian_filter(luminance, scale, mode="nearest")
        )
        broad_response: FloatArray = ndimage.gaussian_filter(luminance, 2 * scale, mode="nearest")
        previous_broad = broad_response
        previous_sigma = 2 * scale
        response -= previous_broad
        response_samples: FloatArray = response[core][support[core]]
        center: float = float(np.median(response_samples))
        noise: float = float(1.4826 * np.median(np.abs(response_samples - center)))
        threshold: float = max(FLOOR, center + threshold_sigma * noise)
        thresholds.append((left, top, scale, threshold))
        selected: BoolArray = (
            (response == ndimage.maximum_filter(response, size=3, mode="nearest"))
            & (response > threshold)
            & support
            & ~saturated
        )
        labels: IndexArray
        labels, _ = ndimage.label(selected, structure=np.ones((3, 3)))
        for peak_x, peak_y in _component_centers(labels):
            if (
                left <= peak_x + bounds_left < right
                and top <= peak_y + bounds_top < bottom
                and _compact_peak(response, peak_x, peak_y)
            ):
                score: float = float(response[round(peak_y), round(peak_x)])
                seeds.append((score, peak_x + bounds_left, peak_y + bounds_top, scale))
        for peak_x, peak_y in plateaus:
            score = float(response[round(peak_y), round(peak_x)])
            if score > threshold and _compact_peak(response, peak_x, peak_y):
                seeds.append((score, peak_x + bounds_left, peak_y + bounds_top, scale))
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
    seeds.sort(key=lambda seed: (-seed[0], seed[2], seed[1], seed[3]))
    rows: list[MultiscaleRecord] = []
    grid: dict[TileKey, list[MultiscaleRecord]] = {}
    cell_size: float = max(min_distance, 4 * max(scales))
    for score, x, y, scale in seeds:
        check_cancel(cancel)
        point: PointRecord = dict(x=x + 0.5, y=y + 0.5, size=0)
        if _nearby(point, grid, cell_size, min_distance):
            continue
        row: MultiscaleRecord | None = _measure(frame, x, y, scale, sky_mask, threshold_sigma)
        if row is None or _nearby(row, grid, cell_size, min_distance):
            continue
        rows.append(row)
        key: tuple[int, int] = (math.floor(row["x"] / cell_size), math.floor(row["y"] / cell_size))
        grid.setdefault(key, []).append(row)
    peak: float = max((max(row["rgb"]) for row in rows), default=0)
    multiplier: float = 0.87 / peak if peak > 0 else 0
    for row in rows:
        row["normalized_rgb"] = tuple(channel * multiplier for channel in row["rgb"])
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
        or any(not math.isfinite(scale) or scale <= 0 for scale in scales)
        or not math.isfinite(threshold_sigma)
        or threshold_sigma <= 0
        or not math.isfinite(min_distance)
        or min_distance <= 0
    ):
        raise ValueError("Use tile_size >=32 and finite positive scales, sigma and distance.")
    started: float = time.monotonic()
    halo: int = math.ceil(max(8 * max(scales), 5 * math.sqrt(2) * max(scales))) + 2
    seeds: list[Seed] = []
    thresholds: list[TileThreshold] = []
    for top in range(0, frame.height, tile_size):
        for left in range(0, frame.width, tile_size):
            check_cancel(cancel)
            right, bottom = min(frame.width, left + tile_size), min(frame.height, top + tile_size)
            bounds: Bounds = (
                max(0, left - halo),
                max(0, top - halo),
                min(frame.width, right + halo),
                min(frame.height, bottom + halo),
            )
            found: tuple[list[RGBA], list[tuple[int, int, float, float]]] | None = _tile_seeds(
                frame, bounds, (left, top, right, bottom), scales, threshold_sigma, sky_mask, cancel
            )
            if found is None:
                continue
            local_seeds, local_thresholds = found
            seeds.extend(local_seeds)
            thresholds.extend(local_thresholds)
            logger.info("Multiscale detector tile (%d,%d): %d total seeds", left, top, len(seeds))
    rows: list[MultiscaleRecord] = _measure_seeds(
        frame, seeds, scales, min_distance, threshold_sigma, sky_mask, cancel
    )
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
