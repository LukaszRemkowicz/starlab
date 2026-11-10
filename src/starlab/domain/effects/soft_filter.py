"""Source-pixel prefilter: small-star reduction and large-star wing expansion."""

import math
from threading import Event
from typing import TypedDict

import numpy as np

from ..array_types import Bounds, DoubleArray, FloatArray
from ..imaging import Pixels, check_cancel
from ..jit import njit
from ..sky_mask import SkyMask
from ..stars import Star


class FilterOptions(TypedDict):
    """Coordinate mapping and cancellation shared by source-correction passes."""

    origin: tuple[int, int]
    sampling: int
    sky_mask: SkyMask | None
    cancel: Event | None


def soft_halo_factors(stars: tuple[Star, ...], strength: float) -> FloatArray:
    """Bias halo light toward large measured sources; unmeasured sources remain neutral."""
    weights: FloatArray = np.array(
        [star.soft_weight if star.source_profile else 0.5 for star in stars], dtype=np.float32
    )
    amount: float = strength / 100
    return 1 + amount * (2 * weights - 1)


def reduction_records(stars: tuple[Star, ...]) -> DoubleArray:
    """Pack 13 columns: x/y, weight, FWHM, RGB background and x/y gradients."""
    return np.asarray(
        [
            (star.x, star.y, star.soft_weight, *star.source_profile)
            for star in stars
            if star.visible and star.source_profile and star.soft_weight < 0.5
        ],
        dtype=np.float64,
    ).reshape(-1, 13)


@njit(cache=True, nogil=True)
def _source_bounds(
    x: float,
    y: float,
    cutoff: float,
    origin_x: int,
    origin_y: int,
    sampling: int,
    width: int,
    height: int,
) -> Bounds:
    """Clip a source's full-image support to the sampled output grid."""
    left: int = max(0, math.ceil((x - cutoff - origin_x) / sampling))
    right: int = min(width, math.floor((x + cutoff - origin_x) / sampling) + 1)
    top: int = max(0, math.ceil((y - cutoff - origin_y) / sampling))
    bottom: int = min(height, math.floor((y + cutoff - origin_y) / sampling) + 1)
    return left, top, right, bottom


@njit(cache=True, nogil=True)
def _raster_reduction(
    pixels: Pixels,
    records: DoubleArray,
    amount: float,
    origin_x: int,
    origin_y: int,
    sampling: int,
    output: Pixels,
) -> None:
    """Write maximum bounded wing subtraction into the caller-owned output buffer."""
    for row in records:
        weight: np.float64
        width: np.float64
        x: np.float64
        y: np.float64
        x, y, weight, width = row[:4]
        sigma: float = width / 2.354820045
        cutoff: float = 3 * sigma
        left, top, right, bottom = _source_bounds(
            x, y, cutoff, origin_x, origin_y, sampling, pixels.shape[1], pixels.shape[0]
        )
        for pixel_y in range(top, bottom):
            delta_y: float = origin_y + pixel_y * sampling - y
            for pixel_x in range(left, right):
                delta_x: float = origin_x + pixel_x * sampling - x
                squared_sigma_distance: float = (delta_x * delta_x + delta_y * delta_y) / (
                    sigma * sigma
                )
                if squared_sigma_distance >= 9:
                    continue
                fraction: float = (
                    0.55
                    * amount
                    * (1 - 2 * weight)
                    * (1 - math.exp(-squared_sigma_distance / 2))
                    * (1 - squared_sigma_distance / 9) ** 2
                )
                for channel in range(pixels.shape[2]):
                    background: float = (
                        row[4 + channel] + row[7 + channel] * delta_x + row[10 + channel] * delta_y
                    )
                    residual: float = max(
                        0, float(pixels[pixel_y, pixel_x, channel]) - max(0, background)
                    )
                    output[pixel_y, pixel_x, channel] = max(
                        output[pixel_y, pixel_x, channel], fraction * residual
                    )


def _mask_filter_result(
    result: Pixels, origin: tuple[int, int], sampling: int, sky_mask: SkyMask | None
) -> None:
    """Restrict a source-correction buffer to sky on its sampled source grid."""
    if sky_mask is not None:
        left, top = origin
        result *= sky_mask.region(
            (left, top, left + result.shape[1] * sampling, top + result.shape[0] * sampling),
            sampling,
        )[..., None]


def star_reduction(
    pixels: Pixels,
    records: DoubleArray,
    strength: float,
    *,
    origin: tuple[int, int] = (0, 0),
    sampling: int = 1,
    sky_mask: SkyMask | None = None,
    cancel: Event | None = None,
) -> Pixels:
    """Return a new bounded wing-subtraction buffer; overlapping corrections use maximum."""
    result: Pixels = np.zeros_like(pixels)
    if not strength:
        return result
    for start in range(0, len(records), 256):
        check_cancel(cancel)
        _raster_reduction(
            pixels, records[start : start + 256], strength / 100, *origin, sampling, result
        )
    _mask_filter_result(result, origin, sampling, sky_mask)
    check_cancel(cancel)
    return result


def enlargement_records(stars: tuple[Star, ...]) -> DoubleArray:
    """Pack seven columns: source x/y, weight, measured FWHM and RGB peak."""
    return np.asarray(
        [
            (star.x, star.y, star.soft_weight, star.source_profile[0], *star.source_peak)
            for star in stars
            if star.visible
            and star.source_profile
            and star.soft_weight > 0.5
            and any(star.source_peak)
        ],
        dtype=np.float64,
    ).reshape(-1, 7)


@njit(cache=True, nogil=True)
def _raster_enlargement(
    records: DoubleArray,
    amount: float,
    origin_x: int,
    origin_y: int,
    sampling: int,
    output: Pixels,
) -> None:
    """Write maximum measured-color wing expansion into the caller-owned output buffer."""
    for row in records:
        weight: np.float64
        width: np.float64
        x: np.float64
        y: np.float64
        x, y, weight, width = row[:4]
        sigma: float = width / 2.354820045
        cutoff: float = 6 * sigma
        left, top, right, bottom = _source_bounds(
            x, y, cutoff, origin_x, origin_y, sampling, output.shape[1], output.shape[0]
        )
        for pixel_y in range(top, bottom):
            delta_y: float = origin_y + pixel_y * sampling - y
            for pixel_x in range(left, right):
                delta_x: float = origin_x + pixel_x * sampling - x
                squared_sigma_distance: float = (delta_x * delta_x + delta_y * delta_y) / (
                    sigma * sigma
                )
                if squared_sigma_distance >= 36:
                    continue
                wings: float = (
                    math.exp(-squared_sigma_distance / 4.5) - math.exp(-squared_sigma_distance / 2)
                ) * (1 - squared_sigma_distance / 36) ** 2
                for channel in range(output.shape[2]):
                    light: float = amount * (2 * weight - 1) * row[4 + channel] * wings
                    output[pixel_y, pixel_x, channel] = max(
                        output[pixel_y, pixel_x, channel], light
                    )


def star_enlargement(
    pixels: Pixels,
    records: DoubleArray,
    strength: float,
    *,
    origin: tuple[int, int] = (0, 0),
    sampling: int = 1,
    sky_mask: SkyMask | None = None,
    cancel: Event | None = None,
) -> Pixels:
    """Return new measured-color wing light; preserve centers and mask foreground pixels."""
    result: Pixels = np.zeros_like(pixels)
    if not strength:
        return result
    for start in range(0, len(records), 256):
        check_cancel(cancel)
        _raster_enlargement(records[start : start + 256], strength / 100, *origin, sampling, result)
    _mask_filter_result(result, origin, sampling, sky_mask)
    check_cancel(cancel)
    return result
