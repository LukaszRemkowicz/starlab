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
from .profiles import source_colors
from .settings import EffectParameters


class FilterOptions(TypedDict):
    """Coordinate mapping and cancellation shared by source-correction passes."""

    origin: tuple[int, int]
    sampling: int
    sky_mask: SkyMask | None
    cancel: Event | None


def soft_filter_sources(stars: tuple[Star, ...], parameters: EffectParameters) -> tuple[Star, ...]:
    """Keep the stars the Soft Flare Quantity selects, with the glow's own brightness threshold."""
    maximum: float = max((star.brightness for star in stars), default=0.0)
    threshold: float = (1 - parameters.quantity / 100) ** 2 * maximum
    return tuple(star for star in stars if star.brightness > threshold)


def halo_shape(size: float, falloff: float) -> tuple[float, float]:
    """Return the halo width factor k (Size) and super-Gaussian exponent g (Edge falloff)."""
    return math.sqrt(size / 12), 2 ** ((60 - falloff) / 60)


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
                    0.9
                    * amount
                    * math.sqrt(1 - 2 * weight)
                    * (1 - math.exp(-squared_sigma_distance * 1.4))
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


def enlargement_records(
    stars: tuple[Star, ...], hue: float = 0, saturation: float = 0
) -> DoubleArray:
    """Pack seven columns: source x/y, weight, measured FWHM and the halo RGB.

    The halo color is the measured peak with the glow's Hue and Saturation applied.
    """
    records: DoubleArray = np.asarray(
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
    if len(records) and (hue or saturation):
        records[:, 4:] = source_colors(records[:, 4:], hue, saturation)
    return records


@njit(cache=True, nogil=True)
def _raster_enlargement(
    records: DoubleArray,
    amount: float,
    size_factor: float,
    shape: float,
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
        # Bigger stars get a wider halo; Size scales the growth; shape g makes edges sharper or softer.
        spread: float = 1 + (0.5 + 2.5 * (2 * weight - 1)) * size_factor
        limit: float = 9 ** (1 / shape) * spread * spread
        cutoff: float = math.sqrt(limit) * sigma
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
                if squared_sigma_distance >= limit:
                    continue
                halo: float = math.exp(-0.5 * (squared_sigma_distance / (spread * spread)) ** shape)
                wings: float = (
                    max(0.0, halo - math.exp(-squared_sigma_distance / 2))
                    * (1 - squared_sigma_distance / limit) ** 2
                )
                for channel in range(output.shape[2]):
                    light: float = (
                        0.8 * amount * math.sqrt(2 * weight - 1) * row[4 + channel] * wings
                    )
                    output[pixel_y, pixel_x, channel] = max(
                        output[pixel_y, pixel_x, channel], light
                    )


def star_enlargement(
    pixels: Pixels,
    records: DoubleArray,
    strength: float,
    *,
    size: float = 12,
    falloff: float = 60,
    origin: tuple[int, int] = (0, 0),
    sampling: int = 1,
    sky_mask: SkyMask | None = None,
    cancel: Event | None = None,
) -> Pixels:
    """Return new measured-color wing light; preserve centers and mask foreground pixels."""
    result: Pixels = np.zeros_like(pixels)
    if not strength:
        return result
    shape: tuple[float, float] = halo_shape(size, falloff)
    for start in range(0, len(records), 256):
        check_cancel(cancel)
        _raster_enlargement(
            records[start : start + 256], strength / 100, *shape, *origin, sampling, result
        )
    _mask_filter_result(result, origin, sampling, sky_mask)
    check_cancel(cancel)
    return result
