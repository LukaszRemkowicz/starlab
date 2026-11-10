"""Source-pixel prefilter: small-star reduction and large-star wing expansion."""

import math
from threading import Event
from typing import TypedDict

import numpy as np

from ..array_types import DoubleArray
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


def reduction_records(stars: tuple[Star, ...]) -> DoubleArray:
    """Pack 13 columns: x/y, weight, FWHM, RGB background and x/y gradients."""
    return np.asarray(
        [
            (s.x, s.y, s.soft_weight, *s.source_profile)
            for s in stars
            if s.visible and s.source_profile and s.soft_weight < 0.5
        ],
        dtype=np.float64,
    ).reshape(-1, 13)


@njit(cache=True, nogil=True)
def _raster(
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
        x, y, weight, width = row[:4]
        sigma = width / 2.354820045
        cutoff = 3 * sigma
        left = max(0, math.ceil((x - cutoff - origin_x) / sampling))
        right = min(pixels.shape[1], math.floor((x + cutoff - origin_x) / sampling) + 1)
        top = max(0, math.ceil((y - cutoff - origin_y) / sampling))
        bottom = min(pixels.shape[0], math.floor((y + cutoff - origin_y) / sampling) + 1)
        for py in range(top, bottom):
            dy = origin_y + py * sampling - y
            for px in range(left, right):
                dx = origin_x + px * sampling - x
                u = (dx * dx + dy * dy) / (sigma * sigma)
                if u >= 9:
                    continue
                fraction = (
                    0.55 * amount * (1 - 2 * weight) * (1 - math.exp(-u / 2)) * (1 - u / 9) ** 2
                )
                for c in range(pixels.shape[2]):
                    background = row[4 + c] + row[7 + c] * dx + row[10 + c] * dy
                    residual = max(0, float(pixels[py, px, c]) - max(0, background))
                    output[py, px, c] = max(output[py, px, c], fraction * residual)


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
    result = np.zeros_like(pixels)
    if not strength:
        return result
    for start in range(0, len(records), 256):
        check_cancel(cancel)
        _raster(pixels, records[start : start + 256], strength / 100, *origin, sampling, result)
    _mask_filter_result(result, origin, sampling, sky_mask)
    check_cancel(cancel)
    return result


def enlargement_records(stars: tuple[Star, ...]) -> DoubleArray:
    """Pack seven columns: source x/y, weight, measured FWHM and RGB peak."""
    return np.asarray(
        [
            (s.x, s.y, s.soft_weight, s.source_profile[0], *s.source_peak)
            for s in stars
            if s.visible and s.source_profile and s.soft_weight > 0.5 and any(s.source_peak)
        ],
        dtype=np.float64,
    ).reshape(-1, 7)


@njit(cache=True, nogil=True)
def _expand(
    records: DoubleArray,
    amount: float,
    origin_x: int,
    origin_y: int,
    sampling: int,
    output: Pixels,
) -> None:
    """Write maximum measured-color wing expansion into the caller-owned output buffer."""
    for row in records:
        x, y, weight, width = row[:4]
        sigma = width / 2.354820045
        cutoff = 6 * sigma
        left = max(0, math.ceil((x - cutoff - origin_x) / sampling))
        right = min(output.shape[1], math.floor((x + cutoff - origin_x) / sampling) + 1)
        top = max(0, math.ceil((y - cutoff - origin_y) / sampling))
        bottom = min(output.shape[0], math.floor((y + cutoff - origin_y) / sampling) + 1)
        for py in range(top, bottom):
            dy = origin_y + py * sampling - y
            for px in range(left, right):
                dx = origin_x + px * sampling - x
                u = (dx * dx + dy * dy) / (sigma * sigma)
                if u >= 36:
                    continue
                wings = (math.exp(-u / 4.5) - math.exp(-u / 2)) * (1 - u / 36) ** 2
                for c in range(output.shape[2]):
                    light = amount * (2 * weight - 1) * row[4 + c] * wings
                    output[py, px, c] = max(output[py, px, c], light)


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
    result = np.zeros_like(pixels)
    if not strength:
        return result
    for start in range(0, len(records), 256):
        check_cancel(cancel)
        _expand(records[start : start + 256], strength / 100, *origin, sampling, result)
    _mask_filter_result(result, origin, sampling, sky_mask)
    check_cancel(cancel)
    return result
