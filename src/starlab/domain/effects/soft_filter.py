"""Source-pixel prefilter: small-star reduction and large-star wing expansion."""

import math

import numpy as np
from numba import njit

from ..imaging import check_cancel


def reduction_records(stars):
    return np.asarray(
        [
            (s.x, s.y, s.soft_weight, *s.source_profile)
            for s in stars
            if s.visible and s.source_profile and s.soft_weight < 0.5
        ],
        dtype=np.float64,
    ).reshape(-1, 13)


@njit(cache=True, nogil=True)
def _raster(pixels, records, amount, origin_x, origin_y, sampling, output):
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


def star_reduction(
    pixels, records, strength, *, origin=(0, 0), sampling=1, sky_mask=None, cancel=None
):
    result = np.zeros_like(pixels)
    if not strength:
        return result
    for start in range(0, len(records), 256):
        check_cancel(cancel)
        _raster(pixels, records[start : start + 256], strength / 100, *origin, sampling, result)
    if sky_mask is not None:
        left, top = origin
        result *= sky_mask.region(
            (left, top, left + pixels.shape[1] * sampling, top + pixels.shape[0] * sampling),
            sampling,
        )[..., None]
    check_cancel(cancel)
    return result


def enlargement_records(stars):
    return np.asarray(
        [
            (s.x, s.y, s.soft_weight, s.source_profile[0], *s.source_peak)
            for s in stars
            if s.visible and s.source_profile and s.soft_weight > 0.5 and any(s.source_peak)
        ],
        dtype=np.float64,
    ).reshape(-1, 7)


@njit(cache=True, nogil=True)
def _expand(records, amount, origin_x, origin_y, sampling, output):
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
    pixels, records, strength, *, origin=(0, 0), sampling=1, sky_mask=None, cancel=None
):
    result = np.zeros_like(pixels)
    if not strength:
        return result
    for start in range(0, len(records), 256):
        check_cancel(cancel)
        _expand(records[start : start + 256], strength / 100, *origin, sampling, result)
    if sky_mask is not None:
        left, top = origin
        result *= sky_mask.region(
            (left, top, left + pixels.shape[1] * sampling, top + pixels.shape[0] * sampling),
            sampling,
        )[..., None]
    check_cancel(cancel)
    return result
