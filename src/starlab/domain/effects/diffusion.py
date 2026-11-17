"""Linear-light soft-focus PSF with an unscattered core and two Gaussian wings."""

import math
from collections.abc import Iterator
from threading import Event
from typing import cast

import cv2
import numpy as np

from ..array_types import Bounds
from ..imaging import ImageFrame, Pixels, check_cancel
from ..sky_mask import SkyMask
from .settings import EffectParameters


def _blur(pixels: Pixels, sigma: float) -> Pixels:
    radius: int = math.ceil(3 * sigma)
    kernel: int = 2 * radius + 1
    blurred = cv2.GaussianBlur(pixels, (kernel, kernel), sigma, borderType=cv2.BORDER_REFLECT)
    return cast(Pixels, blurred.reshape(pixels.shape))


def _diffuse_band(
    frame: ImageFrame, bounds: Bounds, stride: int, sigma: float, sky_mask: SkyMask | None
) -> tuple[Pixels, Pixels]:
    left, top, right, bottom = bounds
    pad: int = math.ceil(3 * 4 * sigma)
    x: int = max(0, left - pad)
    y: int = max(0, top - pad)
    end_x: int = min(frame.width, right + pad)
    end_y: int = min(frame.height, bottom + pad)
    patch: Pixels = frame.pixels[y:end_y, x:end_x]
    rows = slice(top - y, bottom - y, stride)
    cols = slice(left - x, right - x, stride)
    original: Pixels = patch[rows, cols]
    mask: Pixels | None = None
    if sky_mask is not None:
        allowed = sky_mask.region((x, y, end_x, end_y))
        if not allowed[rows, cols].any():
            empty = np.zeros_like(original)
            return empty, empty
        if not allowed.all():
            mask = allowed.astype(np.float32)[..., None]
            patch = patch * mask

    def wing(width: float) -> Pixels:
        incoming: Pixels = _blur(patch, width)[rows, cols]
        # Only exchange light between eligible pixels, in both directions.
        outgoing: Pixels = original if mask is None else original * _blur(mask, width)[rows, cols]
        delta = incoming - outgoing
        if mask is not None:
            delta *= mask[rows, cols]
        return delta

    return wing(sigma), wing(4 * sigma)


def diffusion_bands(
    frame: ImageFrame,
    size: float,
    region: Bounds,
    sampling: int,
    sky_mask: SkyMask | None,
    cancel: Event | None,
) -> Iterator[tuple[slice, Pixels, Pixels]]:
    """Yield narrow and wide light transfers on the same native-aligned sample grid."""
    left, top, right, bottom = region
    sigma: float = max(1.5, max(frame.width, frame.height) / 1400) * math.sqrt(size / 12)
    band_rows: int = max(1, 192 // sampling)
    for row in range(0, len(range(top, bottom, sampling)), band_rows):
        check_cancel(cancel)
        start: int = top + row * sampling
        end: int = min(bottom, start + band_rows * sampling)
        narrow, wide = _diffuse_band(frame, (left, start, right, end), sampling, sigma, sky_mask)
        yield slice(row, row + band_rows), narrow, wide
    check_cancel(cancel)


def mix_diffusion(narrow: Pixels, wide: Pixels, parameters: EffectParameters) -> Pixels:
    """Split the selected scattering strength between the two normalized wings."""
    fraction = parameters.wide_halo_share / 100
    result = (1 - fraction) * narrow + fraction * wide
    result *= 0.55 * parameters.diffusion / 100
    return result


def optical_diffusion(
    frame: ImageFrame,
    parameters: EffectParameters,
    region: Bounds,
    sampling: int,
    sky_mask: SkyMask | None,
    cancel: Event | None,
) -> Pixels | None:
    """Mix each band as it is rendered to bound export memory use."""
    if not parameters.enabled or not parameters.diffusion:
        return None
    left, top, right, bottom = region
    result = np.empty_like(frame.pixels[top:bottom:sampling, left:right:sampling])
    for rows, narrow, wide in diffusion_bands(
        frame, parameters.size, region, sampling, sky_mask, cancel
    ):
        result[rows] = mix_diffusion(narrow, wide, parameters)
    return result
