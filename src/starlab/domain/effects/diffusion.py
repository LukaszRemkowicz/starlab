"""Linear-light soft-focus PSF with an unscattered core and two Gaussian wings."""

import math
import weakref
from collections.abc import Iterator
from threading import Event
from typing import cast

import cv2
import numpy as np

from ..array_types import Bounds
from ..imaging import ImageFrame, Pixels, check_cancel
from ..sky_mask import FOREGROUND, SkyMask
from .profiles import ArrayCache
from .settings import EffectParameters

_PREVIEW_WINGS = ArrayCache[tuple[Pixels, Pixels]](64)


def _sigma(frame: ImageFrame, size: float) -> float:
    return max(1.5, max(frame.width, frame.height) / 1400) * math.sqrt(size / 12)


def _blur(pixels: Pixels, sigma: float, rows: slice, cols: slice) -> Pixels:
    radius: int = math.ceil(3 * sigma)
    kernel: int = 2 * radius + 1
    if rows.step < 4:
        # The narrow wing only needs its own halo inside the shared wide patch.
        top = max(0, rows.start - radius)
        patch = pixels[top : rows.stop + radius]
        blurred = cv2.GaussianBlur(patch, (kernel, kernel), sigma, borderType=cv2.BORDER_REFLECT)
        return cast(
            Pixels,
            blurred.reshape(patch.shape)[rows.start - top : rows.stop - top : rows.step, cols],
        )

    # A zoomed-out preview only needs selected output rows. Apply the same
    # separable Gaussian to those rows, retaining every contributing input pixel.
    weights = cv2.getGaussianKernel(kernel, sigma, cv2.CV_32F).ravel()
    height, width, channels = pixels.shape
    centers = np.arange(rows.start, rows.stop, rows.step)
    taps = (centers[:, None] + np.arange(-radius, radius + 1)) % (2 * height)
    taps = np.minimum(taps, 2 * height - 1 - taps)
    start, stop = int(taps.min()), int(taps.max()) + 1
    matrix = np.zeros((len(centers), stop - start), np.float32)
    np.add.at(matrix, (np.arange(len(centers))[:, None], taps - start), weights)
    vertical = (matrix @ pixels[start:stop].reshape(stop - start, width * channels)).reshape(
        len(centers), width, channels
    )
    # Reflect only at the patch edge. Its halo covers the entire native kernel.
    planar = np.ascontiguousarray(vertical.transpose(0, 2, 1))
    padded = np.pad(planar, ((0, 0), (0, 0), (radius, radius)), mode="symmetric")
    neighbors = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=2)[:, :, cols]
    return cast(Pixels, np.einsum("ycxk,k->ycx", neighbors, weights).transpose(0, 2, 1))


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
        cell = sky_mask.cell_px
        cells = sky_mask.classes[
            y // cell : (end_y - 1) // cell + 1, x // cell : (end_x - 1) // cell + 1
        ]
        if cells.max() == FOREGROUND:
            empty = np.zeros_like(original)
            return empty, empty
        if cells.min() == FOREGROUND:
            allowed = sky_mask.region((x, y, end_x, end_y))
            if not allowed[rows, cols].any():
                empty = np.zeros_like(original)
                return empty, empty
            mask = allowed.astype(np.float32)[..., None]
            patch = cast(Pixels, cv2.copyTo(patch, allowed.view(np.uint8)).reshape(patch.shape))

    def wing(width: float) -> Pixels:
        incoming: Pixels = _blur(patch, width, rows, cols)
        # Only exchange light between eligible pixels, in both directions.
        outgoing: Pixels = original if mask is None else original * _blur(mask, width, rows, cols)
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
    sigma = _sigma(frame, size)
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


def preview_wings(
    frame: ImageFrame,
    size: float,
    region: Bounds,
    sampling: int,
    sky_mask: SkyMask | None,
    cancel: Event | None,
) -> tuple[Pixels, Pixels]:
    """Collect both unit responses for a preview that can be remixed interactively."""
    left, top, right, bottom = region
    narrow = np.empty_like(frame.pixels[top:bottom:sampling, left:right:sampling])
    wide = np.empty_like(narrow)
    for rows, near_band, far_band in diffusion_bands(
        frame, size, region, sampling, sky_mask, cancel
    ):
        narrow[rows], wide[rows] = near_band, far_band
    return narrow, wide


def preview_diffusion(
    frame: ImageFrame,
    parameters: EffectParameters,
    region: Bounds,
    sampling: int,
    sky_mask: SkyMask | None,
    cancel: Event | None,
) -> Pixels | None:
    """Reuse native responses on CPU without retaining the source frame or sky mask."""
    if not parameters.enabled or not parameters.diffusion:
        return None
    check_cancel(cancel)
    left, top, right, bottom = region
    if 2 * frame.pixels[top:bottom:sampling, left:right:sampling].nbytes > _PREVIEW_WINGS.limit:
        return optical_diffusion(frame, parameters, region, sampling, sky_mask, cancel)
    key = (
        weakref.ref(frame),
        region,
        sampling,
        parameters.size,
        weakref.ref(sky_mask) if sky_mask is not None else None,
    )
    wings = _PREVIEW_WINGS.get(key)
    if wings is None:
        wings = _PREVIEW_WINGS.put(
            key, preview_wings(frame, parameters.size, region, sampling, sky_mask, cancel)
        )
    return mix_diffusion(*wings, parameters)


def export_diffusion_bands(
    frame: ImageFrame,
    parameters: EffectParameters,
    band_rows: int,
    sky_mask: SkyMask | None,
    cancel: Event | None,
) -> Iterator[Pixels]:
    """Share each halo across consecutive writer bands, retaining one bounded correction."""
    sigma = _sigma(frame, parameters.size)
    pad = math.ceil(12 * sigma)
    # Amortize the two halos while limiting the retained correction to 128 MiB
    # (or one writer band for exceptionally wide images).
    per_band = frame.width * frame.pixels.shape[2] * frame.pixels.itemsize * band_rows
    group = max(1, min(math.ceil(4 * pad / band_rows), (128 << 20) // per_band))
    for top in range(0, frame.height, group * band_rows):
        check_cancel(cancel)
        bottom = min(frame.height, top + group * band_rows)
        narrow, wide = _diffuse_band(frame, (0, top, frame.width, bottom), 1, sigma, sky_mask)
        correction = mix_diffusion(narrow, wide, parameters)
        del narrow, wide
        for row in range(0, len(correction), band_rows):
            check_cancel(cancel)
            yield correction[row : row + band_rows]


def optical_diffusion(
    frame: ImageFrame,
    parameters: EffectParameters,
    region: Bounds,
    sampling: int,
    sky_mask: SkyMask | None,
    cancel: Event | None,
) -> Pixels | None:
    """Mix each band immediately for CPU previews too large to cache."""
    if not parameters.enabled or not parameters.diffusion:
        return None
    left, top, right, bottom = region
    result = np.empty_like(frame.pixels[top:bottom:sampling, left:right:sampling])
    for rows, narrow, wide in diffusion_bands(
        frame, parameters.size, region, sampling, sky_mask, cancel
    ):
        result[rows] = mix_diffusion(narrow, wide, parameters)
    return result
