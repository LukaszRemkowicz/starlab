"""Native CPU effect profiles, quarter-pixel placement, and Screen composition."""

import math
from dataclasses import replace
from threading import Event
from typing import cast

import numpy as np

from ..array_types import BoolArray, DoubleArray, FloatArray, IntArray, Position
from ..imaging import ImageFrame, ImagePixels, Pixels, check_cancel
from ..sky_mask import SkyMask
from ..stars import Star, filter_stars
from .profiles import (
    ArrayCache,
    CacheKey,
    IndexArray,
    SceneArrays,
    directions,
    gain,
    geometry_key,
    image_diagonal,
    sampled_ray_scene,
    scene_batch_size,
    source_colors,
    sparse_profile,
)
from .settings import KINDS, EffectKind, EffectParameters, EffectSettings
from .soft_filter import soft_halo_factors

type LitArrays = tuple[IndexArray, Pixels]

SCENE_CACHE = ArrayCache[SceneArrays](128)
GROUP_CACHE = ArrayCache[Pixels](32)
LIGHT_CACHE = ArrayCache[LitArrays](64)
# Sources per batch in an export band: bounds the transient ray arrays of long Secondary rays.
EXPORT_BATCH = 4
LUMA = np.array([0.212639004, 0.715168655, 0.072192311], np.float32)


def fit_to_white(light: Pixels) -> Pixels:
    """Limit per-source RGB light to [0, 1] along a path to white.

    Clipping each channel separately pushes bright colors toward pure primaries
    (orange cores turn yellow). Instead, an overbright color keeps its hue and
    luminance and loses only as much saturation as it must; luminance >= 1 is white.
    """
    result: Pixels = np.clip(light, 0, 1)
    if light.shape[1] != 3:
        return result
    peak: Pixels = light.max(axis=1)
    over: BoolArray = peak > 1
    if not over.any():
        return result
    color: Pixels = light[over]
    luma: Pixels = color @ LUMA
    spread: Pixels = peak[over] - luma
    fraction: Pixels = np.clip(
        np.divide(1 - luma, spread, out=np.zeros_like(luma), where=spread > 0), 0, 1
    )
    result[over] = np.clip(luma[:, None] + fraction[:, None] * (color - luma[:, None]), 0, 1)
    return result


def _grouped_ray_scene(
    kind: EffectKind, sampling: int, source_count: int, region: bool, export: bool = False
) -> bool:
    """Select the ray path whose contributions are sorted for grouped reduction."""
    # Export bands are full resolution; the sparse path is faster there.
    return ((sampling > 1 and source_count > 8) or (region and not export)) and kind in (
        "primary",
        "secondary",
    )


def _sampled_scene(
    width: int,
    height: int,
    positions: tuple[tuple[float, float], ...],
    rays: tuple[tuple[int, float], ...],
    parameters: EffectParameters,
    diagonal: int,
    sampling: int,
    cancel: Event | None,
) -> SceneArrays:
    """Assemble sampled source batches and stably group contributions by pixel."""
    indices: list[IndexArray] = []
    profiles: list[Pixels] = []
    radii: list[Pixels] = []
    sources: list[IndexArray] = []
    for start in range(0, len(positions), 512):
        for angle, weight in rays:
            check_cancel(cancel)
            index, values, radius, source = sampled_ray_scene(
                width, height, positions[start : start + 512], parameters, diagonal, sampling, angle
            )
            indices.append(index)
            profiles.append(values * weight)
            radii.append(radius)
            sources.append(source + start)
    arrays: SceneArrays = cast(
        SceneArrays, tuple(np.concatenate(parts) for parts in (indices, profiles, radii, sources))
    )
    # Stable pixel grouping turns repeated scatter multiplication into a
    # contiguous product reduction on warm slider updates.
    order: IntArray = np.argsort(arrays[0], kind="stable")
    arrays = cast(SceneArrays, tuple(array[order] for array in arrays))
    return arrays


def _sparse_scene(
    width: int,
    height: int,
    stars: tuple[Star, ...],
    kind: EffectKind,
    parameters: EffectParameters,
    diagonal: int,
    sampling: int,
    cancel: Event | None,
    region: bool,
    export: bool = False,
) -> SceneArrays | None:
    """Place cached per-source profiles, retaining their original contribution order."""
    out_width: int = (width + sampling - 1) // sampling
    indices: list[IndexArray] = []
    profiles: list[Pixels] = []
    radii: list[Pixels] = []
    sources: list[IndexArray] = []
    rays: tuple[tuple[int, float], ...] = directions(kind, parameters)
    for source_index, star in enumerate(stars):
        check_cancel(cancel)
        position_x: float
        position_y: float
        position_x, position_y = star.x + 0.5, star.y + 0.5
        origin_x: int
        origin_y: int
        origin_x, origin_y = math.floor(position_x), math.floor(position_y)
        phase: tuple[int, int] = int(4 * (position_x - origin_x)), int(4 * (position_y - origin_y))
        residue: tuple[int, int] = (-origin_x) % sampling, (-origin_y) % sampling
        for angle, weight in rays:
            check_cancel(cancel)
            offset_x, offset_y, values, radius = sparse_profile(
                kind,
                parameters,
                diagonal,
                phase,
                sampling,
                residue,
                angle,
                bounds=(-origin_x, -origin_y, width - origin_x, height - origin_y)
                if region and not export  # export bands share the unbounded cached profile
                else None,
            )
            # Rows ascend (rays are sorted, radial grids are row-major): keep only the image's rows.
            if not len(offset_y) or offset_y[0] >= height - origin_y or offset_y[-1] < -origin_y:
                continue
            first, last = np.searchsorted(offset_y, (-origin_y, height - origin_y))
            offset_x, offset_y = offset_x[first:last], offset_y[first:last]
            values, radius = values[first:last], radius[first:last]
            world_x: IndexArray
            world_y: IndexArray
            world_x, world_y = offset_x + origin_x, offset_y + origin_y
            valid: BoolArray = (
                (world_x >= 0) & (world_y >= 0) & (world_x < width) & (world_y < height)
            )
            if not valid.any():
                continue
            indices.append(
                ((world_y[valid] // sampling) * out_width + world_x[valid] // sampling).astype(
                    np.int32
                )
            )
            profiles.append(values[valid] * weight)
            radii.append(radius[valid])
            sources.append(np.full(int(valid.sum()), source_index, np.int32))
    if not indices:
        return None
    arrays: SceneArrays = cast(
        SceneArrays, tuple(np.concatenate(parts) for parts in (indices, profiles, radii, sources))
    )
    return arrays


def _scene(
    width: int,
    height: int,
    stars: tuple[Star, ...],
    kind: EffectKind,
    parameters: EffectParameters,
    sampling: int,
    cancel: Event | None,
    diagonal: int | None = None,
    export: bool = False,
) -> SceneArrays:
    """Cache integer pixel indices, transverse profiles, radii and source indices."""
    region: bool = diagonal is not None
    diagonal = diagonal or image_diagonal(width, height)
    positions: tuple[Position, ...] = tuple((star.x, star.y) for star in stars)
    key: CacheKey = (
        width,
        height,
        positions,
        geometry_key(kind, parameters),
        sampling,
        diagonal,
        region,
    )
    cached: SceneArrays | None = None if export else SCENE_CACHE.get(key)
    if cached is not None:
        return cached
    arrays: SceneArrays | None
    if _grouped_ray_scene(kind, sampling, len(stars), region, export):
        arrays = _sampled_scene(
            width,
            height,
            positions,
            directions(kind, parameters),
            parameters,
            diagonal,
            sampling,
            cancel,
        )
    else:
        arrays = _sparse_scene(
            width, height, stars, kind, parameters, diagonal, sampling, cancel, region, export
        )
        if arrays is None:
            return cast(
                SceneArrays,
                tuple(np.empty(0, dtype) for dtype in (np.int32, np.float32, np.float32, np.int32)),
            )
    check_cancel(cancel)
    return arrays if export else SCENE_CACHE.put(key, arrays)


def _lit_scene(
    frame: ImagePixels,
    stars: tuple[Star, ...],
    maximum: float,
    kind: EffectKind,
    parameters: EffectParameters,
    sampling: int,
    cancel: Event | None,
    diagonal: int | None = None,
    export: bool = False,
) -> LitArrays:
    """Cache source-colored ray light independently of the intensity slider."""
    channels: int = frame.pixels.shape[2]
    key: CacheKey = (
        frame.width,
        frame.height,
        channels,
        stars,
        maximum,
        kind,
        replace(
            parameters, intensity=0, soft_filter=parameters.soft_filter if kind == "soft" else 0
        ),
        sampling,
        diagonal,
    )
    cached: LitArrays | None = None if export else LIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    radius: FloatArray
    indices, profile, radius, source = _scene(
        frame.width, frame.height, stars, kind, parameters, sampling, cancel, diagonal, export
    )
    brightness: FloatArray = np.array([star.brightness for star in stars], np.float32)
    threshold: float = (1 - parameters.quantity / 100) ** 2 * maximum
    # Edited brightness may exceed 1; normalizing by the ceiling keeps the denominator positive.
    ceiling: float = max(maximum, 1.0)
    factors: FloatArray = np.maximum(brightness - threshold, 0) / (
        (ceiling - threshold) * brightness
    )
    original: DoubleArray = np.array([star.color for star in stars])
    colors: FloatArray = source_colors(original, parameters.hue, parameters.saturation)
    # The native catalog already has this luminance. The ratio additionally
    # applies manual brightness edits without changing the source hue.
    original_luma: DoubleArray = original @ np.array([0.212639004, 0.715168655, 0.072192311])
    colors *= np.divide(
        brightness, original_luma, out=np.zeros_like(brightness), where=original_luma > 0
    )[:, None]
    if channels == 1:
        colors = colors.mean(axis=1, keepdims=True)
    amplitudes: FloatArray = colors * factors[:, None]
    if kind == "soft" and parameters.soft_filter:
        amplitudes *= soft_halo_factors(stars, parameters.soft_filter)[:, None]
    scale: float = 800 / max(diagonal or image_diagonal(frame.width, frame.height), 1414)
    rays: bool = kind in ("primary", "secondary")
    values: FloatArray | DoubleArray = profile
    if rays:
        values = values * np.maximum(
            29
            * (
                (1 + radius.astype(np.float64) * scale) ** (-190 / (parameters.length + 25))
                - 0.000015
            ),
            0,
        )
    selected: BoolArray = (values > 0) & (factors[source] > 0)
    radius = radius[selected]
    light: FloatArray | DoubleArray = values[selected, None] * amplitudes[source[selected]]
    if rays and parameters.rainbow:
        rainbow_distance: DoubleArray = (
            radius.astype(np.float64) * scale * 125000 / (107 - parameters.frequency) ** 2
        )
        theta: DoubleArray = np.deg2rad(
            (np.trunc(rainbow_distance) + 36 * (50 - parameters.phase)) * 0.1
        )
        amount: DoubleArray = 0.017 * parameters.rainbow * np.minimum(rainbow_distance / 3000, 1)
        modulation: DoubleArray = (
            np.sin(theta)[:, None]
            if channels == 1
            else np.stack((np.sin(theta), np.cos(theta), -np.sin(theta)), axis=1)
        )
        light *= np.maximum(1 + amount[:, None] * modulation, 0)
    check_cancel(cancel)
    lit: LitArrays = indices[selected], light.astype(np.float32)
    return lit if export else LIGHT_CACHE.put(key, lit)


def _group(
    frame: ImagePixels,
    stars: tuple[Star, ...],
    maximum: float,
    kind: EffectKind,
    parameters: EffectParameters,
    sampling: int,
    cancel: Event | None,
    diagonal: int | None = None,
    export: bool = False,
) -> Pixels:
    """Screen-compose one emission group into an immutable sampled layer."""
    output_height: int
    output_width: int
    output_height, output_width = (
        (frame.height + sampling - 1) // sampling,
        (frame.width + sampling - 1) // sampling,
    )
    channels: int = frame.pixels.shape[2]
    key: CacheKey = (
        frame.width,
        frame.height,
        channels,
        stars,
        maximum,
        kind,
        parameters if kind == "soft" else replace(parameters, soft_filter=0),
        sampling,
        diagonal,
    )
    cached: Pixels | None = None if export else GROUP_CACHE.get(key)
    if cached is not None:
        return cached
    threshold: float = (1 - parameters.quantity / 100) ** 2 * maximum
    stars = tuple(star for star in stars if star.brightness > threshold)
    transmission: Pixels = np.ones((output_height * output_width, channels), np.float32)
    intensity: float = gain(kind, parameters)
    batch: int = scene_batch_size(
        kind, parameters, diagonal or image_diagonal(frame.width, frame.height), sampling
    )
    if diagonal is not None:
        batch = min(64, max(1, 1048576 // (4 * (frame.width + frame.height))))
    if export:
        batch = EXPORT_BATCH
    for base in range(0, len(stars), batch):
        subset: tuple[Star, ...] = stars[base : base + batch]
        indices, light = _lit_scene(
            frame, subset, maximum, kind, parameters, sampling, cancel, diagonal, export
        )
        for start in range(0, len(indices), 131072):
            check_cancel(cancel)
            pixels: IndexArray = indices[start : start + 131072]
            factors: Pixels = 1 - fit_to_white(light[start : start + 131072] * intensity)
            if _grouped_ray_scene(kind, sampling, len(subset), diagonal is not None, export):
                boundaries: IntArray = np.r_[0, np.flatnonzero(np.diff(pixels)) + 1]
                transmission[pixels[boundaries]] *= np.multiply.reduceat(
                    factors, boundaries, axis=0
                )
            else:
                np.multiply.at(transmission, pixels, factors)
    result: Pixels = (1 - transmission).reshape(output_height, output_width, channels)
    check_cancel(cancel)
    return result if export else GROUP_CACHE.put(key, result)


def render_effects(
    frame: ImagePixels,
    stars: tuple[Star, ...],
    settings: EffectSettings,
    cancel: Event | None = None,
    *,
    sampling: int = 1,
    diagonal: int | None = None,
    sky_mask: SkyMask | None = None,
    export: bool = False,
) -> Pixels:
    """Render Screen-composed linear emission without modifying the source pixels.

    ``export`` marks a full-resolution band: layers bypass the shared caches (their keys
    never repeat) and rays use the faster sparse path.
    """
    check_cancel(cancel)
    stars = filter_stars(stars, sky_mask)
    if sampling < 1:
        raise ValueError("Sampling must be positive.")
    result: Pixels = np.zeros_like(frame.pixels[::sampling, ::sampling])
    maximum: float = max((star.brightness for star in stars), default=0.0)
    visible: tuple[Star, ...] = tuple(star for star in stars if star.visible)
    if not visible:
        return result
    for kind in KINDS:
        parameters: EffectParameters = getattr(settings, kind)
        if not parameters.enabled or parameters.intensity == 0 or parameters.quantity == 0:
            continue
        group: Pixels = _group(
            frame, visible, maximum, kind, parameters, sampling, cancel, diagonal, export
        )
        result += (1 - result) * group
    check_cancel(cancel)
    if sky_mask is not None:
        result *= sky_mask.region((0, 0, frame.width, frame.height), sampling)[..., None]
    return result


def composite(
    frame: ImageFrame,
    effects: Pixels,
    *,
    reduction: Pixels | None = None,
    enlargement: Pixels | None = None,
) -> Pixels:
    """Screen in linear working space, preserving exact identity at zero effects."""
    pixels: Pixels = frame.pixels if reduction is None else np.maximum(frame.pixels - reduction, 0)
    if enlargement is not None:
        pixels = pixels + enlargement
    return pixels + (1 - pixels) * effects
