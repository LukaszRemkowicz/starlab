"""Native CPU effect profiles, quarter-pixel placement, and Screen composition."""

import math
from dataclasses import replace
from threading import Event

import numpy as np

from ..imaging import ImageFrame, Pixels, check_cancel
from ..stars import Star, filter_stars
from .profiles import (
    ArrayCache,
    directions,
    gain,
    geometry_key,
    image_diagonal,
    sampled_ray_scene,
    scene_batch_size,
    source_colors,
    sparse_profile,
)
from .settings import KINDS, EffectSettings

SCENE_CACHE = ArrayCache(128)
GROUP_CACHE = ArrayCache(32)
LIGHT_CACHE = ArrayCache(64)


def _grouped_ray_scene(kind: str, sampling: int, source_count: int, region: bool) -> bool:
    """Select the ray path whose contributions are sorted for grouped reduction."""
    return ((sampling > 1 and source_count > 8) or region) and kind in (
        "primary",
        "secondary",
    )


def _scene(width, height, stars, kind, p, sampling, cancel, diagonal=None):
    region = diagonal is not None
    diagonal = diagonal or image_diagonal(width, height)
    positions = tuple((star.x, star.y) for star in stars)
    key = width, height, positions, geometry_key(kind, p), sampling, diagonal, region
    cached = SCENE_CACHE.get(key)
    if cached is not None:
        return cached
    out_width = (width + sampling - 1) // sampling
    indices, profiles, radii, sources = [], [], [], []
    rays = directions(kind, p)
    if _grouped_ray_scene(kind, sampling, len(stars), region):
        for start in range(0, len(stars), 512):
            for angle, weight in rays:
                check_cancel(cancel)
                index, values, radius, source = sampled_ray_scene(
                    width, height, positions[start : start + 512], p, diagonal, sampling, angle
                )
                indices.append(index)
                profiles.append(values * weight)
                radii.append(radius)
                sources.append(source + start)
        arrays = tuple(np.concatenate(parts) for parts in (indices, profiles, radii, sources))
        # Stable pixel grouping turns repeated scatter multiplication into a
        # contiguous product reduction on warm slider updates.
        order = np.argsort(arrays[0], kind="stable")
        arrays = tuple(array[order] for array in arrays)
        check_cancel(cancel)
        return SCENE_CACHE.put(key, arrays)
    for i, star in enumerate(stars):
        check_cancel(cancel)
        raw_x, raw_y = star.x + 0.5, star.y + 0.5
        origin_x, origin_y = math.floor(raw_x), math.floor(raw_y)
        phase = int(4 * (raw_x - origin_x)), int(4 * (raw_y - origin_y))
        residue = (-origin_x) % sampling, (-origin_y) % sampling
        for angle, weight in rays:
            check_cancel(cancel)
            dx, dy, values, radius = sparse_profile(
                kind,
                p,
                diagonal,
                phase,
                sampling,
                residue,
                angle,
                bounds=(-origin_x, -origin_y, width - origin_x, height - origin_y)
                if region
                else None,
            )
            xx, yy = dx + origin_x, dy + origin_y
            valid = (xx >= 0) & (yy >= 0) & (xx < width) & (yy < height)
            if not valid.any():
                continue
            indices.append(
                ((yy[valid] // sampling) * out_width + xx[valid] // sampling).astype(np.int32)
            )
            profiles.append(values[valid] * weight)
            radii.append(radius[valid])
            sources.append(np.full(int(valid.sum()), i, np.int32))
    if not indices:
        return tuple(np.empty(0, dtype) for dtype in (np.int32, np.float32, np.float32, np.int32))
    arrays = tuple(np.concatenate(parts) for parts in (indices, profiles, radii, sources))
    check_cancel(cancel)
    return SCENE_CACHE.put(key, arrays)


def _lit_scene(frame, stars, maximum, kind, p, sampling, cancel, diagonal=None):
    """Cache source-colored ray light independently of the intensity slider."""
    channels = frame.pixels.shape[2]
    key = (
        frame.width,
        frame.height,
        channels,
        stars,
        maximum,
        kind,
        replace(p, intensity=0, soft_filter=0),
        sampling,
        diagonal,
    )
    cached = LIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    indices, profile, radius, source = _scene(
        frame.width, frame.height, stars, kind, p, sampling, cancel, diagonal
    )
    brightness = np.array([s.brightness for s in stars], np.float32)
    threshold = (1 - p.quantity / 100) ** 2 * maximum
    factors = np.maximum(brightness - threshold, 0) / ((1 - threshold) * brightness)
    original = np.array([s.color for s in stars])
    colors = source_colors(original, p.hue, p.saturation)
    # The native catalog already has this luminance. The ratio additionally
    # applies manual brightness edits without changing the source hue.
    original_luma = original @ np.array([0.212639004, 0.715168655, 0.072192311])
    colors *= np.divide(
        brightness, original_luma, out=np.zeros_like(brightness), where=original_luma > 0
    )[:, None]
    if channels == 1:
        colors = colors.mean(axis=1, keepdims=True)
    amplitudes = colors * factors[:, None]
    scale = 800 / max(diagonal or image_diagonal(frame.width, frame.height), 1414)
    rays = kind in ("primary", "secondary")
    values = profile
    if rays:
        values = values * np.maximum(
            29 * ((1 + radius.astype(np.float64) * scale) ** (-190 / (p.length + 25)) - 0.000015), 0
        )
    selected = (values > 0) & (factors[source] > 0)
    radius = radius[selected]
    light = values[selected, None] * amplitudes[source[selected]]
    if rays and p.rainbow:
        u = radius.astype(np.float64) * scale * 125000 / (107 - p.frequency) ** 2
        theta = np.deg2rad((np.trunc(u) + 36 * (50 - p.phase)) * 0.1)
        amount = 0.017 * p.rainbow * np.minimum(u / 3000, 1)
        modulation = (
            np.sin(theta)[:, None]
            if channels == 1
            else np.stack((np.sin(theta), np.cos(theta), -np.sin(theta)), axis=1)
        )
        light *= np.maximum(1 + amount[:, None] * modulation, 0)
    check_cancel(cancel)
    return LIGHT_CACHE.put(key, (indices[selected], light.astype(np.float32)))


def _group(frame, stars, maximum, kind, p, sampling, cancel, diagonal=None):
    h, w = (frame.height + sampling - 1) // sampling, (frame.width + sampling - 1) // sampling
    channels = frame.pixels.shape[2]
    key = (
        frame.width,
        frame.height,
        channels,
        stars,
        maximum,
        kind,
        replace(p, soft_filter=0),
        sampling,
        diagonal,
    )
    cached = GROUP_CACHE.get(key)
    if cached is not None:
        return cached
    threshold = (1 - p.quantity / 100) ** 2 * maximum
    stars = tuple(star for star in stars if star.brightness > threshold)
    transmission = np.ones((h * w, channels), np.float32)
    intensity = gain(kind, p)
    batch = scene_batch_size(
        kind, p, diagonal or image_diagonal(frame.width, frame.height), sampling
    )
    if diagonal is not None:
        batch = min(64, max(1, 1048576 // (4 * (frame.width + frame.height))))
    for base in range(0, len(stars), batch):
        subset = stars[base : base + batch]
        indices, light = _lit_scene(frame, subset, maximum, kind, p, sampling, cancel, diagonal)
        for start in range(0, len(indices), 131072):
            check_cancel(cancel)
            pixels = indices[start : start + 131072]
            factors = 1 - np.clip(light[start : start + 131072] * intensity, 0, 1)
            if _grouped_ray_scene(kind, sampling, len(subset), diagonal is not None):
                boundaries = np.r_[0, np.flatnonzero(np.diff(pixels)) + 1]
                transmission[pixels[boundaries]] *= np.multiply.reduceat(
                    factors, boundaries, axis=0
                )
            else:
                np.multiply.at(transmission, pixels, factors)
    result = (1 - transmission).reshape(h, w, channels)
    check_cancel(cancel)
    return GROUP_CACHE.put(key, result)


def render_effects(
    frame: ImageFrame,
    stars: tuple[Star, ...],
    settings: EffectSettings,
    cancel: Event | None = None,
    *,
    sampling: int = 1,
    diagonal: int | None = None,
    sky_mask=None,
) -> Pixels:
    check_cancel(cancel)
    stars = filter_stars(stars, sky_mask)
    if sampling < 1:
        raise ValueError("Sampling must be positive.")
    result = np.zeros_like(frame.pixels[::sampling, ::sampling])
    maximum = max((star.brightness for star in stars), default=0.0)
    visible = tuple(star for star in stars if star.visible)
    if not visible:
        return result
    for kind in KINDS:
        p = getattr(settings, kind)
        if not p.enabled or p.intensity == 0 or p.quantity == 0:
            continue
        group = _group(frame, visible, maximum, kind, p, sampling, cancel, diagonal)
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
    pixels = frame.pixels if reduction is None else np.maximum(frame.pixels - reduction, 0)
    if enlargement is not None:
        pixels = pixels + enlargement
    return pixels + (1 - pixels) * effects
