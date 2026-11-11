from dataclasses import replace

import numpy as np
import pytest
import tifffile

from starlab.application import use_cases
from starlab.application.use_cases import ImageServices, _apply_export_source_filter
from starlab.domain.effects.profiles import image_diagonal
from starlab.domain.effects.render import composite, render_effects
from starlab.domain.effects.settings import KINDS, EffectSettings
from starlab.domain.imaging import ImageFrame, linear_to_srgb
from starlab.domain.sky_mask import SkyMask
from starlab.domain.stars import Star
from starlab.infrastructure.tiff_io import write_tiff


def scene(width, height, count, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    pixels = np.full((height, width, 3), 0.02) + rng.random((height, width, 3)) * 0.01
    stars = []
    for i in range(count):
        x, y = rng.uniform(0, width), rng.uniform(0, height)
        sigma, level = rng.uniform(1, 3), rng.uniform(0.2, 1)
        rgb = np.array([0.5 + rng.random() / 2, 0.5 + rng.random() / 2, 0.5 + rng.random() / 2])
        lo_x, hi_x = max(int(x) - 15, 0), int(x) + 16
        lo_y, hi_y = max(int(y) - 15, 0), int(y) + 16
        blob = np.exp(
            -((xx[lo_y:hi_y, lo_x:hi_x] - x) ** 2 + (yy[lo_y:hi_y, lo_x:hi_x] - y) ** 2)
            / (2 * sigma**2)
        )
        pixels[lo_y:hi_y, lo_x:hi_x] += level * blob[..., None] * rgb
        profile = (sigma * 2.354820045, 0.02, 0.02, 0.02, 0, 0, 0, 0, 0, 0)
        stars.append(
            Star(
                i + 1,
                x,
                y,
                level,
                tuple(rgb / rgb.max()),
                source_profile=profile,
                source_peak=tuple(level * rgb),
                soft_weight=rng.random(),
            )
        )
    return ImageFrame(pixels.astype(np.float32)), tuple(stars)


def all_effects():
    base = EffectSettings()
    base = replace(base, **{k: replace(getattr(base, k), enabled=True) for k in KINDS})
    return replace(
        base,
        soft=replace(base.soft, soft_filter=50),
        primary=replace(base.primary, rainbow=40),
    )


def reference(frame, stars, settings, effects_only, sky_mask):
    effects = render_effects(frame, stars, settings, sky_mask=sky_mask)
    output = effects if effects_only else composite(frame, effects)
    if not effects_only and settings.soft.soft_filter:
        _apply_export_source_filter(
            output, effects, frame.pixels, 0, stars, settings, None, sky_mask
        )
    return np.rint(np.clip(linear_to_srgb(output), 0, 1) * 65535).astype(np.uint16)


@pytest.mark.parametrize("effects_only", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_streamed_export_matches_whole_image_export(tmp_path, monkeypatch, effects_only, masked):
    monkeypatch.setattr(use_cases, "EXPORT_BAND_ROWS", 97)  # does not divide the height
    frame, stars = scene(400, 301, 60)
    settings = all_effects()
    assert image_diagonal(frame.width, frame.height) > 0
    labels = np.ones((301, 400), np.uint8)
    labels[200:] = 0
    sky = SkyMask(1, labels, (301, 400)) if masked else None
    path = tmp_path / "out.tif"
    ImageServices(None, write_tiff).export(
        path, frame, stars, settings, effects_only, None, None, sky
    )
    expected = reference(frame, stars, settings, effects_only, sky)
    difference = np.abs(tifffile.imread(path).astype(np.int32) - expected)
    assert difference.max() <= 1
    assert (difference > 0).mean() < 1e-3
