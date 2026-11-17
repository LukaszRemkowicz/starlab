from dataclasses import replace

import numpy as np
import pytest

from starlab.application.use_cases import _export_bands, render_preview
from starlab.domain.effects.render import composite
from starlab.domain.effects.settings import EffectSettings
from starlab.domain.imaging import ImageFrame
from starlab.domain.presets import parse_preset, preset_data
from starlab.domain.sky_mask import SkyMask


def optical_settings(strength):
    base = EffectSettings.disabled()
    return replace(base, soft=replace(base.soft, enabled=True, intensity=0, diffusion=strength))


def displayed(preview):
    return composite(preview.frame, preview.effects, diffusion=preview.diffusion)


def test_diffusion_moves_light_from_core_to_colored_wings_without_catalog():
    pixels = np.zeros((51, 51, 3), np.float32)
    pixels[25, 25] = [1, 0.4, 0.1]
    frame = ImageFrame(pixels)
    result = displayed(render_preview(frame, (), optical_settings(100)))
    profile = result[25, 25:33, 0]
    assert 0.4 < profile[0] < 1
    assert np.all(np.diff(profile) < 0) and profile[-1] > 0
    np.testing.assert_allclose(result.sum(axis=(0, 1)), pixels[25, 25], rtol=1e-5)
    np.testing.assert_allclose(result[..., 1], result[..., 0] * 0.4, atol=1e-7)
    half = displayed(render_preview(frame, (), optical_settings(50)))
    np.testing.assert_allclose(half, (result + pixels) / 2, atol=1e-7)


@pytest.mark.parametrize("channels", [1, 3])
def test_diffusion_uses_native_neighbors_across_crop_and_export_bands(channels):
    pixels = np.random.default_rng(4).random((281, 65, channels), dtype=np.float32)
    pixels[255, 20] = 10
    frame = ImageFrame(pixels)
    settings = optical_settings(75)
    full = displayed(render_preview(frame, (), settings))
    region = render_preview(frame, (), settings, region=(13, 256, 50, 279), region_sampling=3)
    np.testing.assert_allclose(displayed(region), full[256:279:3, 13:50:3], atol=2e-7)
    exported = np.concatenate(list(_export_bands(frame, (), settings, False, None, None)))
    np.testing.assert_allclose(exported, full, atol=2e-7)
    assert not any(band.any() for band in _export_bands(frame, (), settings, True, None, None))


def test_mask_protects_both_sides_of_the_horizon_and_preserves_flat_sky():
    pixels = np.full((60, 80, 1), 0.03, np.float32)
    pixels[30:] = 2
    mask = SkyMask(1, (pixels[..., 0] < 1).astype(np.uint8), pixels.shape[:2])
    preview = render_preview(ImageFrame(pixels), (), optical_settings(100), sky_mask=mask)
    np.testing.assert_allclose(displayed(preview), pixels, atol=1e-7)


def test_older_presets_default_to_clear_glass():
    data = preset_data(optical_settings(66))
    assert parse_preset(data).soft.diffusion == 66
    for group in data["settings"].values():
        del group["diffusion"]
    assert parse_preset(data).soft.diffusion == 0
