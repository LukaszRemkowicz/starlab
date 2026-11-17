from dataclasses import replace

import numpy as np
import pytest

from starlab.application.use_cases import _export_bands, render_preview
from starlab.domain.effects import diffusion
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


def test_wide_halo_share_redistributes_the_same_light():
    pixels = np.zeros((65, 65, 3), np.float32)
    pixels[32, 32] = [1, 0.4, 0.1]
    frame = ImageFrame(pixels)
    settings = optical_settings(100)

    def profile(share):
        varied = replace(settings, soft=replace(settings.soft, wide_halo_share=share))
        return displayed(render_preview(frame, (), varied))

    narrow, wide = profile(0), profile(100)
    assert narrow[32, 33, 0] > wide[32, 33, 0]
    assert wide[32, 40, 0] > narrow[32, 40, 0]
    for image in (narrow, wide):
        np.testing.assert_allclose(image.sum(axis=(0, 1)), pixels[32, 32], rtol=1e-5)
    default = displayed(render_preview(frame, (), settings))
    np.testing.assert_allclose(default, 0.7 * narrow + 0.3 * wide, atol=1e-7)


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


@pytest.mark.parametrize("channels", [1, 3])
def test_sampled_masked_wings_match_native_at_crop_and_frame_edges(channels):
    pixels = np.random.default_rng(3).random((455, 379, channels), dtype=np.float32)
    mask = SkyMask(
        1, (np.indices(pixels.shape[:2]).sum(axis=0) > 140).astype(np.uint8), pixels.shape[:2]
    )
    frame = ImageFrame(pixels)
    settings = optical_settings(90)
    settings = replace(settings, soft=replace(settings.soft, size=100))
    full = render_preview(frame, (), settings, sky_mask=mask)
    crop = render_preview(
        frame, (), settings, sky_mask=mask, region=(7, 0, 379, 455), region_sampling=9
    )
    np.testing.assert_allclose(crop.diffusion, full.diffusion[::9, 7::9], atol=6e-7)


def test_older_presets_default_to_clear_glass():
    data = preset_data(optical_settings(66))
    assert parse_preset(data).soft.diffusion == 66
    data["settings"]["soft"]["wide_halo_share"] = 72
    assert parse_preset(data).soft.wide_halo_share == 72
    for group in data["settings"].values():
        del group["wide_halo_share"]
    assert parse_preset(data).soft.wide_halo_share == 30
    assert parse_preset(data).soft.diffusion == 66
    for group in data["settings"].values():
        del group["diffusion"]
    assert parse_preset(data).soft.diffusion == 0


def test_cpu_preview_reuses_wings_only_for_the_same_source_and_geometry(monkeypatch):
    frame = ImageFrame(np.random.default_rng(7).random((93, 131, 3), dtype=np.float32))
    base = optical_settings(80)
    builds = []
    original = diffusion.preview_wings

    def count(*args):
        builds.append(1)
        return original(*args)

    monkeypatch.setattr(diffusion, "preview_wings", count)
    settings = base
    first = render_preview(frame, (), settings)
    settings = replace(base, soft=replace(base.soft, diffusion=40))
    np.testing.assert_array_equal(
        render_preview(frame, (), settings).diffusion, first.diffusion / 2
    )
    settings = replace(base, soft=replace(base.soft, wide_halo_share=90))
    assert not np.array_equal(render_preview(frame, (), settings).diffusion, first.diffusion)
    assert len(builds) == 1
    settings = replace(base, soft=replace(base.soft, size=40))
    mask = SkyMask(1, np.ones((93, 131), np.uint8), (93, 131))
    for count_expected, (source, region, sampling, sky) in enumerate(
        (
            (frame, None, 1, None),
            (frame, (7, 12, 91, 71), 1, None),
            (frame, (7, 12, 91, 71), 2, None),
            (frame, (7, 12, 91, 71), 2, mask),
            (ImageFrame(frame.pixels / 2), (7, 12, 91, 71), 2, mask),
        ),
        start=2,
    ):
        render_preview(source, (), settings, region=region, region_sampling=sampling, sky_mask=sky)
        assert len(builds) == count_expected


def test_export_shares_halos_across_writer_bands(monkeypatch):
    frame = ImageFrame(np.random.default_rng(11).random((113, 47, 1), dtype=np.float32))
    settings = optical_settings(73)
    reference = render_preview(frame, (), settings).diffusion
    calls = []
    original = diffusion._diffuse_band

    def record(frame, bounds, *args):
        calls.append(bounds)
        return original(frame, bounds, *args)

    monkeypatch.setattr(diffusion, "_diffuse_band", record)
    chunks = list(diffusion.export_diffusion_bands(frame, settings.soft, 17, None, None))
    assert [len(c) for c in chunks] == [17] * 6 + [11]
    assert len(calls) < len(chunks)
    np.testing.assert_allclose(np.concatenate(chunks), reference, atol=2e-7)
