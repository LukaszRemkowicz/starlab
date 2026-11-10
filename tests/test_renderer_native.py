"""Image regressions using self-contained numerical inputs and expected pixels."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starlab.domain.effects.render import render_effects
from starlab.domain.effects.settings import EffectSettings
from starlab.domain.imaging import ImageFrame, srgb_to_linear
from starlab.domain.presets import parse_preset
from starlab.domain.stars import Star

PROBES = Path(__file__).resolve().parent / "fixtures/rendering"
_RECORDS = json.loads((PROBES / "cases.json").read_text())
_BY_IMAGE = {row["image"]: row for row in _RECORDS}
CASES = [PROBES / row["image"] for row in _RECORDS]


def native_case(path):
    record = _BY_IMAGE[path.relative_to(PROBES).as_posix()]
    stars = tuple(
        Star(
            **{
                **row,
                "color": tuple(row["color"]),
                "source_profile": tuple(row["source_profile"]),
                "source_peak": tuple(row["source_peak"]),
            }
        )
        for row in record["stars"]
    )
    expected = srgb_to_linear(tifffile.imread(path).astype(np.float32) / 65535)
    return ImageFrame(np.zeros_like(expected)), stars, parse_preset(record["preset"]), expected


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_production_matches_native_full_image(path):
    frame, stars, settings, expected = native_case(path)
    result = render_effects(frame, stars, settings)
    assert float(np.max(np.abs(result - expected))) < 0.00025


@pytest.mark.parametrize("sampling", [2, 3, 7])
def test_sampled_native_rays_preserve_pixel_antialiasing(sampling):
    path = PROBES / "color-quantity-1793869750289/secondary-q75.tif"
    frame, stars, settings, _ = native_case(path)
    full = render_effects(frame, stars, settings)
    sampled = render_effects(frame, stars, settings, sampling=sampling)
    np.testing.assert_allclose(sampled, full[::sampling, ::sampling], atol=1e-6)


@pytest.mark.parametrize("kind", ["primary", "secondary"])
@pytest.mark.parametrize("sampling", [2, 7])
def test_batched_scene_preserves_exact_rasterization(kind, sampling):
    path = PROBES / f"color-quantity-1793869750289/{kind}-q75.tif"
    frame, stars, settings, _ = native_case(path)
    stars += (replace(stars[0], id=99, x=12.31, y=9.89),)
    full = render_effects(frame, stars, settings)
    sampled = render_effects(frame, stars, settings, sampling=sampling)
    np.testing.assert_allclose(sampled, full[::sampling, ::sampling], atol=1e-6)


def test_manual_brightness_edit_changes_emitted_light():
    path = PROBES / "rays-1793868364216/primary-reference.tif"
    frame, stars, settings, _ = native_case(path)
    dimmer = replace(stars[0], brightness=stars[0].brightness / 2)
    equivalent = replace(dimmer, color=tuple(c / 2 for c in dimmer.color))
    edited = render_effects(frame, (dimmer,), settings)
    native_color = render_effects(frame, (equivalent,), settings)
    np.testing.assert_allclose(edited, native_color, atol=1e-6)
    assert edited.sum() < render_effects(frame, stars, settings).sum()


def test_streamed_source_batches_preserve_screen_composition(monkeypatch):
    from starlab.domain.effects import render

    path = PROBES / "color-quantity-1793869750289/secondary-q75.tif"
    frame, stars, settings, _ = native_case(path)
    expected = render_effects(frame, stars, settings, sampling=3)
    monkeypatch.setattr(render, "GROUP_CACHE", render.ArrayCache(1))
    monkeypatch.setattr(render, "scene_batch_size", lambda *args: 1)
    actual = render_effects(frame, stars, settings, sampling=3)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("kind", ["soft", "ring"])
def test_many_radial_sources_preserve_sampled_screen_composition(kind):
    path = PROBES / f"batch-1793867561852/{kind}-base.tif"
    frame, stars, settings, _ = native_case(path)
    stars = tuple(replace(stars[0], id=i, x=500.25 + i * 2, y=380.25 + i * 3) for i in range(9))
    full = render_effects(frame, stars, settings)
    sampled = render_effects(frame, stars, settings, sampling=3)
    np.testing.assert_allclose(sampled, full[::3, ::3], atol=1e-6)


def test_drag_and_release_use_the_entire_catalog_even_when_dense(monkeypatch):
    from starlab.application import use_cases

    stars = tuple(Star(i, i % 96, i % 64, 0.5, (0.5, 0.5, 0.5)) for i in range(4100))
    frame = ImageFrame(np.zeros((64, 96, 3), np.float32))
    settings = EffectSettings.disabled()
    settings = replace(settings, primary=replace(settings.primary, enabled=True, quantity=100))
    seen = []

    def capture(frame, catalog, settings, cancel, *, sampling):
        seen.append(catalog)
        return np.zeros_like(frame.pixels[::sampling, ::sampling])

    monkeypatch.setattr(use_cases, "render_effects", capture)
    draft = use_cases.render_preview(frame, stars, settings, interactive=True)
    assert draft.draft_count is None
    assert seen[-1] is stars
    refined = use_cases.render_preview(frame, stars, settings, interactive=False)
    assert refined.draft_count is None
    assert seen[-1] is stars


@pytest.mark.parametrize("kind", ["primary", "secondary", "soft", "ring"])
@pytest.mark.parametrize("sampling", [1, 3])
def test_native_region_matches_full_frame_and_keeps_outside_beams(kind, sampling):
    from starlab.application.use_cases import render_preview

    frame = ImageFrame(np.zeros((720, 1600, 3), np.float32))
    stars = (
        Star(1, 400.1875, 350.1875, 0.4, (0.2, 0.4, 0.5)),
        Star(2, 440.45, 380.61, 0.2, (0.2, 0.2, 0.2)),
        Star(3, 1400, 650, 0.9, (0.9, 0.9, 0.9), visible=False),
    )
    settings = EffectSettings.disabled()
    p = replace(
        getattr(settings, kind),
        enabled=True,
        intensity=20,
        quantity=75,
        length=50,
        size=50,
        angle=10,
        hue=20,
        saturation=40,
    )
    settings = replace(settings, **{kind: p})
    full = render_effects(frame, stars, settings)
    region = (408, 330, 480, 400)
    preview = render_preview(
        frame, stars, settings, interactive=True, region=region, region_sampling=sampling
    )
    assert preview.origin == (408, 330)
    assert preview.scale == sampling and preview.is_region
    assert preview.draft_count is None
    expected = full[330:400:sampling, 408:480:sampling]
    assert expected.max() > 0
    np.testing.assert_allclose(preview.effects, expected, atol=1e-6)
    np.testing.assert_array_equal(
        preview.frame.pixels, frame.pixels[330:400:sampling, 408:480:sampling]
    )


def test_region_drag_never_applies_spatial_source_lod(monkeypatch):
    from starlab.application import use_cases

    frame = ImageFrame(np.zeros((64, 96, 3), np.float32))
    stars = tuple(Star(i, 40, 32, 0.5, (0.5, 0.5, 0.5)) for i in range(4100))
    settings = EffectSettings.disabled()
    settings = replace(settings, primary=replace(settings.primary, enabled=True, quantity=100))
    seen = []

    def capture(frame, catalog, settings, cancel, *, sampling, diagonal):
        seen.append(catalog)
        return np.zeros_like(frame.pixels[::sampling, ::sampling])

    monkeypatch.setattr(use_cases, "render_effects", capture)
    preview = use_cases.render_preview(
        frame, stars, settings, interactive=True, region=(16, 0, 80, 64)
    )
    assert len(seen[0]) == len(stars)
    assert preview.draft_count is None and preview.scale == 1
