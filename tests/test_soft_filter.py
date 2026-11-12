from dataclasses import replace

import numpy as np
import pytest

from starlab.application.use_cases import render_preview
from starlab.domain.effects.render import composite
from starlab.domain.effects.settings import EffectSettings
from starlab.domain.imaging import ImageFrame
from starlab.domain.presets import parse_preset, preset_data
from starlab.domain.stars import Star


def scene():
    yy, xx = np.mgrid[:96, :160]
    pixels = np.full((96, 160, 3), 0.02)
    stars = []
    for i, (x, sigma, weight) in enumerate([(40, 1.5, 0), (115, 4, 1)]):
        rgb = np.array([0.2, 0.4, 0.6])
        pixels += np.exp(-((xx - x) ** 2 + (yy - 48) ** 2) / (2 * sigma**2))[..., None] * rgb
        profile = (sigma * 2.354820045, 0.02, 0.02, 0.02, 0, 0, 0, 0, 0, 0)
        stars.append(
            Star(
                i + 1,
                x,
                48,
                0.4,
                tuple(rgb),
                source_profile=profile,
                source_peak=tuple(rgb),
                soft_weight=weight,
            )
        )
    return ImageFrame(pixels.astype(np.float32)), tuple(stars)


def settings(value):
    base = EffectSettings.disabled()
    return replace(
        base,
        soft=replace(
            base.soft, enabled=True, quantity=100, intensity=40, size=35, soft_filter=value
        ),
    )


def test_legacy_presets_are_neutral_and_new_value_roundtrips():
    old = preset_data(EffectSettings())
    for group in old["settings"].values():
        group.pop("soft_filter", None)
    assert parse_preset(old).soft.soft_filter == 0
    assert parse_preset(preset_data(settings(73))).soft.soft_filter == 73


def test_small_star_wings_shrink_but_background_and_big_star_are_protected():
    from starlab.domain.effects.soft_filter import reduction_records, star_reduction

    frame, stars = scene()
    original = frame.pixels.copy()
    records = reduction_records(stars)
    light = star_reduction(frame.pixels, records, 100)
    assert light[48, 42].max() > 0
    assert not light[48, 40].any()
    assert not light[:, 80:].any()
    assert not light[:30].any()
    assert np.min(frame.pixels - light) >= 0.02 - 1e-6
    np.testing.assert_array_equal(
        star_reduction(frame.pixels, np.concatenate([records, records]), 100), light
    )
    np.testing.assert_array_equal(frame.pixels, original)
    assert not star_reduction(frame.pixels, records, 0).any()


def test_reduction_narrows_small_source_at_fixed_half_height_without_raising_peak():
    from starlab.domain.effects.soft_filter import reduction_records, star_reduction

    frame, stars = scene()
    reduced = frame.pixels - star_reduction(frame.pixels, reduction_records(stars), 100)

    def half_width(pixels):
        profile = pixels[48, 40:50, 2] - 0.02
        x = np.arange(len(profile))
        return np.interp(profile[0] / 2, profile[::-1], x[::-1])

    before, after = half_width(frame.pixels), half_width(reduced)
    assert 0.4 * before < after < 0.8 * before  # clearly narrower, not erased
    assert reduced[48, 40, 2] == frame.pixels[48, 40, 2]


def test_filter_strength_is_continuous_and_roi_matches_full_result():
    frame, stars = scene()
    previews = [
        render_preview(frame, stars, settings(s), full_resolution=True) for s in [0, 50, 100]
    ]
    result = [
        composite(p.frame, p.effects, reduction=p.reduction, enlargement=p.enlargement)
        for p in previews
    ]
    assert result[2][48, 42, 2] < result[1][48, 42, 2] < result[0][48, 42, 2]
    assert result[2][48, 122, 2] > result[1][48, 122, 2] > result[0][48, 122, 2]
    roi = render_preview(frame, stars, settings(100), region=(25, 35, 140, 65))
    np.testing.assert_allclose(
        composite(roi.frame, roi.effects, reduction=roi.reduction, enlargement=roi.enlargement),
        result[2][35:65, 25:140],
        atol=1e-6,
    )


def test_filter_changes_original_stars_before_halo_selection_and_other_effects():
    frame, stars = scene()
    base = settings(100)
    no_halo = replace(base, soft=replace(base.soft, intensity=0))
    p = render_preview(frame, stars, no_halo, full_resolution=True)
    result = composite(p.frame, p.effects, reduction=p.reduction, enlargement=p.enlargement)
    assert not p.effects.any()
    assert result[48, 42, 2] < frame.pixels[48, 42, 2]
    assert result[48, 122, 2] > frame.pixels[48, 122, 2]
    assert result[48, 115, 2] == frame.pixels[48, 115, 2]
    for value in [20, 73, 100]:
        halo = render_preview(frame, stars, settings(value), full_resolution=True)
        np.testing.assert_allclose(halo.enlargement / value, p.enlargement / 100, atol=1e-7)
    glow = np.full_like(frame.pixels, 0.2)
    filtered = frame.pixels - p.reduction + p.enlargement
    np.testing.assert_allclose(
        composite(frame, glow, reduction=p.reduction, enlargement=p.enlargement),
        filtered + (1 - filtered) * glow,
        atol=1e-7,
    )


def test_measured_profiles_and_weights_use_original_width():
    from starlab.domain.photometry import calibrate_sources

    frame, stars = scene()
    measured = calibrate_sources(
        frame, tuple(replace(s, source_profile=(), soft_weight=1) for s in stars)
    )
    assert measured[0].source_profile[0] < measured[1].source_profile[0]
    assert measured[0].soft_weight < measured[1].soft_weight
    assert all(any(s.source_peak) for s in measured)
    np.testing.assert_allclose(
        np.array(measured[1].source_peak) / measured[1].source_peak[2], [1 / 3, 2 / 3, 1], atol=1e-5
    )


def test_large_source_expansion_preserves_center_color_and_masks_foreground():
    from starlab.domain.effects.soft_filter import enlargement_records, star_enlargement
    from starlab.domain.sky_mask import SkyMask

    frame, stars = scene()
    records = enlargement_records(stars)
    large = star_enlargement(frame.pixels, records, 100)
    assert not large[48, 115].any()
    np.testing.assert_allclose(large[48, 122] / large[48, 122, 2], [1 / 3, 2 / 3, 1], atol=1e-6)
    assert not large[:, :60].any()  # halo reaches at most 12 sigma = 48 px
    np.testing.assert_array_equal(
        star_enlargement(frame.pixels, np.concatenate([records, records]), 100), large
    )
    before, after = frame.pixels[48, 115:135, 2], (frame.pixels + large)[48, 115:135, 2]
    threshold = (before[0] + 0.02) / 2
    assert np.count_nonzero(after > threshold) > np.count_nonzero(before > threshold)
    labels = np.ones((96, 160), np.uint8)
    labels[48:] = 0
    masked = star_enlargement(frame.pixels, records, 100, sky_mask=SkyMask(1, labels, (96, 160)))
    assert not masked[48:].any()


def test_mask_and_missing_profiles_cannot_reduce_foreground():
    from starlab.domain.effects.soft_filter import reduction_records, star_reduction
    from starlab.domain.sky_mask import SkyMask

    frame, stars = scene()
    labels = np.ones((96, 160), np.uint8)
    labels[48:] = 0
    light = star_reduction(
        frame.pixels, reduction_records(stars), 100, sky_mask=SkyMask(1, labels, (96, 160))
    )
    assert not light[48:].any()
    assert not reduction_records((Star(1, 40, 48, 0.4, (0.2, 0.4, 0.6)),)).size


def test_export_includes_reduction_and_effects_only_remains_added_light(tmp_path):
    from starlab.application.use_cases import ImageServices
    from starlab.infrastructure.tiff_io import read_tiff, write_tiff

    frame, stars = scene()
    path = tmp_path / "result.tif"
    service = ImageServices(None, write_tiff)
    service.export(path, frame, stars, settings(100), False, None)
    p = render_preview(frame, stars, settings(100), full_resolution=True)
    np.testing.assert_allclose(
        read_tiff(path).pixels,
        composite(p.frame, p.effects, reduction=p.reduction, enlargement=p.enlargement),
        atol=2e-5,
    )
    service.export(path, frame, stars, settings(100), True, None)
    np.testing.assert_allclose(read_tiff(path).pixels, p.effects, atol=2e-5)


def test_disabling_soft_clears_reduction_from_overview_and_zoom_immediately(qtbot):
    from starlab.application.use_cases import Preview
    from starlab.bootstrap import create_window

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    frame, stars = scene()
    c = window.controller
    c.session.open(frame)
    c.session.set_settings(settings(100))
    glow = np.full_like(frame.pixels, 0.1)
    p = Preview(
        frame,
        glow,
        1,
        groups={"soft": glow},
        reduction=np.full_like(glow, 0.01),
        enlargement=np.full_like(glow, 0.02),
    )
    c.preview = c.overview = p
    c.change_settings(replace(settings(100), soft=replace(settings(100).soft, enabled=False)))
    assert c.preview.reduction is None and c.overview.reduction is None
    assert c.preview.enlargement is None and c.overview.enlargement is None
    assert not c.preview.effects.any() and not c.overview.effects.any()
    assert c.shutdown()


def test_manual_sources_keep_normal_glow_with_filter_enabled():
    frame, stars = scene()
    manual = tuple(
        Star(s.id, s.x, s.y, s.brightness, s.color, soft_weight=s.soft_weight) for s in stars
    )
    before = render_preview(frame, manual, settings(0), full_resolution=True)
    after = render_preview(frame, manual, settings(100), full_resolution=True)
    np.testing.assert_array_equal(after.effects, before.effects)
    assert not after.reduction.any()


def test_production_multiscale_measures_profiles(monkeypatch):
    from starlab.application import use_cases
    from starlab.domain.stars import DetectionSettings

    frame, stars = scene()
    raw = tuple(Star(s.id, s.x, s.y, s.brightness, s.color) for s in stars)
    monkeypatch.setattr(use_cases, "detect_stars", lambda *args, **kwargs: raw)
    measured = use_cases.ImageServices(None, None).analyze(
        frame, None, DetectionSettings(method="multiscale")
    )
    assert all(s.source_profile for s in measured)
    assert measured[0].soft_weight < measured[1].soft_weight


def test_uncertain_zero_signal_measurement_cannot_subtract_pixels(monkeypatch):
    from starlab.domain import photometry
    from starlab.domain.effects.soft_filter import reduction_records

    frame, stars = scene()
    row = np.zeros((1, 17))
    row[0, 4] = 3  # A width alone is insufficient if the fitted source has no positive flux.
    monkeypatch.setattr(photometry, "_measure_sources", lambda *args, **kwargs: row)
    measured = photometry.calibrate_sources(frame, stars[:1])
    assert measured[0].source_profile == ()
    assert measured[0].soft_weight == 0.5
    assert not reduction_records(measured).size


def test_catalog_change_clears_old_reduction(qtbot):
    from starlab.application.use_cases import Preview
    from starlab.bootstrap import create_window

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    c = window.controller
    frame, stars = scene()
    c.session.open(frame)
    glow = np.full_like(frame.pixels, 0.1)
    c.overview = c.preview = Preview(frame, glow, 1, reduction=glow, enlargement=glow)
    c._clear_effect_history()
    assert c.preview.reduction is None and c.overview.reduction is None
    assert c.preview.enlargement is None and c.overview.enlargement is None
    assert not c.preview.effects.any()
    assert c.shutdown()


def test_soft_filter_suppresses_small_star_halos_and_strengthens_large_star_halos():
    frame, stars = scene()
    previews = [
        render_preview(frame, stars, settings(value), full_resolution=True)
        for value in (0, 50, 100, 0, 100)
    ]
    small_halos = [preview.effects[48, 40, 2] for preview in previews]
    large_halos = [preview.effects[48, 115, 2] for preview in previews]
    assert small_halos[0] > small_halos[1] > small_halos[2]
    assert small_halos[2] == 0
    assert 0 < large_halos[0] < large_halos[1] < large_halos[2]
    np.testing.assert_array_equal(previews[0].effects, previews[3].effects)
    np.testing.assert_array_equal(previews[2].effects, previews[4].effects)
    # Suppressing the halo must still shrink the measured small source itself.
    assert previews[2].reduction[48, 42, 2] > 0


def _soft(**changes):
    base = EffectSettings.disabled()
    return replace(base, soft=replace(base.soft, enabled=True, soft_filter=100, **changes))


def _layers(stars, frame=None, **changes):
    frame = frame or scene()[0]
    preview = render_preview(frame, stars, _soft(**changes), full_resolution=True)
    return preview.reduction, preview.enlargement


def test_quantity_zero_disables_the_source_filter_and_quantity_gates_stars():
    frame, stars = scene()
    reduction, enlargement = _layers(stars, frame, quantity=0)
    assert not reduction.any() and not enlargement.any()
    # Same threshold as the glow: (1 - quantity / 100) ** 2 * brightest.
    small, large = replace(stars[0], brightness=0.05), stars[1]
    reduction, enlargement = _layers((small, large), frame, quantity=50)  # threshold 0.1
    assert not reduction.any() and enlargement.any()
    reduction, _ = _layers((small, large), frame, quantity=100)
    assert reduction.any()


def test_defaults_reproduce_the_original_halo_formula():
    frame, stars = scene()
    _, enlargement = _layers(stars, frame)  # Size 12, Edge falloff 60, no color change
    star = stars[1]  # weight 1: spread 4, sigma 4
    sigma, spread = star.source_profile[0] / 2.354820045, 4.0
    for dx in (2, 5, 9, 20, 30):
        d2 = (dx / sigma) ** 2
        limit = 9 * spread**2
        wings = (np.exp(-d2 / (2 * spread**2)) - np.exp(-d2 / 2)) * (1 - d2 / limit) ** 2
        expected = 0.8 * 1.0 * 0.6 * wings if d2 < limit else 0
        assert enlargement[48, 115 + dx, 2] == pytest.approx(expected, abs=2e-7)


def test_size_widens_the_halo_and_falloff_changes_its_tail():
    _, stars = scene()

    def radius(**changes):
        columns = np.flatnonzero(_layers(stars, **changes)[1][48, :, 2] > 1e-6)
        return 115 - columns.min()  # leftwards: the image edge is far away on this side

    assert radius(size=6) < radius(size=12) < radius(size=25) < radius(size=40)

    def tail(**changes):
        halo = _layers(stars, **changes)[1][48, :, 2]
        total = halo.sum()
        return halo[115 + 12 :].sum() / total  # beyond 3 sigma of the largest star (sigma 4)

    assert tail(falloff=0) < tail(falloff=60) < tail(falloff=100)


def test_color_controls_change_only_the_halo_color():
    _, stars = scene()
    plain = _layers(stars)[1]
    tinted = _layers(stars, hue=40, saturation=50)[1]
    assert not np.allclose(plain[48, 120] / plain[48, 120, 2], tinted[48, 120] / tinted[48, 120, 2])
    assert np.array_equal(plain.any(axis=2), tinted.any(axis=2))
    assert np.array_equal(_layers(stars)[0], _layers(stars, hue=40, saturation=50)[0])
