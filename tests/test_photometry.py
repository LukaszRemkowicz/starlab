from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from starlab.domain.imaging import Cancelled, ImageFrame
from starlab.domain.sky_mask import SkyMask
from starlab.domain.stars import Star


def field(widths, *, gradient=False, phase=(0.3, 0.7)):
    yy, xx = np.mgrid[:144, : len(widths) * 96]
    pixels = np.full(xx.shape, 0.02, np.float64)
    if gradient:
        pixels += 0.001 * xx + 0.0007 * yy
    stars = []
    for index, sigma in enumerate(widths):
        x, y = index * 96 + 48 + phase[0], 70 + phase[1]
        pixels += (
            5 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        )
        # Deliberately wrong recovered strength/size: neither should define flux.
        stars.append(Star(index + 1, x, y, 0.8 / sigma**2, (0.8 / sigma**2,) * 3, 1))
    return ImageFrame(np.repeat(pixels[..., None], 3, axis=2).astype(np.float32)), tuple(stars)


@pytest.mark.parametrize("gradient", [False, True])
def test_equal_flux_stars_get_equal_strength_across_width_and_background(gradient):
    from starlab.domain.photometry import calibrate_sources, measure_source_signals

    frame, stars = field((1, 1.5, 2, 3, 5), gradient=gradient)
    measured = measure_source_signals(frame, stars)
    np.testing.assert_allclose(measured, 5, rtol=0.04)
    result = calibrate_sources(frame, stars)
    assert np.ptp([s.brightness for s in result]) < 0.04
    assert [(s.id, s.x, s.y, s.radius) for s in result] == [
        (s.id, s.x, s.y, s.radius) for s in stars
    ]


@pytest.mark.parametrize("phase", [(0.3, 0.3), (0.7, 0.7), (0.5, 0.5), (0.7, 0.3)])
def test_subpixel_background_gradient_does_not_change_integrated_signal(phase):
    from starlab.domain.photometry import measure_source_signals

    frame, stars = field((1, 3, 5), gradient=True, phase=phase)
    np.testing.assert_allclose(measure_source_signals(frame, stars), 5, rtol=0.02)


def test_strength_is_monotonic_and_preserves_source_colour_ratios():
    from starlab.domain.photometry import calibrate_sources

    frame, stars = field((2, 2))
    pixels = 0.02 + (frame.pixels - 0.02) * np.array([0.5, 1, 0.75], np.float32)
    pixels[:, 96:] = 0.02 + (pixels[:, 96:] - 0.02) * 0.5
    stars = tuple(replace(s, color=(0.8, 0.8, 0.8)) for s in stars)
    result = calibrate_sources(ImageFrame(pixels), stars)
    assert result[0].brightness == pytest.approx(2 * result[1].brightness, rel=0.01)
    for star in result:
        np.testing.assert_allclose(np.array(star.color) / star.color[1], [0.5, 1, 0.75])


def test_effect_colour_comes_from_original_pixels_after_subtracting_coloured_background():
    from starlab.domain.effects.render import render_effects
    from starlab.domain.effects.settings import EffectSettings
    from starlab.domain.photometry import calibrate_sources

    yy, xx = np.mgrid[:96, :96]
    background = np.array([0.12, 0.04, 0.01])
    gradient = xx[..., None] * [0.0007, 0.0003, 0.0001]
    profile = np.exp(-((xx - 48.3) ** 2 + (yy - 48.3) ** 2) / 18) / (18 * np.pi)
    pixels = background + gradient + profile[..., None] * [1, 3, 8]
    frame = ImageFrame(pixels.astype(np.float32))
    # The analysis proxy incorrectly supplied a white colour for this blue star.
    stars = (Star(1, 48.3, 48.3, 0.4, (0.4,) * 3),)
    result = calibrate_sources(frame, stars)
    np.testing.assert_allclose(
        np.array(result[0].color) / result[0].color[2], [0.125, 0.375, 1], rtol=0.02
    )
    settings = EffectSettings.disabled()
    settings = replace(
        settings,
        soft=replace(settings.soft, enabled=True, quantity=100, intensity=62, size=29, falloff=40),
    )
    effects = render_effects(frame, result, settings)
    light = effects.sum(axis=(0, 1))
    np.testing.assert_allclose(light / light[2], [0.125, 0.375, 1], rtol=0.02)


def test_foreground_and_flat_background_do_not_supply_source_signal():
    from starlab.domain.photometry import measure_source_signals

    frame, stars = field((2,))
    labels = np.ones((144, 96), np.uint8)
    labels[85:] = 0
    mask = SkyMask(1, labels, (144, 96))
    pixels = frame.pixels.copy()
    baseline = measure_source_signals(frame, stars, sky_mask=mask)
    pixels[85:] = 1
    np.testing.assert_array_equal(
        measure_source_signals(ImageFrame(pixels), stars, sky_mask=mask), baseline
    )
    flat = ImageFrame(np.full((144, 96, 3), 0.1, np.float32))
    assert measure_source_signals(flat, stars)[0] < 1e-6


def test_nearby_source_does_not_change_flux_of_separated_stars():
    from starlab.domain.photometry import measure_source_signals

    yy, xx = np.mgrid[:96, :96]
    pixels = np.full((96, 96), 0.03)
    stars = (Star(1, 40, 48, 0.4, (0.4,) * 3), Star(2, 52, 48, 0.4, (0.4,) * 3))
    for s in stars:
        pixels += 5 / (8 * np.pi) * np.exp(-((xx - s.x) ** 2 + (yy - s.y) ** 2) / 8)
    signals = measure_source_signals(
        ImageFrame(np.repeat(pixels[..., None], 3, axis=2).astype(np.float32)), stars
    )
    np.testing.assert_allclose(signals, 5, rtol=0.06)


def test_calibration_cancellation_and_empty_catalogue():
    from starlab.domain.photometry import calibrate_sources

    frame, stars = field((2,))
    token = Event()
    token.set()
    with pytest.raises(Cancelled):
        calibrate_sources(frame, stars, cancel=token)
    assert calibrate_sources(frame, ()) == ()


def test_no_background_support_never_treats_sky_brightness_as_stellar_flux():
    from starlab.domain.photometry import measure_source_signals

    frame = ImageFrame(np.full((96, 96, 3), 0.4, np.float32))
    stars = (Star(1, 48, 48, 0.5, (0.5,) * 3),)
    labels = np.zeros((96, 96), np.uint8)
    labels[46:51, 46:51] = 1
    assert measure_source_signals(frame, stars, sky_mask=SkyMask(1, labels, (96, 96)))[0] == 0


def test_application_analysis_applies_photometry(monkeypatch):
    from starlab.application import use_cases

    frame, stars = field((1, 3))
    monkeypatch.setattr(use_cases, "detect_stars", lambda *args, **kwargs: stars)
    result = use_cases.ImageServices(None, None).analyze(frame, None)
    assert result[0].brightness == pytest.approx(result[1].brightness, rel=0.04)


@pytest.mark.parametrize("x", [1.3, 3.3])
def test_border_source_flux_matches_interior_source(x):
    from starlab.domain.photometry import measure_source_signals

    yy, xx = np.mgrid[:144, :96]
    sigma, y = 2.0, 70.7
    pixels = 0.02 + 5 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)) / (
        2 * np.pi * sigma**2
    )
    frame = ImageFrame(np.repeat(pixels[..., None], 3, axis=2).astype(np.float32))
    star = Star(1, x, y, 0.2, (0.2,) * 3, 1)
    np.testing.assert_allclose(measure_source_signals(frame, (star,)), 5, rtol=0.05)


def test_coincident_sources_split_flux_instead_of_counting_it_twice():
    from starlab.domain.photometry import measure_source_signals

    frame, stars = field((2,))
    twin = replace(stars[0], id=2)
    assert measure_source_signals(frame, (stars[0], twin)).sum() == pytest.approx(5, rel=0.05)


def test_corner_source_flux_matches_interior_source():
    from starlab.domain.photometry import measure_source_signals

    yy, xx = np.mgrid[:96, :96]
    sigma = 2.0
    pixels = 0.02 + 5 * np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    frame = ImageFrame(np.repeat(pixels[..., None], 3, axis=2).astype(np.float32))
    star = Star(1, 0.0, 0.0, 0.2, (0.2,) * 3, 1)
    np.testing.assert_allclose(measure_source_signals(frame, (star,)), 5, rtol=0.05)


def test_identical_close_sources_get_identical_flux():
    from starlab.domain.photometry import measure_source_signals

    yy, xx = np.mgrid[:96, :160]
    pixels = np.full(xx.shape, 0.02)
    stars = []
    for index, x in enumerate((77.0, 83.0)):
        pixels += 5 * np.exp(-((xx - x) ** 2 + (yy - 48.0) ** 2) / 8) / (8 * np.pi)
        stars.append(Star(index + 1, x, 48.0, 0.2, (0.2,) * 3, 1))
    frame = ImageFrame(np.repeat(pixels[..., None], 3, axis=2).astype(np.float32))
    first, second = measure_source_signals(frame, tuple(stars))
    assert first == pytest.approx(second, rel=0.01)


def test_coincident_sources_share_flux_equally():
    from starlab.domain.photometry import measure_source_signals

    frame, stars = field((2,))
    measured = measure_source_signals(frame, (stars[0], replace(stars[0], id=2)))
    np.testing.assert_allclose(measured, 2.5, rtol=0.05)
