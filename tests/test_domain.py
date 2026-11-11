from dataclasses import replace
from threading import Event

import numpy as np
import pytest


def sample():
    from starlab.domain.imaging import ImageFrame

    y, x = np.mgrid[:96, :128]
    pixels = 0.01 + 0.7 * np.exp(-((x - 53.3) ** 2 + (y - 41.7) ** 2) / 5)
    return ImageFrame(pixels[..., None].astype(np.float32), "test", "linear-gray")


def test_detector_recovered_position_and_hot_pixel_rejection():
    from starlab.domain.stars import DetectionSettings, detect_stars

    frame = sample()
    stars = detect_stars(frame, DetectionSettings())
    assert len(stars) == 1
    # Original96A40 on this prepared proxy: raw catalog(54.055557,42.138889).
    # The recovered score maximizer differs from a Gaussian center of mass.
    assert abs(stars[0].x - 53.55555725097656) < 1e-5
    assert abs(stars[0].y - 41.63888931274414) < 1e-5
    from starlab.domain.imaging import ImageFrame

    hot = np.zeros((64, 64, 1), np.float32)
    hot[30, 30] = 1
    assert detect_stars(ImageFrame(hot), DetectionSettings()) == ()
    assert detect_stars(ImageFrame(np.zeros_like(hot)), DetectionSettings()) == ()


def test_frame_owns_readonly_finite_pixels():
    from starlab.domain.imaging import ImageFrame

    original = np.zeros((8, 8, 3), np.float32)
    frame = ImageFrame(original)
    original[:] = 1
    assert frame.pixels.max() == 0
    with pytest.raises(ValueError):
        frame.pixels[0, 0] = 1
    with pytest.raises(ValueError):
        ImageFrame(np.full((8, 8, 3), np.nan, np.float32))


@pytest.mark.parametrize("kind", ["primary", "secondary", "soft", "ring"])
def test_each_effect_is_nonzero_deterministic_and_preserves_input(kind):
    from starlab.domain.effects.render import render_effects
    from starlab.domain.effects.settings import EffectSettings
    from starlab.domain.stars import detect_stars

    frame = sample()
    stars = detect_stars(frame)
    settings = EffectSettings.disabled()
    settings = replace(settings, **{kind: replace(getattr(settings, kind), enabled=True)})
    before = frame.pixels.copy()
    first = render_effects(frame, stars, settings)
    assert first.shape == frame.pixels.shape
    assert np.isfinite(first).all()
    assert first.max() > 0
    np.testing.assert_array_equal(first, render_effects(frame, stars, settings))
    np.testing.assert_array_equal(frame.pixels, before)


def test_disabled_effects_are_identity_and_cancellation_is_cooperative():
    from starlab.domain.effects.render import composite, render_effects
    from starlab.domain.effects.settings import EffectSettings
    from starlab.domain.imaging import Cancelled
    from starlab.domain.stars import detect_stars

    frame = sample()
    effects = render_effects(frame, detect_stars(frame), EffectSettings.disabled())
    assert not effects.any()
    np.testing.assert_array_equal(composite(frame, effects), frame.pixels)
    token = Event()
    token.set()
    with pytest.raises(Cancelled):
        render_effects(frame, (), EffectSettings(), token)


def test_invalid_parameters_rejected_and_edge_star_renders():
    from starlab.domain.effects.render import render_effects
    from starlab.domain.effects.settings import EffectParameters, EffectSettings
    from starlab.domain.stars import Star

    with pytest.raises(ValueError):
        EffectParameters(intensity=float("nan"))
    with pytest.raises(ValueError):
        EffectParameters(points=0)
    result = render_effects(sample(), (Star(1, 0.2, 0.1, 1, (1, 1, 1), 2),), EffectSettings())
    assert result.max() > 0


def test_srgb_transfer_round_trip():
    from starlab.domain.imaging import linear_to_srgb, srgb_to_linear

    x = np.array([0, 0.04045, 0.5, 1], np.float32)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-6)


@pytest.mark.parametrize("kind", ["primary", "secondary", "soft", "ring"])
def test_reduced_preview_matches_full_resolution_effect_geometry(kind):
    from dataclasses import replace

    from starlab.application.use_cases import render_preview
    from starlab.domain.effects.render import render_effects
    from starlab.domain.effects.settings import EffectSettings
    from starlab.domain.imaging import ImageFrame
    from starlab.domain.stars import Star

    frame = ImageFrame(np.zeros((128, 1700, 3), np.float32))
    stars = (Star(1, 800.3, 64.7, 1, (0.6, 0.8, 1), 2),)
    settings = EffectSettings.disabled()
    settings = replace(settings, **{kind: replace(getattr(settings, kind), enabled=True)})
    preview = render_preview(frame, stars, settings)
    assert preview.scale == 2
    full = render_effects(frame, stars, settings)
    np.testing.assert_allclose(preview.effects, full[::2, ::2], atol=1e-6)
