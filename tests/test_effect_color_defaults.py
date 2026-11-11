from dataclasses import replace

import numpy as np

from starlab.domain.effects.profiles import source_colors
from starlab.domain.effects.render import composite, render_effects
from starlab.domain.effects.settings import KINDS, EffectSettings
from starlab.domain.imaging import ImageFrame
from starlab.domain.presets import parse_preset, preset_data
from starlab.domain.stars import Star


def test_default_color_does_not_add_saturation():
    settings = EffectSettings()
    original = np.array([[0.2, 0.4, 0.6]], np.float32)
    for kind in KINDS:
        params = getattr(settings, kind)
        assert params.saturation == 0
        assert params.rainbow == 0
        np.testing.assert_array_equal(
            source_colors(original, params.hue, params.saturation), original
        )


def test_explicit_saturation_is_preserved_and_affects_only_effect_light():
    frame = ImageFrame(np.full((128, 128, 3), (0.02, 0.04, 0.06), np.float32))
    star = Star(1, 64, 64, 0.4, (0.2, 0.4, 0.6))
    settings = EffectSettings.disabled()
    settings = replace(
        settings, soft=replace(settings.soft, enabled=True, saturation=30, size=20, intensity=40)
    )
    assert parse_preset(preset_data(settings)).soft.saturation == 30
    effects = render_effects(frame, (star,), settings)
    assert effects.max() > 0
    # Saturation 30 maps this source to RGB (0, 0.4, 2.1) before scalar gains.
    np.testing.assert_allclose(effects[64, 64] / effects[64, 64, 2], [0, 4 / 21, 1], rtol=1e-6)
    neutral = render_effects(
        frame, (star,), replace(settings, soft=replace(settings.soft, saturation=0))
    )
    np.testing.assert_allclose(neutral[64, 64] / neutral[64, 64, 2], [1 / 3, 2 / 3, 1], rtol=1e-6)
    assert not np.allclose(effects, neutral)
    result = composite(frame, effects)
    np.testing.assert_array_equal(result[~effects.any(axis=2)], frame.pixels[~effects.any(axis=2)])
    np.testing.assert_array_equal(frame.pixels[0, 0], np.array([0.02, 0.04, 0.06], np.float32))
