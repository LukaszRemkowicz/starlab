"""Small hand-calculated light and correction oracles, independent of renderer helpers."""

from dataclasses import replace

import numpy as np

from starlab.domain.effects.render import composite, render_effects
from starlab.domain.effects.settings import EffectSettings
from starlab.domain.effects.soft_filter import (
    enlargement_records,
    reduction_records,
    star_enlargement,
    star_reduction,
)
from starlab.domain.imaging import ImageFrame
from starlab.domain.stars import Star


def test_overlapping_sources_and_effect_groups_use_screen_products():
    frame = ImageFrame(np.zeros((9, 9, 3), np.float32))
    source = Star(1, 4, 4, 0.5, (0.5, 0.5, 0.5))
    settings = EffectSettings.disabled()
    settings = replace(
        settings,
        soft=replace(
            settings.soft, enabled=True, quantity=100, intensity=20, size=100, falloff=100 / 31
        ),
        # This size places the ring maximum three pixels from the source center.
        ring=replace(
            settings.ring, enabled=True, quantity=100, intensity=50, size=1.446818885022818
        ),
    )
    effects = render_effects(frame, (source, replace(source, id=2)), settings)
    # At radius 3, each source emits soft=0.02265408474576271 and
    # ring=0.16368951558707749. Four transmissions multiply, giving this value.
    np.testing.assert_allclose(effects[4, 7], [0.33191505210223634] * 3, atol=1e-7, rtol=0)


def test_composite_clamps_reduction_then_adds_wings_before_screen():
    frame = ImageFrame(np.array([[[0.2], [0.95]], [[0.1], [0.4]]], np.float32))
    reduction = np.array([[[0.05], [0.1]], [[0.2], [0]]], np.float32)
    enlargement = np.array([[[0.1], [0.05]], [[0.2], [0]]], np.float32)
    effects = np.array([[[0.4], [0.2]], [[0.5], [0]]], np.float32)
    actual = composite(frame, effects, reduction=reduction, enlargement=enlargement)
    # Corrected source values are 0.25, 0.90, 0.20, 0.40 before Screen.
    expected = np.array([[[0.55], [0.92]], [[0.60], [0.40]]], np.float32)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)


def test_source_corrections_match_closed_form_pixels_and_maximum_overlap():
    frame = ImageFrame(np.full((15, 15, 3), (0.31, 0.62, 0.93), np.float32))
    profile = (2.354820045, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
    small = Star(1, 7, 7, 0.4, (0.2, 0.4, 0.6), source_profile=profile, soft_weight=0)
    large = replace(small, soft_weight=1, source_peak=(0.2, 0.4, 0.6))
    reduction = star_reduction(
        frame.pixels, reduction_records((small, replace(small, id=2, soft_weight=0.25))), 50
    )
    enlargement = star_enlargement(
        frame.pixels, enlargement_records((large, replace(large, id=2, soft_weight=0.75))), 50
    )
    # Sigma=1, distance=1 and strength=50%; the plane leaves RGB residuals
    # (0.2, 0.4, 0.6) to the right and (0.17, 0.37, 0.57) below the source.
    # Closed-form values below follow the soft-filter formulas (the larger of the two overlaps).
    np.testing.assert_allclose(
        reduction[[7, 8], [8, 7]],
        [
            [0.05357532700859688, 0.10715065401719376, 0.1607259810257906],
            [0.04553902795730735, 0.09911435496590422, 0.15268968197450108],
        ],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        enlargement[7, 8],
        [0.028614602435737352, 0.057229204871474704, 0.08584380730721206],
        atol=1e-8,
        rtol=0,
    )
    assert not reduction[7, 7].any() and not enlargement[7, 7].any()
    assert not reduction[7, 10].any()
