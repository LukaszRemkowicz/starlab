from dataclasses import replace

import numpy as np
import pytest

from starlab.application.use_cases import render_preview
from starlab.domain.effects.render import render_effects
from starlab.domain.effects.settings import EffectSettings
from starlab.domain.imaging import ImageFrame
from starlab.domain.sky_mask import SkyMask
from starlab.domain.stars import Star


@pytest.mark.parametrize("kind", ("primary", "secondary", "soft", "ring"))
def test_mask_blocks_ground_light_and_ground_maximum_in_full_and_roi(kind):
    frame = ImageFrame(np.zeros((128, 256, 3), np.float32))
    mask = SkyMask(
        1, np.repeat((np.arange(128) < 64)[:, None], 256, axis=1).astype(np.uint8), (128, 256)
    )
    sky = Star(1, 100.3, 52.7, 0.4, (0.2, 0.4, 0.6))
    ground = Star(2, 120, 90, 10, (1, 1, 1))
    settings = EffectSettings.disabled()
    settings = replace(
        settings,
        **{kind: replace(getattr(settings, kind), enabled=True, quantity=80, length=60, size=40)},
    )
    expected = render_effects(frame, (sky,), settings, sky_mask=mask)
    actual = render_effects(frame, (sky, ground), settings, sky_mask=mask)
    assert expected[:64].max() > 0
    assert not actual[64:].any()
    np.testing.assert_array_equal(actual, expected)
    preview = render_preview(
        frame, (sky, ground), settings, sky_mask=mask, region=(30, 40, 180, 100), region_sampling=3
    )
    np.testing.assert_allclose(preview.effects, expected[40:100:3, 30:180:3], atol=1e-6)


def test_export_keeps_original_ground_pixels(tmp_path):
    from starlab.application.use_cases import ImageServices

    frame = ImageFrame(np.full((64, 96, 3), 0.1, np.float32))
    mask = SkyMask(
        1, np.repeat((np.arange(64) < 32)[:, None], 96, axis=1).astype(np.uint8), (64, 96)
    )
    stars = (Star(1, 48, 30, 0.4, (0.4,) * 3), Star(2, 60, 40, 10, (1,) * 3))
    captured = []
    services = ImageServices(
        read=None,
        write=lambda path, frame, bands, **kwargs: captured.append(np.concatenate(list(bands))),
    )
    services.export(
        tmp_path / "result.tif", frame, stars, EffectSettings(), False, None, sky_mask=mask
    )
    np.testing.assert_array_equal(captured[0][32:], frame.pixels[32:])
    assert np.any(captured[0][:32] > frame.pixels[:32])
