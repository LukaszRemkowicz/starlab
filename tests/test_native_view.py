from threading import Event

import numpy as np
import pytest

from starlab.application.use_cases import ImageServices, Preview
from starlab.domain.imaging import ImageFrame, display_pixels


@pytest.mark.parametrize("preview_scale", [8, 1 / 8])
def test_thumbnail_is_visible_before_full_import_finishes(qtbot, preview_scale):
    from starlab.bootstrap import create_window

    window = create_window()
    qtbot.addWidget(window)
    window.auto_detect.setChecked(False)
    release = Event()
    small = ImageFrame(np.full((16, 32, 3), 0.1, np.float32))
    full = ImageFrame(np.full((128, 256, 3), 0.1, np.float32))

    def read(path, cancel):
        release.wait(5)
        return full

    window.controller.services = ImageServices(
        read, None, read_preview=lambda *_: (small, preview_scale, 256, 128)
    )
    window.show()
    try:
        window.controller.open_path("image.tif")
        qtbot.waitUntil(lambda: not window.image_view.item.pixmap().isNull())
        assert window.stack.currentIndex() == 1
        assert window.image_view.item.scale() == 8
        assert window.controller.session.frame is None
        assert window.controller.runner.busy
    finally:
        release.set()
    qtbot.waitUntil(
        lambda: window.controller.session.frame is full and not window.controller.runner.busy
    )
    assert window.controller.shutdown()


def test_original_zoom_uses_source_pixels_not_thumbnail(qtbot):
    from starlab.bootstrap import create_window

    y, x = np.mgrid[:256, :512]
    data = np.repeat(((x % 8) / 7).astype(np.float32)[..., None], 3, axis=2)
    frame = ImageFrame(data)
    thumbnail = ImageFrame(frame.pixels[::8, ::8])
    window = create_window()
    qtbot.addWidget(window)
    window.show()
    controller = window.controller
    controller.session.open(frame)
    controller.preview = controller.overview = Preview(
        thumbnail, np.zeros_like(thumbnail.pixels), 8
    )
    window.set_loaded(True)
    window._view("original")
    window.image_view.actual_size()
    detail = window.image_view.detail_item.pixmap().toImage()
    assert not detail.isNull()
    assert window.image_view.detail_item.scale() == 1
    (left, top, _, _), sampling = controller.detail_region
    assert sampling == 1
    expected = display_pixels(frame.pixels, frame.color_space)
    assert detail.pixelColor(3, 3).red() == int(expected[top + 3, left + 3, 0])
    assert detail.pixelColor(3, 3).red() != 0  # The stride8 thumbnail contains only zeros.
    assert not controller.runner.busy  # Original viewing doesn't wait for effect rendering.

    window._view("result")
    qtbot.waitUntil(lambda: controller.preview.is_region and not controller.runner.busy)
    result = window.image_view.detail_item.pixmap().toImage()
    window._view("original")
    window._view("result")
    assert not controller.runner.busy  # Hold for Before reuses an already rendered ROI.
    assert window.image_view.detail_item.pixmap().toImage() == result
    window.image_view.fit_image()
    qtbot.waitUntil(lambda: not controller.runner.busy and not controller.preview.is_region)
    assert controller.detail_region is None
    assert window.image_view.detail_item.pixmap().isNull()
    assert controller.shutdown()


def test_detector_and_effect_counts_are_distinct(qtbot):
    from dataclasses import replace

    from starlab.bootstrap import create_window
    from starlab.domain.effects.settings import EffectSettings
    from starlab.domain.stars import Star

    window = create_window()
    qtbot.addWidget(window)
    stars = (Star(1, 1, 1, 0.87, (0.87,) * 3), Star(2, 2, 2, 0.001, (0.001,) * 3))
    settings = EffectSettings.disabled()
    settings = replace(settings, primary=replace(settings.primary, enabled=True))
    window.controller.update_star_count(stars, settings)
    assert window.star_count.text() == "2 detected · 1 with effects"
    assert window.effect_panel.source_counts["primary"].text() == "1 / 2"
    settings = replace(settings, primary=replace(settings.primary, quantity=100))
    window.controller.update_star_count(stars, settings)
    assert window.star_count.text() == "2 detected · 2 with effects"
    assert window.controller.shutdown()
