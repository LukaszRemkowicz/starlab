from dataclasses import replace

import numpy as np

from starlab.application.use_cases import render_preview
from starlab.domain.effects.settings import EffectSettings
from starlab.domain.imaging import ImageFrame
from starlab.domain.stars import Star


def test_drag_and_release_keep_the_same_stars_and_pixel_grid():
    frame = ImageFrame(np.zeros((120, 1800, 3), np.float32))
    stars = tuple(Star(i, 30 + i * 20.3, 60.2, 0.5, (0.5,) * 3) for i in range(70))
    settings = EffectSettings.disabled()
    settings = replace(settings, primary=replace(settings.primary, enabled=True, quantity=100))
    dragging = render_preview(frame, stars, settings, interactive=True)
    released = render_preview(frame, stars, settings)
    assert dragging.scale == released.scale
    assert dragging.draft_count is None
    np.testing.assert_array_equal(dragging.effects, released.effects)


def test_zoom_does_not_replace_a_ready_result_with_original(qtbot):
    from starlab.application.use_cases import Preview
    from starlab.bootstrap import create_window

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.show()
    controller = window.controller
    frame = ImageFrame(np.zeros((400, 800, 3), np.float32))
    controller.session.open(frame)
    controller.preview = controller.overview = Preview(frame, np.full_like(frame.pixels, 0.5), 1)
    window.set_loaded(True)
    controller.display()
    controller.exclusive = True  # Pause worker submission to inspect the immediate zoom frame.
    window.image_view.actual_size()
    detail = window.image_view.detail_item.pixmap().toImage()
    assert detail.isNull() or detail.pixelColor(10, 10).red() > 0
    assert controller.shutdown()


def test_disabled_spikes_never_return_from_overview_or_an_older_frame(qtbot):
    from starlab.application.use_cases import Preview
    from starlab.bootstrap import create_window

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.show()
    controller = window.controller
    frame = ImageFrame(np.zeros((400, 800, 3), np.float32))
    controller.session.open(frame)
    primary = np.zeros_like(frame.pixels)
    primary[..., 0] = 0.8
    soft = np.zeros_like(frame.pixels)
    soft[..., 1] = 0.2
    old = Preview(frame, primary + soft, 1, groups={"primary": primary, "soft": soft})
    controller.preview = controller.overview = old
    window.set_loaded(True)
    controller.display()
    old_revision = controller.session.revision
    settings = replace(
        controller.session.settings,
        primary=replace(controller.session.settings.primary, enabled=False),
    )
    controller.change_settings(settings)
    assert window.image_view.item.pixmap().toImage().pixelColor(20, 20).red() == 0
    assert window.image_view.item.pixmap().toImage().pixelColor(20, 20).green() > 0
    window.image_view.actual_size()
    controller.dragging = True
    controller.completed("preview", (old_revision, old))
    image = window.image_view.item.pixmap().toImage()
    assert image.pixelColor(20, 20).red() == 0
    assert controller.shutdown()
