"""Continuous slider input must produce frames before the user releases it."""

import time

import numpy as np


def test_drag_displays_completed_frames_and_finishes_with_latest_value(qtbot, monkeypatch):
    from starlab.application.use_cases import Preview
    from starlab.bootstrap import create_window
    from starlab.domain.imaging import ImageFrame
    from starlab.presentation.qt import controller as controller_module

    calls = []

    def slow_preview(frame, stars, settings, cancel, *, full_resolution=False, interactive=False):
        calls.append((interactive, full_resolution))
        time.sleep(0.045)
        return Preview(frame, np.full_like(frame.pixels, settings.primary.intensity / 200), 1)

    monkeypatch.setattr(controller_module, "render_preview", slow_preview)
    window = create_window()
    qtbot.addWidget(window)
    controller = window.controller
    controller.session.open(ImageFrame(np.zeros((32, 32, 3), np.float32)))
    controller.full_resolution = True
    controller.begin_edit()
    for value in range(50, 62):
        window.effect_panel.controls["primary"]["intensity"].spin.setValue(value)
        qtbot.wait(12)
    assert controller.preview is not None  # A frame appeared during the drag.
    assert calls and all(interactive and not full for interactive, full in calls)
    controller.end_edit()
    qtbot.waitUntil(lambda: not controller.runner.busy and not controller.debounce.isActive())
    np.testing.assert_allclose(controller.preview.effects, 61 / 200)
    assert calls[-1] == (False, True)
    assert controller.shutdown()


def test_sampled_pixmap_center_matches_full_image_pixel(qtbot):
    from starlab.presentation.qt.image_view import ImageView

    view = ImageView()
    qtbot.addWidget(view)
    view.set_pixels(np.zeros((8, 8, 3), np.float32), "linear-srgb", 11, 88, 88)
    from PySide6.QtCore import QPointF

    center = view.item.mapToScene(QPointF(3.5, 2.5))
    assert center.x() == 33.5
    assert center.y() == 22.5
