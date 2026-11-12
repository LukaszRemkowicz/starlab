"""Exercise held QSlider mouse gestures through the real controller and worker."""

import time
from threading import Event
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QStyle, QStyleOptionSlider

if TYPE_CHECKING:
    from PySide6.QtWidgets import QSlider
    from pytestqt.qtbot import QtBot

    from starlab.presentation.qt.main_window import MainWindow


def _slider_window(qtbot: "QtBot") -> tuple["MainWindow", "QSlider", QPoint]:
    """Show a loaded image and return the visible intensity slider's handle center."""
    from starlab.bootstrap import create_window
    from starlab.domain.imaging import ImageFrame

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.show()
    window.controller.session.open(ImageFrame(np.zeros((32, 32, 3), np.float32)))
    window.set_loaded(True)
    window.effect_panel.tabs.setCurrentIndex(1)
    slider = window.effect_panel.controls["primary"]["intensity"].slider
    qtbot.waitUntil(slider.isVisible)
    option = QStyleOptionSlider()
    slider.initStyleOption(option)
    handle = slider.style().subControlRect(
        QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, slider
    )
    return window, slider, handle.center()


@pytest.mark.parametrize("busy_at_press", [False, True])
def test_slider_displays_frames_before_mouse_release(qtbot, monkeypatch, busy_at_press):
    from starlab.application.use_cases import Preview
    from starlab.domain.imaging import check_cancel
    from starlab.presentation.qt import controller as controller_module

    entered_full_preview = Event()
    allow_full_preview = Event()
    calls = []

    def render(frame, stars, settings, cancel, **options):
        interactive = options.get("interactive", False)
        calls.append(interactive)
        if busy_at_press and not interactive and not entered_full_preview.is_set():
            entered_full_preview.set()
            while not allow_full_preview.wait(0.01):
                check_cancel(cancel)
        time.sleep(0.045)
        check_cancel(cancel)
        return Preview(frame, np.full_like(frame.pixels, settings.primary.intensity / 200), 1)

    monkeypatch.setattr(controller_module, "render_preview", render)
    window, slider, start_position = _slider_window(qtbot)
    controller = window.controller
    position = start_position
    displayed = []
    original_set_pixels = window.image_view.set_pixels

    def show_pixels(*args, **kwargs):
        displayed.append((slider.isSliderDown(), np.array(args[0], copy=True)))
        return original_set_pixels(*args, **kwargs)

    monkeypatch.setattr(window.image_view, "set_pixels", show_pixels)
    if busy_at_press:
        controller.refresh()
        qtbot.waitUntil(entered_full_preview.is_set)
    try:
        qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=position)
        assert slider.isSliderDown()
        assert controller.dragging
        initial_value = slider.value()
        for offset in range(8, 89, 8):
            position = QPoint(start_position.x() + offset, start_position.y())
            qtbot.mouseMove(slider, pos=position)
            qtbot.wait(15)
        assert slider.value() != initial_value
        qtbot.waitUntil(lambda: len(displayed) >= 2, timeout=1500)
        assert slider.isSliderDown()
        assert all(held for held, _ in displayed)
        assert any(not np.array_equal(displayed[0][1], pixels) for _, pixels in displayed[1:])
        assert True in calls
    finally:
        allow_full_preview.set()
        qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=position)
        qtbot.waitUntil(
            lambda: not controller.runner.busy and not controller.debounce.isActive(), timeout=5000
        )
        assert controller.shutdown()
    assert not controller.dragging
    np.testing.assert_allclose(controller.preview.effects, slider.value() / 200)
    assert calls[-1] is False
