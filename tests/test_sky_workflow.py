from dataclasses import replace

import numpy as np

from starlab.bootstrap import create_window
from starlab.domain.imaging import ImageFrame
from starlab.domain.sky_mask import SkyMask
from starlab.domain.stars import Star


def test_mask_precedes_detection_and_sliders_reuse_it(qtbot, monkeypatch):
    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    controller = window.controller
    frame = ImageFrame(np.zeros((64, 96, 3), np.float32))
    controller.session.open(frame)
    mask = SkyMask(
        1, np.repeat((np.arange(64) < 40)[:, None], 96, axis=1).astype(np.uint8), (64, 96)
    )
    calls = []

    def mask_service(*args):
        calls.append("mask")
        return mask

    def analyze(*args, **kwargs):
        assert calls == ["mask"] and kwargs["sky_mask"] is mask
        calls.append("detect")
        return (Star(1, 40, 25, 0.8, (0.8,) * 3),)

    controller.services.sky_mask = mask_service
    monkeypatch.setattr(controller.services, "analyze", analyze)
    controller.analyze(force=False)
    qtbot.waitUntil(lambda: not controller.runner.busy)
    assert controller.session.sky_mask is mask
    controller.change_settings(
        replace(
            controller.session.settings,
            primary=replace(controller.session.settings.primary, intensity=60),
        )
    )
    qtbot.waitUntil(lambda: not controller.runner.busy and not controller.debounce.isActive())
    assert calls == ["mask", "detect"]
    assert not controller.preview.effects[40:].any()
    assert controller.shutdown()


def test_catalog_undo_restores_its_mask():
    from starlab.application.session import Session
    from starlab.domain.stars import DetectionSettings

    session = Session()
    session.open(ImageFrame(np.zeros((8, 8, 3), np.float32)))
    mask = SkyMask(2, np.ones((4, 4), np.uint8), (8, 8))
    settings = DetectionSettings(sky_only=True, method="multiscale", threshold_sigma=7)
    session.set_catalog((Star(1, 3, 3, 1, (1,) * 3),), sky_mask=mask, detection_settings=settings)
    session.set_catalog((), sky_mask=None, detection_settings=DetectionSettings())
    assert session.undo() and session.sky_mask is mask
    assert session.detection_settings == settings
    assert session.redo() and session.sky_mask is None
    assert session.detection_settings == DetectionSettings()


def test_detector_selector_exposes_larger_stars_with_correct_units(qtbot):
    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.detection_method.setCurrentIndex(1)
    assert window.detection_method.currentData() == "multiscale"
    assert window.threshold.value() == 5 and "σ" in window.threshold.suffix()
    window.threshold.setValue(7)
    window.detection_method.setCurrentIndex(0)
    assert window.threshold.value() == 0.2 and window.threshold.suffix() == ""
    window.detection_method.setCurrentIndex(1)
    assert window.threshold.value() == 7
    assert window.controller.shutdown()


def test_cancel_after_mask_restores_committed_catalog_controls(qtbot, monkeypatch):
    from threading import Event

    from starlab.domain.imaging import check_cancel

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    controller = window.controller
    controller.session.open(ImageFrame(np.zeros((32, 48, 3), np.float32)))
    controller.session.set_catalog((Star(1, 20, 25, 0.5, (0.5,) * 3),))
    mask = SkyMask(
        2, np.repeat((np.arange(16) < 8)[:, None], 24, axis=1).astype(np.uint8), (32, 48)
    )
    controller.services.sky_mask = lambda *_: mask
    entered = Event()

    def analyze(frame, source, settings, cancel, **kwargs):
        entered.set()
        cancel.wait(3)
        check_cancel(cancel)
        return ()

    monkeypatch.setattr(controller.services, "analyze", analyze)
    controller.analyze(force=False)
    qtbot.waitUntil(entered.is_set)
    assert controller.mask_preview is mask
    controller.cancel()
    qtbot.waitUntil(lambda: not controller.runner.busy)
    assert controller.session.sky_mask is None
    assert controller.mask_preview is None
    assert window.detection_method.currentData() == "native"
    assert controller.shutdown()


def test_mask_icon_restores_view_without_reanalysis(qtbot):
    from PySide6.QtCore import Qt

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.show()
    controller = window.controller
    controller.session.open(ImageFrame(np.zeros((64, 96, 3), np.float32)))
    mask = SkyMask(1, np.ones((64, 96), np.uint8), (64, 96))
    stars = (Star(1, 40, 25, 0.8, (0.8,) * 3),)
    controller.session.set_catalog(stars, sky_mask=mask)
    window.set_loaded(True)
    controller.refresh()
    qtbot.waitUntil(lambda: not controller.runner.busy)
    jobs = []
    controller.runner.activity.connect(jobs.append)
    window._view("effects")
    qtbot.mouseClick(window.mask_button, Qt.MouseButton.LeftButton)
    assert window.mode == "mask" and window.mask_button.isChecked()
    qtbot.mouseClick(window.mask_button, Qt.MouseButton.LeftButton)
    assert window.mode == "effects" and not window.mask_button.isChecked()
    window._view("mask")  # Dropdown and icon stay synchronized.
    window._before_pressed()
    window._before_released()
    assert window.mode == "mask" and window.mask_button.isChecked()
    qtbot.mouseClick(window.mask_button, Qt.MouseButton.LeftButton)
    assert window.mode == "effects" and not window.mask_button.isChecked()
    assert jobs == [] and controller.session.stars is stars and controller.session.sky_mask is mask
    assert controller.shutdown()


def test_open_without_automatic_detection_still_masks_and_keeps_mask_on_cancel(qtbot):
    from starlab.application.use_cases import ImageServices
    from starlab.presentation.qt.controller import Controller
    from starlab.presentation.qt.main_window import MainWindow

    frame = ImageFrame(np.zeros((32, 48, 3), np.float32))
    mask = SkyMask(1, np.ones((32, 48), np.uint8), (32, 48))
    window = MainWindow()
    qtbot.addWidget(window)
    controller = Controller(
        window,
        ImageServices(read=lambda *_: frame, write=None, sky_mask=lambda *_: mask),
        None,
        None,
    )
    window.controller = controller
    window.auto_detect.setChecked(False)
    controller.open_path("unused.tif")
    qtbot.waitUntil(lambda: not controller.runner.busy)
    assert controller.session.sky_mask is mask and controller.session.stars == ()
    assert not controller.session.undo(), "Initial mask must not be removable through Undo."
    controller.cancel()
    assert controller.session.sky_mask is mask
    assert controller.shutdown()


def test_import_overlaps_mask_and_detection_reuses_prepared_mask(qtbot):
    from threading import Event

    from starlab.application.use_cases import ImageServices
    from starlab.presentation.qt.controller import Controller
    from starlab.presentation.qt.main_window import MainWindow

    frame = ImageFrame(np.zeros((32, 48, 3), np.float32))
    mask = SkyMask(1, np.ones((32, 48), np.uint8), (32, 48))
    reading, segmenting = Event(), Event()

    def read(*_):
        reading.set()
        assert segmenting.wait(1), "Mask did not run alongside import"
        return frame

    def read_mask(*_):
        segmenting.set()
        assert reading.wait(1), "Import did not run alongside mask"
        return mask

    def repeated_mask(*_):
        raise AssertionError("Mask was calculated twice")

    window = MainWindow()
    qtbot.addWidget(window)
    window.controller = Controller(
        window, ImageServices(read, None, sky_mask=repeated_mask, read_mask=read_mask), None, None
    )
    window.controller.open_path("unused.tif")
    qtbot.waitUntil(lambda: not window.controller.runner.busy)
    assert not window.error_label.text(), window.error_label.text()
    assert window.controller.session.sky_mask is mask
    assert window.controller.session.detection_settings.sky_only
    assert window.controller.shutdown()
