from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QDialog

from starlab.application.use_cases import Preview
from starlab.bootstrap import create_window
from starlab.domain.imaging import ImageFrame
from starlab.domain.sky_mask import SkyMask


@pytest.fixture
def document(qtbot):
    window = create_window()
    qtbot.addWidget(window)
    window.show()
    controller = window.controller
    frame = ImageFrame(np.full((120, 240, 3), 0.1, np.float32), "old.tif")
    controller.session.open(frame)
    controller.source = Path("old.tif")
    controller.preview = controller.overview = Preview(frame, np.full_like(frame.pixels, 0.4), 1)
    window.filename.setText("old.tif")
    window.filename.setToolTip("old source")
    window.image_info.setText("old image info")
    window.image_info.setToolTip("old color info")
    window.set_loaded(True)
    controller.display()
    yield window, controller, frame
    assert controller.shutdown()


@pytest.mark.parametrize("termination", ["cancel", "failure"])
def test_interrupted_progressive_open_keeps_committed_document(document, monkeypatch, termination):
    window, controller, frame = document
    controller.session.add_star(10, 20)
    controller.session.set_settings(
        replace(
            controller.session.settings,
            primary=replace(controller.session.settings.primary, intensity=99),
        )
    )
    session, preview, source = controller.session, controller.preview, controller.source
    view = window.image_view
    view.fitted = False
    view.scale(2, 2)
    view.horizontalScrollBar().setValue(15)
    controller.detail_region = ((0, 0, 120, 80), 1)
    transform = view.transform()
    scroll = (view.horizontalScrollBar().value(), view.verticalScrollBar().value())
    overview_pixels = view.item.pixmap().toImage()
    monkeypatch.setattr(controller.runner, "submit", lambda *_: None)
    thumbnail = ImageFrame(np.zeros((20, 40, 3), np.float32))
    controller.completed("open_preview", (Path("new.tif"), (thumbnail, 6, 240, 120), None))
    assert controller.session is session and controller.session.frame is frame
    assert controller.source == source and controller.preview is preview
    # Thumbnail navigation must not change the committed document's ROI.
    view.actual_size()
    if termination == "cancel":
        controller.cancel()
    else:
        controller._failed("full read failed")
    controller.idle()
    assert controller.session.frame is frame and controller.session.edited
    assert controller.session.settings.primary.intensity == 99
    assert controller.session.undo() and controller.session.settings.primary.intensity != 99
    assert controller.detail_region == ((0, 0, 120, 80), 1)
    assert view.transform() == transform and not view.fitted
    assert (view.horizontalScrollBar().value(), view.verticalScrollBar().value()) == scroll
    assert view.item.pixmap().toImage() == overview_pixels
    assert window.filename.text() == "old.tif"
    assert window.filename.toolTip() == "old source"
    assert window.image_info.text() == "old image info"
    assert window.image_info.toolTip() == "old color info"
    assert not controller.progressive_open


@pytest.mark.parametrize("mode", ["original", "mask", "effects"])
def test_zoomed_mode_change_updates_overview_and_detail(document, mode):
    window, controller, frame = document
    controller.session.sky_mask = SkyMask(1, np.ones((120, 240), np.uint8), (120, 240))
    controller.detail_region = ((120, 20, 200, 100), 1)
    region_frame = ImageFrame(frame.pixels[20:100, 120:200])
    controller.preview = Preview(
        region_frame, np.full_like(region_frame.pixels, 0.4), 1, origin=(120, 20), is_region=True
    )
    controller.exclusive = True
    window.image_view.fitted = False
    window._view(mode)
    base = window.image_view.item.pixmap().toImage()
    detail = window.image_view.detail_item.pixmap().toImage()
    assert base.pixelColor(130, 30) == detail.pixelColor(10, 10)


def test_compare_divider_uses_full_image_coordinates(document):
    window, controller, frame = document
    region_frame = ImageFrame(frame.pixels[20:100, 140:220])
    controller.preview = Preview(
        region_frame, np.full_like(region_frame.pixels, 0.4), 1, origin=(140, 20), is_region=True
    )
    controller.detail_region = ((140, 20, 220, 100), 1)
    window.image_view.fitted = False
    controller.exclusive = True
    window._view("compare")
    detail = window.image_view.detail_item.pixmap().toImage()
    assert detail.pixelColor(0, 0) == detail.pixelColor(60, 0)


@pytest.mark.parametrize("termination", ["cancel", "failure"])
def test_analysis_termination_restores_committed_count(document, termination):
    window, controller, _ = document
    controller.update_star_count(controller.session.stars, controller.session.settings)
    expected = window.star_count.text()
    window.star_count.setText("Detecting stars…")
    if termination == "cancel":
        controller.cancel()
    else:
        controller._failed("analysis failed")
    assert window.star_count.text() == expected


def test_star_dialog_is_deleted_after_cancel(document, monkeypatch):
    window, controller, _ = document
    controller.session.add_star(10, 20)
    monkeypatch.setattr(QDialog, "exec", lambda _: QDialog.DialogCode.Rejected)
    controller.edit_dialog(controller.session.stars[0])
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert window.findChildren(QDialog) == []
