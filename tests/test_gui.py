from dataclasses import replace

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt


def _write_star_image(tmp_path):
    y, x = np.mgrid[:128, :160]
    gray = 0.02 + 0.6 * np.exp(-((x - 70.3) ** 2 + (y - 53.7) ** 2) / 7)
    data = np.repeat(np.rint(gray[..., None] * 65535).astype(np.uint16), 3, axis=2)
    source = tmp_path / "stars.tif"
    tifffile.imwrite(source, data, photometric="rgb")
    return source, data


def test_window_load_detect_render_export_and_undo(qtbot, tmp_path, monkeypatch):
    """Check GUI orchestration with deterministic analysis independent of cold detector JIT."""
    from starlab.bootstrap import create_window
    from starlab.domain.stars import Star

    source, data = _write_star_image(tmp_path)
    target = tmp_path / "result.tif"
    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    catalog = (Star(1, 70.3, 53.7, 0.6, (0.6, 0.6, 0.6)),)
    monkeypatch.setattr(window.controller.services, "analyze", lambda *args, **kwargs: catalog)
    window.show()
    window.controller.open_path(source)
    qtbot.waitUntil(
        lambda: window.controller.session.frame is not None and not window.controller.runner.busy,
        timeout=10000,
    )
    # Opening a TIFF must detect stars automatically, without a second hidden step.
    assert window.controller.session.stars is catalog
    before = window.controller.preview.effects.copy()
    window.effect_panel.controls["primary"]["intensity"].spin.setValue(150)
    qtbot.waitUntil(
        lambda: not window.controller.debounce.isActive() and not window.controller.runner.busy,
        timeout=10000,
    )
    assert np.any(window.controller.preview.effects > before)
    window.controller.session.add_star(20, 30)
    window.controller.refresh()
    qtbot.waitUntil(lambda: not window.controller.runner.busy, timeout=10000)
    assert len(window.controller.session.stars) == 2
    window.controller.undo()
    assert len(window.controller.session.stars) == 1
    window.controller.export_path(target, False)
    qtbot.waitUntil(lambda: target.exists() and not window.controller.runner.busy, timeout=10000)
    result = tifffile.imread(target)
    assert result.shape == data.shape
    assert np.any(result > data)
    assert window.controller.shutdown()


@pytest.mark.slow
def test_numerical_import_detection_and_preview_with_cold_jit(qtbot, tmp_path):
    """Allow 60s for cold JIT correctness; this budget does not assert interactive latency."""
    from starlab.bootstrap import create_window

    source, _ = _write_star_image(tmp_path)
    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.show()
    controller = window.controller
    try:
        controller.open_path(source)
        qtbot.waitUntil(lambda: not controller.runner.busy, timeout=60000)
        assert not window.error_label.text()
        assert controller.session.frame is not None and controller.session.stars
        assert abs(controller.session.stars[0].x - 70.3) < 0.25
        assert abs(controller.session.stars[0].y - 53.7) < 0.25
        assert controller.preview is not None and controller.preview.effects.any()
    finally:
        assert controller.shutdown()


def test_effect_panel_preserves_settings_across_modes_and_tabs(qtbot):
    from starlab.domain.effects.settings import EffectSettings
    from starlab.presentation.qt.effect_panel import EffectPanel

    panel = EffectPanel()
    qtbot.addWidget(panel)
    settings = EffectSettings()
    settings = replace(settings, ring=replace(settings.ring, size=73, enabled=True))
    panel.set_settings(settings)
    panel.tabs.setCurrentIndex(3)
    assert panel.controls["ring"]["size"].spin.value() == 73
    panel.basic.setChecked(True)
    panel.advanced.setChecked(True)
    assert panel.settings == settings
    with qtbot.waitSignal(panel.settings_changed) as signal:
        panel.controls["ring"]["size"].spin.setValue(61)
    assert signal.args[0].ring.size == 61
    assert signal.args[0].primary == settings.primary


def test_worker_replaces_stale_job_and_shuts_down(qtbot):
    from starlab.domain.imaging import check_cancel
    from starlab.presentation.qt.worker import JobRunner

    runner = JobRunner()
    results = []
    runner.completed.connect(lambda kind, value: results.append(value))

    def slow(cancel):
        cancel.wait(1)
        check_cancel(cancel)
        return "old"

    runner.submit("preview", slow)
    runner.submit("preview", lambda cancel: "latest")
    qtbot.waitUntil(lambda: not runner.busy, timeout=5000)
    assert results == ["latest"]
    assert runner.shutdown()


def test_empty_window_disables_export_and_has_accessible_controls(qtbot):
    from starlab.bootstrap import create_window

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.show()
    assert not window.export_button.isEnabled()
    assert not window.analyze_button.isEnabled()
    assert window.open_button.isEnabled()
    assert window.effect_panel.tabs.count() == 4
    qtbot.keyClick(window, Qt.Key.Key_Escape)
    assert window.controller.shutdown()


def test_queued_export_cannot_be_replaced_by_preview(qtbot, tmp_path):
    from starlab.bootstrap import create_window
    from starlab.domain.imaging import ImageFrame, check_cancel

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    controller = window.controller
    controller.session.open(ImageFrame(np.zeros((32, 32, 3), np.float32)))

    def slow(cancel):
        cancel.wait(0.2)
        check_cancel(cancel)

    controller.runner.submit("preview", slow)
    target = tmp_path / "queued.tif"
    controller.export_path(target, False)
    controller.refresh()  # A pending debounce must not replace the export.
    qtbot.waitUntil(lambda: not controller.runner.busy, timeout=5000)
    assert target.exists()
    assert controller.shutdown()


def test_actual_size_requests_full_resolution_preview(qtbot):
    from starlab.bootstrap import create_window
    from starlab.domain.imaging import ImageFrame

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.controller.session.open(ImageFrame(np.zeros((64, 1700, 3), np.float32)))
    window.controller.refresh()
    qtbot.waitUntil(lambda: not window.controller.runner.busy)
    assert window.controller.preview.scale == 2
    window.image_view.actual_size()
    qtbot.waitUntil(lambda: not window.controller.runner.busy)
    assert window.controller.preview.scale == 1
    assert window.controller.shutdown()


def test_open_dialog_remembers_folder_across_windows_and_cancel(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from starlab.presentation.qt.main_window import MainWindow, QFileDialog

    preferences_file = str(tmp_path / "preferences.ini")
    folder = tmp_path / "photos"
    folder.mkdir()
    selected = str(folder / "sky.tif")
    answers = iter([selected, "", ""])
    initial_directories = []

    def choose(parent, title, directory, file_filter):
        initial_directories.append(directory)
        return next(answers), ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose)
    first = MainWindow(preferences=QSettings(preferences_file, QSettings.Format.IniFormat))
    qtbot.addWidget(first)
    with qtbot.waitSignal(first.open_requested) as signal:
        first.choose_open()
    assert signal.args == [selected]
    first.close()
    second = MainWindow(preferences=QSettings(preferences_file, QSettings.Format.IniFormat))
    qtbot.addWidget(second)
    second.choose_open()  # Cancel must retain the previous location.
    second.choose_open()
    assert initial_directories[1:] == [str(folder), str(folder)]


def test_open_dialog_falls_back_when_remembered_folder_is_missing(qtbot, tmp_path, monkeypatch):
    from pathlib import Path

    from PySide6.QtCore import QSettings

    from starlab.presentation.qt.main_window import MainWindow, QFileDialog

    preferences = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    preferences.setValue("dialogs/open_image_directory", str(tmp_path / "removed"))
    initial_directories = []

    def choose(parent, title, directory, file_filter):
        initial_directories.append(directory)
        return "", ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose)
    window = MainWindow(preferences=preferences)
    qtbot.addWidget(window)
    window.choose_open()
    assert Path(initial_directories[0]).is_dir()


def test_reference_layout_retains_every_effect_control(qtbot):
    from starlab.bootstrap import create_window
    from starlab.domain.effects.settings import PARAMETERS

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    panel = window.effect_panel
    assert [panel.tabs.tabText(i) for i in range(4)] == [
        "General",
        "Spikes",
        "Soft Flare",
        "Ring Flare",
    ]
    for kind, keys in PARAMETERS.items():
        assert set(panel.controls[kind]) == set(keys)
    panel.spike_selector.setCurrentIndex(1)
    panel.controls["secondary"]["rainbow"].spin.setValue(91)
    panel.spike_selector.setCurrentIndex(0)
    assert panel.settings.secondary.rainbow == 91
    assert window.canvas.acceptDrops()
    assert window.controller.shutdown()


def test_hold_before_restores_view_and_preset_updates_session(qtbot):
    from starlab.bootstrap import create_window
    from starlab.domain.imaging import ImageFrame

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.controller.session.open(ImageFrame(np.zeros((32, 32, 3), np.float32)))
    window.set_loaded(True)
    window.show()
    window._view("effects")
    qtbot.mousePress(window.before_button, Qt.MouseButton.LeftButton)
    assert window.mode == "original"
    qtbot.mouseRelease(window.before_button, Qt.MouseButton.LeftButton)
    assert window.mode == "effects"
    window.preset_combo.setCurrentIndex(1)
    qtbot.waitUntil(
        lambda: not window.controller.debounce.isActive() and not window.controller.runner.busy
    )
    assert window.controller.session.settings.primary.intensity < 45
    window.apply_button.click()
    qtbot.waitUntil(lambda: not window.controller.runner.busy)
    assert window.controller.preview is not None
    assert window.controller.shutdown()


def test_drop_on_empty_canvas_opens_file(qtbot, tmp_path):
    from pathlib import Path

    from PySide6.QtCore import QMimeData, QPointF, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import QApplication

    from starlab.presentation.qt.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    path = tmp_path / "drop.tif"
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    enter = QDragEnterEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window.canvas, enter)
    assert enter.isAccepted()
    drop = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    with qtbot.waitSignal(window.open_requested) as signal:
        QApplication.sendEvent(window.canvas, drop)
    assert Path(signal.args[0]) == path


def test_activity_timer_and_exclusive_controls(qtbot):
    from starlab.bootstrap import create_window
    from starlab.domain.imaging import ImageFrame

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.controller.session.open(ImageFrame(np.zeros((32, 32, 3), np.float32)))
    window.set_loaded(True)
    window.controller.activity("analyze")
    assert "Detecting stars" in window.status_text.text()
    assert window.activity_timer.isActive()
    assert not window.apply_button.isEnabled()
    assert not window.preset_combo.isEnabled()
    window.controller.idle()
    assert not window.activity_timer.isActive()
    assert window.apply_button.isEnabled()
    assert window.controller.shutdown()


def test_undo_preset_allows_reselecting_same_preset(qtbot):
    from starlab.bootstrap import create_window

    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.preset_combo.setCurrentIndex(1)
    assert window.controller.session.settings.primary.intensity == 12
    window.controller.undo()
    assert window.controller.session.settings.primary.intensity == 45
    assert window.preset_combo.currentText() == "Custom"
    assert window.effect_panel.controls["primary"]["intensity"].spin.value() == 45
    window.preset_combo.setCurrentIndex(1)
    assert window.controller.session.settings.primary.intensity == 12
    window.controller.undo()
    window.controller.redo()
    assert window.preset_combo.currentText() == "Custom"
    assert window.controller.session.settings.primary.intensity == 12
    assert window.controller.shutdown()


def test_footer_shows_opened_path_elided_with_full_tooltip(qtbot, tmp_path):
    from starlab.bootstrap import create_window

    source, _ = _write_star_image(tmp_path)
    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    window.show()
    assert window.path_label.full_text() == ""
    window.controller.open_path(source)
    qtbot.waitUntil(
        lambda: window.controller.session.frame is not None and not window.controller.runner.busy,
        timeout=10000,
    )
    label = window.path_label
    assert label.full_text() == label.toolTip() == str(source)
    label.setFixedWidth(60)
    qtbot.waitUntil(lambda: "…" in label.text())
    assert label.text().startswith(str(source)[:2]) and label.text().endswith(source.name[-2:])
    window.set_loaded(False)
    assert label.full_text() == "" and label.text() == ""
