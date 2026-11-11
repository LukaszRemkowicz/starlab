from starlab.bootstrap import create_window


def test_explicit_cpu_test_backend_omits_native_services(qtbot, monkeypatch):
    from PySide6.QtGui import QGuiApplication

    monkeypatch.setattr(QGuiApplication, "platformName", lambda: "windows")
    window = create_window(backend="cpu")
    qtbot.addWidget(window)
    assert window.controller is not None
    assert window.controller.services.preview is None
    assert window.controller.services.sky_mask is None
    assert window.controller.services.read_mask is None
    assert window.controller.shutdown()
