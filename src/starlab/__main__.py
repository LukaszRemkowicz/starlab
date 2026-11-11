"""Parse the optional input path and run the standalone Qt application."""

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication, QMessageBox

from .bootstrap import create_window, start_preparation
from .infrastructure.desktop_identity import set_process_identity
from .infrastructure.diagnostics import DiagnosticsUnavailable, application_diagnostics
from .presentation.qt.branding import APPLICATION_ID, configure_application
from .presentation.qt.main_window import MainWindow


def _run(app: QApplication, image: str | None, data: Path, logger: logging.Logger) -> int:
    """Show the window, start background preparation and run the Qt event loop."""
    logger.info("Application version: %s", app.applicationVersion())
    app.lastWindowClosed.connect(lambda: logger.info("Qt lastWindowClosed"))
    app.aboutToQuit.connect(lambda: logger.info("Qt aboutToQuit"))
    window: MainWindow = create_window()
    try:
        preparation, ready = start_preparation(
            data / "logs" / "preparation.log", data / "prepared-code"
        )
    except OSError as exc:
        raise DiagnosticsUnavailable(f"Cannot create {data / 'logs'}: {exc}") from exc
    logger.info("Compiling image code in background: PID %d", preparation.pid)
    app.aboutToQuit.connect(preparation.terminate)
    if window.controller is not None:
        window.controller.services.ready = ready
        window.controller.runner.activity.connect(lambda kind: logger.info("Job started: %s", kind))
        window.controller.runner.failed.connect(
            lambda message: logger.error("Job failed: %s", message)
        )
        window.controller.runner.idle.connect(lambda: logger.info("Worker idle"))
    window.show()
    logger.info("Main window shown")
    if image and window.controller is not None:
        window.controller.open_path(image)
    result: int = app.exec()
    logger.info("Qt event loop returned: %d", result)
    return result


def main() -> int:
    """Create the GUI, optionally open an image, and return the Qt event-loop exit code.

    Logs are mandatory: if they cannot be written, the user is told why and the app
    does not open.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Star Effects Lab — standalone TIFF diffraction effects"
    )
    parser.add_argument("image", nargs="?", help="Optional TIFF image to open")
    args: argparse.Namespace = parser.parse_args()
    set_process_identity(APPLICATION_ID)
    app: QApplication = QApplication(sys.argv[:1])
    configure_application(app)
    data: Path = Path(
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    )
    try:
        with application_diagnostics(data / "logs") as logger:
            return _run(app, args.image, data, logger)
    except DiagnosticsUnavailable as exc:
        QMessageBox.critical(
            None,
            "Star Effects Lab cannot start",
            f"{exc}\n\nCheck that this folder exists and that you can write to it.",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
