"""Parse the optional input path and run the standalone Qt application."""

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication

from .bootstrap import create_window, start_preparation
from .infrastructure.desktop_identity import set_process_identity
from .infrastructure.diagnostics import application_diagnostics
from .presentation.qt.branding import APPLICATION_ID, configure_application
from .presentation.qt.main_window import MainWindow


def main() -> int:
    """Create the GUI, optionally open an image, and return the Qt event-loop exit code."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Star Effects Lab — standalone TIFF diffraction effects"
    )
    parser.add_argument("image", nargs="?", help="Optional TIFF image to open")
    args: argparse.Namespace = parser.parse_args()
    set_process_identity(APPLICATION_ID)
    app: QApplication = QApplication(sys.argv[:1])
    configure_application(app)
    log_directory: Path = (
        Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
        / "logs"
    )
    with application_diagnostics(log_directory) as logger:
        logger.info("Application version: %s", app.applicationVersion())
        app.lastWindowClosed.connect(lambda: logger.info("Qt lastWindowClosed"))
        app.aboutToQuit.connect(lambda: logger.info("Qt aboutToQuit"))
        window: MainWindow = create_window()
        preparation, ready = start_preparation(log_directory / "preparation.log")
        logger.info("Compiling image code in background: PID %d", preparation.pid)
        app.aboutToQuit.connect(preparation.terminate)
        if window.controller is not None:
            window.controller.services.ready = ready
            window.controller.runner.activity.connect(
                lambda kind: logger.info("Job started: %s", kind)
            )
            window.controller.runner.failed.connect(
                lambda message: logger.error("Job failed: %s", message)
            )
            window.controller.runner.idle.connect(lambda: logger.info("Worker idle"))
        window.show()
        logger.info("Main window shown")
        if args.image and window.controller is not None:
            window.controller.open_path(args.image)
        result: int = app.exec()
        logger.info("Qt event loop returned: %d", result)
        return result


if __name__ == "__main__":
    raise SystemExit(main())
