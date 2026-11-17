"""Parse the optional input path and run the standalone Qt application."""

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication, QMessageBox

from .application.use_cases import ImageServices
from .bootstrap import create_services, create_window, repeat_last, start_preparation
from .infrastructure.desktop_identity import set_process_identity
from .infrastructure.diagnostics import DiagnosticsUnavailable, application_diagnostics
from .infrastructure.photoshop_bridge import EXIT_CANCELLED, MODE_REPEAT, PhotoshopBridge
from .presentation.qt.branding import APPLICATION_ID, configure_application
from .presentation.qt.main_window import MainWindow


def _run(
    app: QApplication,
    image: str | None,
    data: Path,
    logger: logging.Logger,
    bridge: PhotoshopBridge | None = None,
) -> int:
    """Show the window, start background preparation and run the Qt event loop."""
    logger.info("Application version: %s", app.applicationVersion())
    app.lastWindowClosed.connect(lambda: logger.info("Qt lastWindowClosed"))
    app.aboutToQuit.connect(lambda: logger.info("Qt aboutToQuit"))
    window: MainWindow = create_window(bridge=bridge)
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
    if window.controller is not None:
        if bridge is not None:
            window.controller.open_layer()
        elif image:
            window.controller.open_path(image)
    result: int = app.exec()
    logger.info("Qt event loop returned: %d", result)
    if bridge is not None:
        bridge.cancel()  # no-op after a successful Apply
        result = bridge.exit_code
        logger.info("Photoshop bridge exit code: %d", result)
    return result


def _run_repeat(
    app: QApplication, bridge: PhotoshopBridge, data: Path, logger: logging.Logger
) -> int:
    """Re-apply the stored settings headlessly; any failure is shown and leaves status 0."""
    try:
        preparation, ready = start_preparation(
            data / "logs" / "preparation.log", data / "prepared-code"
        )
    except OSError as exc:
        raise DiagnosticsUnavailable(f"Cannot create {data / 'logs'}: {exc}") from exc
    app.aboutToQuit.connect(preparation.terminate)
    try:
        # Same sky-mask rule as create_window: real platforms only.
        services: ImageServices = create_services(
            sky_mask=app.platformName() != "offscreen", bridge=bridge
        )
        services.ready = ready
        logger.info("Repeating last settings without a window")
        repeat_last(services, bridge)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Repeat failed")
        QMessageBox.critical(None, "Star Effects Lab could not repeat the last filter", str(exc))
    preparation.terminate()
    result: int = bridge.exit_code
    logger.info("Photoshop bridge exit code: %d", result)
    return result


def _run_bridge(app: QApplication, name: str, data: Path, logger: logging.Logger) -> int:
    """Attach to the plug-in's block (reporting problems clearly) and edit its layer."""
    try:
        bridge: PhotoshopBridge = PhotoshopBridge(name)
    except (OSError, ValueError) as exc:
        logger.error("Photoshop bridge %r unusable: %s", name, exc)
        QMessageBox.critical(None, "Cannot open the Photoshop layer", str(exc))
        return EXIT_CANCELLED
    try:
        if bridge.mode == MODE_REPEAT:
            return _run_repeat(app, bridge, data, logger)
        return _run(app, None, data, logger, bridge)
    finally:
        bridge.close()


def main() -> int:
    """Create the GUI, optionally open an image, and return the Qt event-loop exit code.

    Logs are mandatory: if they cannot be written, the user is told why and the app
    does not open.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Star Effects Lab — standalone TIFF diffraction effects"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("image", nargs="?", help="Optional TIFF image to open")
    source.add_argument(
        "--photoshop-bridge", metavar="NAME", help="Edit the layer in this shared-memory block"
    )
    args: argparse.Namespace = parser.parse_args()
    set_process_identity(APPLICATION_ID)
    app: QApplication = QApplication(sys.argv[:1])
    configure_application(app)
    data: Path = Path(
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    )
    try:
        with application_diagnostics(data / "logs") as logger:
            if args.photoshop_bridge is None:
                return _run(app, args.image, data, logger)
            return _run_bridge(app, args.photoshop_bridge, data, logger)
    except DiagnosticsUnavailable as exc:
        QMessageBox.critical(
            None,
            "Star Effects Lab cannot start",
            f"{exc}\n\nCheck that this folder exists and that you can write to it.",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
