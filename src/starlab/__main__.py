"""Parse the optional input path and run the standalone Qt application."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from subprocess import Popen
from threading import Event
from typing import TYPE_CHECKING

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication, QMessageBox

from .infrastructure.desktop_identity import set_process_identity
from .infrastructure.diagnostics import DiagnosticsUnavailable, application_diagnostics
from .kernel_bundle import install
from .presentation.qt.branding import APPLICATION_ID, configure_application

if TYPE_CHECKING:
    from .infrastructure.photoshop_bridge import PhotoshopBridge

# Modules that use Numba are imported only after install(): its decorators pick the cache at import.


def _prepare(
    app: QApplication, packaged: bool, data: Path, logger: logging.Logger
) -> tuple[Popen[bytes], Callable[[Event | None], None]] | None:
    """Compile image kernels in the background unless the packaged ones were installed."""
    if packaged:
        return None
    from .bootstrap import start_preparation

    try:
        preparation, ready = start_preparation(
            data / "logs" / "preparation.log", data / "prepared-code"
        )
    except OSError as exc:
        raise DiagnosticsUnavailable(f"Cannot create {data / 'logs'}: {exc}") from exc
    logger.info("Preparing local image kernels: PID %d", preparation.pid)
    app.aboutToQuit.connect(preparation.terminate)
    return preparation, ready


def _run(
    app: QApplication,
    image: str | None,
    data: Path,
    logger: logging.Logger,
    packaged: bool,
    bridge: PhotoshopBridge | None = None,
) -> int:
    """Show the window, start background preparation and run the Qt event loop."""
    logger.info("Application version: %s", app.applicationVersion())
    app.lastWindowClosed.connect(lambda: logger.info("Qt lastWindowClosed"))
    app.aboutToQuit.connect(lambda: logger.info("Qt aboutToQuit"))
    from .bootstrap import create_window

    window = create_window(bridge=bridge)
    prepared = _prepare(app, packaged, data, logger)
    if prepared is not None and window.controller is not None:
        window.controller.services.ready = prepared[1]
    if window.controller is not None:
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
    app: QApplication, bridge: PhotoshopBridge, data: Path, logger: logging.Logger, packaged: bool
) -> int:
    """Re-apply the stored settings headlessly; any failure is shown and leaves status 0."""
    from .bootstrap import create_services, repeat_last

    prepared = _prepare(app, packaged, data, logger)
    try:
        # Same sky-mask rule as create_window: real platforms only.
        services = create_services(sky_mask=app.platformName() != "offscreen", bridge=bridge)
        if prepared is not None:
            services.ready = prepared[1]
        logger.info("Repeating last settings without a window")
        repeat_last(services, bridge)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Repeat failed")
        QMessageBox.critical(None, "Star Effects Lab could not repeat the last filter", str(exc))
    if prepared is not None:
        prepared[0].terminate()
    result: int = bridge.exit_code
    logger.info("Photoshop bridge exit code: %d", result)
    return result


def _run_bridge(
    app: QApplication, name: str, data: Path, logger: logging.Logger, packaged: bool
) -> int:
    """Attach to the plug-in's block (reporting problems clearly) and edit its layer."""
    from .infrastructure.photoshop_bridge import EXIT_CANCELLED, MODE_REPEAT, PhotoshopBridge

    try:
        bridge: PhotoshopBridge = PhotoshopBridge(name)
    except (OSError, ValueError) as exc:
        logger.error("Photoshop bridge %r unusable: %s", name, exc)
        QMessageBox.critical(None, "Cannot open the Photoshop layer", str(exc))
        return EXIT_CANCELLED
    try:
        if bridge.mode == MODE_REPEAT:
            return _run_repeat(app, bridge, data, logger, packaged)
        return _run(app, None, data, logger, packaged, bridge)
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
            packaged = install(data)
            if args.photoshop_bridge is None:
                return _run(app, args.image, data, logger, packaged)
            return _run_bridge(app, args.photoshop_bridge, data, logger, packaged)
    except DiagnosticsUnavailable as exc:
        QMessageBox.critical(
            None,
            "Star Effects Lab cannot start",
            f"{exc}\n\nCheck that this folder exists and that you can write to it.",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
