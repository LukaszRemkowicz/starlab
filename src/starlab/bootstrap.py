"""Construct the application dependency graph without leaking Qt into domain code."""

import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING, Literal

from .application.use_cases import ImageServices
from .domain.imaging import check_cancel
from .infrastructure.photoshop_bridge import LAYER_NAME, PhotoshopBridge
from .infrastructure.preset_json import load_preset, save_preset
from .infrastructure.tiff_io import read_tiff, read_tiff_preview, write_tiff

if TYPE_CHECKING:
    from .presentation.qt.main_window import MainWindow

# Runs in the background process every launch: compiles kernels, or loads them if already compiled.
PREPARE = "import sys; from starlab.preparation import main; main(sys.argv[1:])"
PREPARATION_WAIT_LIMIT = 120.0


def create_services(*, sky_mask: bool, bridge: PhotoshopBridge | None = None) -> ImageServices:
    """Build the image services shared by the window and background preparation.

    With a Photoshop bridge, pixels come from and go to its shared memory; there is no
    encoded source file, so the sky mask is built from the decoded frame instead.
    """
    services: ImageServices = (
        ImageServices(read_tiff, write_tiff, read_preview=read_tiff_preview)
        if bridge is None
        else ImageServices(bridge.read, bridge.write, read_preview=bridge.read_preview)
    )
    if sky_mask:
        from .infrastructure.sky_mask import SkyMaskService

        services.sky_mask = SkyMaskService(persistent_cache=False)
        if bridge is None:
            services.read_mask = services.sky_mask.from_source
    return services


def repeat_last(services: ImageServices, bridge: PhotoshopBridge) -> None:
    """Re-apply the plug-in's stored settings to the layer without a window ("Last Filter").

    Mirrors the GUI path: decode the frame, build the sky mask from it, detect, then
    export the composite into the block (status 1 only after the last row).
    """
    detection, effects = bridge.settings()
    path: Path = Path(LAYER_NAME)
    frame, _ = services.open_image(path)
    mask = services.sky_mask(frame, None, None) if services.sky_mask is not None else None
    stars = services.analyze(frame, None, detection, None, sky_mask=mask)
    services.export(path, frame, stars, effects, False, None, None, sky_mask=mask)


def create_window(
    *, backend: Literal["auto", "cpu", "gpu"] = "auto", bridge: PhotoshopBridge | None = None
) -> MainWindow:
    """Build production services; explicit cpu is a bounded test backend without the model.

    Auto retains native GPU previews and mandatory sky masking on desktop platforms.
    Explicit gpu also attempts a real context under a caller-selected Qt platform.
    """
    from PySide6.QtGui import QGuiApplication

    from .presentation.qt.controller import Controller
    from .presentation.qt.main_window import MainWindow

    window: MainWindow = MainWindow()

    gpu: bool = backend == "gpu" or (
        backend == "auto" and QGuiApplication.platformName() != "offscreen"
    )
    services: ImageServices = create_services(sky_mask=gpu, bridge=bridge)
    if gpu:
        from .infrastructure.gpu_preview import GpuPreviewRenderer

        services.preview = GpuPreviewRenderer()
    controller: Controller = Controller(
        window,
        services,
        load_preset,
        save_preset,
    )
    window.controller = controller
    if bridge is not None:
        window.enter_bridge_mode()
        controller.store_settings = bridge.store_settings
    if services.preview is not None:
        controller.runner.cleanup = services.preview.close
    return window


def start_preparation(
    log_path: Path, stamp_path: Path | None = None
) -> tuple[subprocess.Popen[bytes], Callable[[Event | None], None]]:
    """Compile numerical code in a separate process so the window stays responsive.

    Returns the process and a wait function for the image services. Work waits at
    most PREPARATION_WAIT_LIMIT seconds; after that, or if the process fails, it
    continues and compiles anything still missing itself. Failures are logged; a
    log file that cannot be created raises OSError, so the app does not start.
    The stamp file records which program build has already been prepared.
    """
    logger: logging.Logger = logging.getLogger("starlab")
    arguments: list[str] = [str(stamp_path)] if stamp_path is not None else []
    with open(log_path, "wb") as errors:  # the child keeps its own handle
        process: subprocess.Popen[bytes] = subprocess.Popen(
            [sys.executable, "-P", "-c", PREPARE, *arguments],
            stdin=subprocess.PIPE,
            env={**os.environ, "STARLAB_PARENT_PID": str(os.getpid())},
            stdout=subprocess.PIPE,
            stderr=errors,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    started: float = time.monotonic()
    reported: list[bool] = [False]
    available: Event = Event()

    def read_ready() -> None:
        """The child signals after native detection, or exits on a warm cache or failure."""
        assert process.stdout is not None
        with process.stdout as pipe:
            for line in pipe:
                if line.strip() == b"ready":
                    available.set()
        process.wait()
        available.set()

    Thread(target=read_ready, name="detector-readiness", daemon=True).start()

    def ready(cancel: Event | None) -> None:
        """Wait for the native detector cache while polling cooperative cancellation."""
        while not available.wait(0.05) and time.monotonic() - started < PREPARATION_WAIT_LIMIT:
            check_cancel(cancel)
        check_cancel(cancel)
        if reported[0]:
            return
        if not available.is_set():
            reported[0] = True
            logger.warning(
                "Background preparation still running after %.0fs", PREPARATION_WAIT_LIMIT
            )
        elif process.poll() not in (None, 0):
            reported[0] = True
            logger.warning(
                "Background preparation failed with exit code %d; see %s",
                process.returncode,
                log_path,
            )

    return process, ready
