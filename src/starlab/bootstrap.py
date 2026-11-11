"""Construct the application dependency graph without leaking Qt into domain code."""

import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from typing import Literal

from .application.use_cases import ImageServices
from .domain.imaging import check_cancel
from .infrastructure.preset_json import load_preset, save_preset
from .infrastructure.tiff_io import read_tiff, read_tiff_preview, write_tiff
from .presentation.qt.controller import Controller
from .presentation.qt.main_window import MainWindow

# Runs in the background process every launch: compiles kernels, or loads them if already compiled.
PREPARE = "from starlab.bootstrap import prepare_in_child; prepare_in_child()"
PREPARATION_WAIT_LIMIT = 120.0


def create_services(*, sky_mask: bool) -> ImageServices:
    """Build the image services shared by the window and background preparation."""
    services: ImageServices = ImageServices(read_tiff, write_tiff, read_preview=read_tiff_preview)
    if sky_mask:
        from .infrastructure.sky_mask import SkyMaskService

        services.sky_mask = SkyMaskService(persistent_cache=False)
        services.read_mask = services.sky_mask.from_source
    return services


def create_window(*, backend: Literal["auto", "cpu", "gpu"] = "auto") -> MainWindow:
    """Build production services; explicit cpu is a bounded test backend without the model.

    Auto retains native GPU previews and mandatory sky masking on desktop platforms.
    Explicit gpu also attempts a real context under a caller-selected Qt platform.
    """
    window: MainWindow = MainWindow()
    from PySide6.QtGui import QGuiApplication

    gpu: bool = backend == "gpu" or (
        backend == "auto" and QGuiApplication.platformName() != "offscreen"
    )
    services: ImageServices = create_services(sky_mask=gpu)
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
    if services.preview is not None:
        controller.runner.cleanup = services.preview.close
    return window


def prepare_in_child() -> None:
    """Background-process entry: warm up, and exit at once if the app goes away.

    The app holds this process's stdin open; when the app exits or crashes, the
    operating system closes it, the read returns, and the process stops.
    """

    def exit_with_parent() -> None:
        """Stop this process as soon as the parent's end of stdin closes."""
        sys.stdin.read()
        os._exit(0)

    Thread(target=exit_with_parent, daemon=True).start()
    from .application.warmup import warm_up

    warm_up(create_services(sky_mask=True))


def start_preparation(
    log_path: Path | None = None,
) -> tuple[subprocess.Popen[bytes], Callable[[Event | None], None]]:
    """Compile numerical code in a separate process so the window stays responsive.

    Returns the process and a wait function for the image services. Work waits at
    most PREPARATION_WAIT_LIMIT seconds; after that, or if the process fails, it
    continues and compiles anything still missing itself. Failures are logged.
    """
    logger: logging.Logger = logging.getLogger("starlab")
    errors = None
    if log_path is not None:
        try:
            errors = open(log_path, "wb")  # handed to the child process
        except OSError:
            logger.warning("Preparation log %s is not writable", log_path)
    process: subprocess.Popen[bytes] = subprocess.Popen(
        [sys.executable, "-P", "-c", PREPARE],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=errors if errors is not None else subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if errors is not None:
        errors.close()  # the child keeps its own handle
    started: float = time.monotonic()
    reported: list[bool] = [False]

    def ready(cancel: Event | None) -> None:
        """Block a worker job until background compilation finishes, fails or times out."""
        while process.poll() is None and time.monotonic() - started < PREPARATION_WAIT_LIMIT:
            check_cancel(cancel)
            time.sleep(0.05)
        if reported[0]:
            return
        if process.returncode is None:
            reported[0] = True
            logger.warning(
                "Background preparation still running after %.0fs", PREPARATION_WAIT_LIMIT
            )
        elif process.returncode != 0:
            reported[0] = True
            logger.warning(
                "Background preparation failed with exit code %d; see %s",
                process.returncode,
                log_path,
            )

    return process, ready
