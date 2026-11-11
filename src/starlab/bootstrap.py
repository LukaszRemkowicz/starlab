"""Construct the application dependency graph without leaking Qt into domain code."""

import subprocess
import sys
import time
from collections.abc import Callable
from threading import Event
from typing import Literal

from .application.use_cases import ImageServices
from .domain.imaging import check_cancel
from .infrastructure.preset_json import load_preset, save_preset
from .infrastructure.tiff_io import read_tiff, read_tiff_preview, write_tiff
from .presentation.qt.controller import Controller
from .presentation.qt.main_window import MainWindow


def create_services(*, sky_mask: bool) -> ImageServices:
    """Build the image services shared by the window and the --prepare command."""
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


def start_preparation() -> tuple[subprocess.Popen[bytes], Callable[[Event | None], None]]:
    """Compile numerical code in a separate process so the window stays responsive.

    Returns the process and a wait function for the image services; when the process
    ends (or fails), work continues and compiles anything still missing itself.
    """
    process: subprocess.Popen[bytes] = subprocess.Popen(
        [sys.executable, "-m", "starlab", "--prepare"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    def ready(cancel: Event | None) -> None:
        """Block a worker job until background compilation finishes or is cancelled."""
        while process.poll() is None:
            check_cancel(cancel)
            time.sleep(0.05)

    return process, ready
