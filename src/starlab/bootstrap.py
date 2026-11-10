"""Construct the application dependency graph without leaking Qt into domain code."""

from typing import Literal

from .application.use_cases import ImageServices
from .infrastructure.preset_json import load_preset, save_preset
from .infrastructure.tiff_io import read_tiff, read_tiff_preview, write_tiff
from .presentation.qt.controller import Controller
from .presentation.qt.main_window import MainWindow


def create_window(*, backend: Literal["auto", "cpu", "gpu"] = "auto") -> MainWindow:
    """Build production services; explicit cpu is a bounded test backend without the model.

    Auto retains native GPU previews and mandatory sky masking on desktop platforms.
    Explicit gpu also attempts a real context under a caller-selected Qt platform.
    """
    window: MainWindow = MainWindow()
    services: ImageServices = ImageServices(read_tiff, write_tiff, read_preview=read_tiff_preview)
    from PySide6.QtGui import QGuiApplication

    if backend == "gpu" or (backend == "auto" and QGuiApplication.platformName() != "offscreen"):
        from .infrastructure.gpu_preview import GpuPreviewRenderer
        from .infrastructure.sky_mask import SkyMaskService

        services.preview = GpuPreviewRenderer()
        services.sky_mask = SkyMaskService(persistent_cache=False)
        services.read_mask = services.sky_mask.from_source
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
