from .application.use_cases import ImageServices
from .infrastructure.preset_json import load_preset, save_preset
from .infrastructure.tiff_io import read_tiff, read_tiff_preview, write_tiff
from .presentation.qt.controller import Controller
from .presentation.qt.main_window import MainWindow


def create_window() -> MainWindow:
    window = MainWindow()
    services = ImageServices(read_tiff, write_tiff, read_preview=read_tiff_preview)
    from PySide6.QtGui import QGuiApplication

    if QGuiApplication.platformName() != "offscreen":
        from .infrastructure.gpu_preview import GpuPreviewRenderer
        from .infrastructure.sky_mask import SkyMaskService

        services.preview = GpuPreviewRenderer()
        services.sky_mask = SkyMaskService(persistent_cache=False)
        services.read_mask = services.sky_mask.from_source
    window.controller = Controller(
        window,
        services,
        load_preset,
        save_preset,
    )
    if services.preview is not None:
        window.controller.runner.cleanup = services.preview.close
    return window
