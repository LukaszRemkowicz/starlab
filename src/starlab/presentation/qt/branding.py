"""Packaged application artwork and version metadata for the desktop interface."""

from importlib.metadata import version
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

APPLICATION_NAME: str = "Star Effects Lab"
APPLICATION_ID: str = "StarEffectsLab.Desktop"


def application_version() -> str:
    """Read the installed distribution version generated from pyproject.toml."""
    return version("star-effects-lab")


def application_icon() -> QIcon:
    """Load native icon resolutions and a portable full-resolution image."""
    assets: Path = Path(__file__).with_name("assets")
    artwork: QIcon = QIcon(str(assets / "application.ico"))
    artwork.addFile(str(assets / "application.png"))
    return artwork


def configure_application(application: QApplication) -> None:
    """Set Qt identity and shared artwork before any top-level windows are built."""
    application.setApplicationName(APPLICATION_NAME)
    application.setOrganizationName(APPLICATION_NAME)
    application.setApplicationVersion(application_version())
    application.setDesktopFileName("star-effects-lab")
    application.setWindowIcon(application_icon())
