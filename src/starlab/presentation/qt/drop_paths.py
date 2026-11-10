"""File selection shared by the empty canvas and image preview drop targets."""

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QMimeData
from PySide6.QtGui import QDropEvent


def local_tiff_path(mime: QMimeData) -> str | None:
    """Return the first local TIFF URL, skipping remote URLs and other suffixes."""
    for url in mime.urls():
        if url.isLocalFile():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in (".tif", ".tiff"):
                return path
    return None


def accept_tiff_drag(event: QDropEvent) -> None:
    """Accept a drag containing at least one eligible local TIFF URL."""
    if local_tiff_path(event.mimeData()) is not None:
        event.acceptProposedAction()


def drop_tiff(event: QDropEvent, emit: Callable[[str], None]) -> None:
    """Emit the first eligible local TIFF path and accept its drop."""
    path = local_tiff_path(event.mimeData())
    if path is not None:
        emit(path)
        event.acceptProposedAction()
