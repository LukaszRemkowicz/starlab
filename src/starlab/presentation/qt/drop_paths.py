"""File selection shared by the empty canvas and image preview drop targets."""

from pathlib import Path

from PySide6.QtCore import QMimeData


def local_tiff_path(mime: QMimeData) -> str | None:
    """Return the first local TIFF URL, skipping remote URLs and other suffixes."""
    for url in mime.urls():
        if url.isLocalFile():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in (".tif", ".tiff"):
                return path
    return None
