from pathlib import Path
from threading import Event
from typing import Protocol

from ..domain.imaging import ImageFrame, Pixels


class ImageReader(Protocol):
    def __call__(self, path: Path, cancel: Event | None = None) -> ImageFrame: ...


class ImageWriter(Protocol):
    def __call__(
        self,
        path: Path,
        frame: ImageFrame,
        pixels: Pixels,
        source_path: Path | None = None,
        cancel: Event | None = None,
    ) -> None: ...
