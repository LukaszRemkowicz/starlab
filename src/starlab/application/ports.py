"""Typed I/O and preview boundaries supplied by infrastructure adapters."""

from collections.abc import Iterable
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Protocol

from ..domain.array_types import Bounds
from ..domain.imaging import ImageFrame, Pixels
from ..domain.sky_mask import SkyMask

if TYPE_CHECKING:
    from ..domain.effects.settings import EffectSettings
    from ..domain.stars import Star
    from .use_cases import Preview


class ImageReader(Protocol):
    """Read one complete immutable image in the linear working color space."""

    def __call__(self, path: Path, cancel: Event | None = None) -> ImageFrame:
        """Import source pixels, checking the optional cancellation token."""
        ...


class ImageWriter(Protocol):
    """Write a result while preserving the source-protection and cancellation contract."""

    def __call__(
        self,
        path: Path,
        frame: ImageFrame,
        pixels: Pixels | Iterable[Pixels],
        *,
        source_path: Path | None = None,
        cancel: Event | None = None,
    ) -> None:
        """Encode a full array or full-width row bands (top to bottom), never the source path."""
        ...


class PreviewReader(Protocol):
    """Read a bounded thumbnail and report its scale and full image dimensions."""

    def __call__(
        self, path: Path, cancel: Event | None = None, *, max_edge: int = 1600
    ) -> tuple[ImageFrame, float, int, int]:
        """Return thumbnail, source-to-thumbnail scale, full width and full height."""
        ...


class MaskReader(Protocol):
    """Prepare a source-image mask independently of full linear-pixel import."""

    def __call__(self, source: Path, cancel: Event | None = None) -> SkyMask:
        """Read encoded source samples and return a mask covering the full image."""
        ...


class SkyMaskProvider(Protocol):
    """Construct a mask from an imported image and an optional encoded source."""

    def __call__(
        self, frame: ImageFrame, source: Path | None = None, cancel: Event | None = None
    ) -> SkyMask:
        """Return the model mask with source-coordinate dimensions preserved."""
        ...


class PreviewRenderer(Protocol):
    """Render immutable preview layers and own any backend resources until close."""

    def __call__(
        self,
        frame: ImageFrame,
        stars: tuple["Star", ...],
        settings: "EffectSettings",
        cancel: Event | None = None,
        *,
        full_resolution: bool = False,
        interactive: bool = False,
        region: Bounds | None = None,
        region_sampling: int = 1,
        max_edge: int = 1600,
        sky_mask: SkyMask | None = None,
    ) -> "Preview":
        """Return full-coordinate effect layers for an overview or sampled region."""
        ...

    def close(self) -> None:
        """Release backend resources on the rendering worker's owning thread."""
        ...
