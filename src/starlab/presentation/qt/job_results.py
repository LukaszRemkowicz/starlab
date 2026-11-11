"""Tuple-compatible payloads crossing the worker's Qt object-signal boundary."""

from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import NamedTuple

from ...application.use_cases import Preview
from ...domain.imaging import ImageFrame
from ...domain.sky_mask import SkyMask
from ...domain.stars import DetectionSettings, Star

type Thumbnail = tuple[ImageFrame, float, int, int]


class OpenResult(NamedTuple):
    """Fully decoded document and its initial effect-free overview."""

    path: Path
    frame: ImageFrame
    preview: Preview
    mask: SkyMask | None


class OpenPreviewResult(NamedTuple):
    """Temporary thumbnail and the deferred full document read."""

    path: Path
    thumbnail: Thumbnail
    read: Callable[[Event], OpenResult]


class MaskResult(NamedTuple):
    """Mask preparation tied to the document revision that requested it."""

    revision: int
    mask: SkyMask
    settings: DetectionSettings
    force: bool
    detect: bool


class AnalysisResult(NamedTuple):
    """Catalog and the exact mask and detection settings that produced it."""

    revision: int
    stars: tuple[Star, ...]
    mask: SkyMask | None
    settings: DetectionSettings


class PreviewResult(NamedTuple):
    """Rendered preview tied to the session revision used by its worker."""

    revision: int
    preview: Preview
