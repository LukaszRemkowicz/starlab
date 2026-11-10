"""Detector record layouts, cache keys, and asynchronous tile contracts."""

from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING, NamedTuple, NotRequired, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..array_types import RGB, BoolArray, Bounds, CandidateRecord, DoubleArray, FloatArray

if TYPE_CHECKING:
    from .candidate import Image

type CacheKey = tuple[int, int, int, int]
type CandidateRow = tuple[float, float, float, float, float, float, float, float]
type AttemptRow = tuple[float, float, float, float, float, float, float, float, float, float]
type IndexArray = NDArray[np.int32]
type BitsArray = NDArray[np.uint32]
type CodesArray = NDArray[np.uint16]
type Seed = tuple[float, float, float, float]
type TileThreshold = tuple[int, int, float, float]
type TileKey = tuple[int, int]
type PreparedTile = tuple[Image | None, list[tuple[int, int]], DoubleArray]
type TileResult = tuple[list[CandidateRecord], int, int, DoubleArray]
type ReferenceAttempt = tuple[int, int, CandidateRecord | None]


class CompiledStats(NamedTuple):
    """Five scalar statistics in the native minimum/mean/maximum/stddev/score order."""

    minimum: float
    mean: float
    maximum: float
    stddev: float
    score: float


class Buffer(NamedTuple):
    """Numba image tile: float32 RGB/luma, optional empty mask, origin, typed caches."""

    rgb: FloatArray
    lum: FloatArray
    allowed: BoolArray
    ox: int
    oy: int
    stats: MutableMapping[CacheKey, CompiledStats]
    colors: MutableMapping[CacheKey, RGB]


class PointRecord(TypedDict):
    """Minimal source geometry used by spatial overlap suppression."""

    x: float
    y: float
    size: float


class MultiscaleRecord(PointRecord):
    """Local-background measurement in native half-pixel catalog coordinates."""

    rgb: tuple[float, ...]
    brightness: float
    flux: float
    peak: float
    snr: float
    fwhm: float
    ellipticity: float
    saturated: bool
    scale: float
    normalized_rgb: NotRequired[tuple[float, ...]]
    normalized_brightness: NotRequired[float]


class MeasurementMask(Protocol):
    """Minimal mask interface accepted by the optional multiscale detector."""

    def region(self, bounds: Bounds) -> BoolArray:
        """Return allowed pixels for the given full-resolution half-open bounds."""
        ...

    def allowed_at(self, x: float, y: float) -> bool:
        """Report whether a full-resolution pixel-index position is permitted."""
        ...


class SortableAttempts(Protocol):
    """Structural operations supplied by a local Numba list of accepted seed attempts."""

    def append(self, row: AttemptRow) -> None:
        """Append a seed-coordinate prefix followed by its eight-field source row."""
        ...

    def sort(self) -> None:
        """Sort attempts lexicographically by their native seed coordinates."""
        ...

    def __iter__(self) -> Iterator[AttemptRow]:
        """Visit attempts in their current merge order."""
        ...
