"""Shared numerical shapes and units at Python/JIT/application boundaries."""

from collections.abc import Sequence
from typing import Literal, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

type Bounds = tuple[int, int, int, int]
type Position = tuple[float, float]
type RGB = tuple[float, float, float]
type RGBA = tuple[float, float, float, float]
type BoolArray = NDArray[np.bool_]
type FloatArray = NDArray[np.float32]
type DoubleArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type ByteArray = NDArray[np.uint8]
type CoordinateKey = Literal["x", "y", "size", "brightness"]
type Diagnostics = dict[str, object]


class CandidateRecord(TypedDict):
    """Raw candidate in half-pixel coordinates, with optional global normalization."""

    x: float
    y: float
    size: float
    brightness: float
    rgb: Sequence[float]
    count: int
    normalized_rgb: NotRequired[Sequence[float]]
    normalized_brightness: NotRequired[float]
