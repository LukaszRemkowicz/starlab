"""Owned linear-light image buffers, cancellation, and exact display transfer."""

from dataclasses import dataclass
from functools import lru_cache
from threading import Event
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from .array_types import BoolArray, ByteArray, FloatArray

Pixels = FloatArray


class ImagePixels(Protocol):
    """Read-only pixel and dimension access shared by owned frames and ROI views."""

    @property
    def pixels(self) -> Pixels:
        """Expose H×W×C linear float32 pixels without copying."""
        ...

    @property
    def width(self) -> int:
        """Return the number of source pixel columns."""
        ...

    @property
    def height(self) -> int:
        """Return the number of source pixel rows."""
        ...


class Cancelled(Exception):
    """Cooperative cancellation of an image operation."""


def check_cancel(cancel: Event | None) -> None:
    """Raise at a cooperative boundary when the caller has cancelled the operation."""
    if cancel is not None and cancel.is_set():
        raise Cancelled()


@dataclass(frozen=True, eq=False)
class ImageFrame:
    """Own immutable contiguous linear pixels; construction copies caller data."""

    pixels: Pixels
    name: str = "Untitled"
    color_space: str = "linear-srgb"
    note: str = ""

    def __post_init__(self) -> None:
        """Validate working-space values and take an independent read-only copy."""
        data = np.array(self.pixels, dtype=np.float32, order="C", copy=True)
        if data.ndim != 3 or data.shape[2] not in (1, 3) or min(data.shape[:2]) < 2:
            raise ValueError("Expected an H × W × 1 or H × W × 3 image (at least 2 × 2).")
        if not np.isfinite(data).all() or data.min() < 0:
            raise ValueError("Pixels must be finite and nonnegative.")
        if self.color_space not in ("linear-srgb", "linear-gray"):
            raise ValueError("Unsupported working color space.")
        data.flags.writeable = False
        object.__setattr__(self, "pixels", data)

    @classmethod
    def _from_owned_validated(
        cls, pixels: Pixels, name: str, color_space: str, note: str
    ) -> "ImageFrame":
        """Transfer an exclusively owned, validated reader buffer without a copy.

        Internal TIFF-reader contract: finite, nonnegative values are guaranteed
        by the unsigned-integer conversion table. The caller relinquishes every
        writable reference. Ordinary callers must use the copying constructor.
        """
        if (
            not isinstance(pixels, np.ndarray)
            or pixels.dtype != np.float32
            or not pixels.flags.owndata
            or not pixels.flags.c_contiguous
            or pixels.ndim != 3
            or pixels.shape[2] not in (1, 3)
            or min(pixels.shape[:2]) < 2
        ):
            raise ValueError("Expected an owned, contiguous float32 image buffer.")
        if color_space not in ("linear-srgb", "linear-gray"):
            raise ValueError("Unsupported working color space.")
        frame = object.__new__(cls)
        pixels.flags.writeable = False
        for key, value in (
            ("pixels", pixels),
            ("name", name),
            ("color_space", color_space),
            ("note", note),
        ):
            object.__setattr__(frame, key, value)
        return frame

    @property
    def width(self) -> int:
        """Return the source image width in pixels."""
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        """Return the source image height in pixels."""
        return int(self.pixels.shape[0])


def srgb_to_linear(data: ArrayLike) -> Pixels:
    """Decode encoded sRGB samples into float32 linear-light values."""
    data = np.asarray(data, dtype=np.float32)
    return np.where(data <= 0.04045, data / 12.92, ((data + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(data: ArrayLike) -> Pixels:
    """Encode nonnegative working values with the sRGB transfer curve."""
    data = np.maximum(np.asarray(data, dtype=np.float32), 0)
    return np.where(data <= 0.0031308, data * 12.92, 1.055 * data ** (1 / 2.4) - 0.055)


@lru_cache(maxsize=1)
def _display_transfer() -> tuple[ByteArray, BoolArray]:
    """Exact uint8 transfer for float32 buckets, with ambiguous buckets marked.

    Top 20 float bits index a 1 MiB table. Only buckets crossing an output
    rounding boundary require the original gamma calculation at display time.
    This does not quantize the source or change any exported pixels.
    """
    bits = np.arange(1 << 20, dtype=np.uint32) << 12
    lower = bits.view(np.float32)
    upper = (bits | 4095).view(np.float32)
    codes = []
    for values in (lower, upper):
        values = np.nan_to_num(np.clip(values, 0, 1), copy=False)
        codes.append(np.rint(linear_to_srgb(values) * 255).astype(np.uint8))
    boundaries = codes[0] != codes[1]
    codes[0].flags.writeable = boundaries.flags.writeable = False
    return codes[0], boundaries


def display_pixels(data: Pixels, color_space: str) -> ByteArray:
    """Quantize working pixels for display without changing the source or export."""
    if color_space != "linear-srgb":
        return np.ascontiguousarray(np.rint(np.clip(data, 0, 1) * 255), dtype=np.uint8)
    values = np.asarray(data, dtype=np.float32)
    table, boundaries = _display_transfer()
    index = values.view(np.uint32) >> 12
    result = table[index]
    exact = boundaries[index]
    if exact.any():
        encoded = linear_to_srgb(values[exact])
        result[exact] = np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(result)
