"""Shared-memory exchange with the Photoshop plug-in: pixels in, result out, nothing on disk."""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterable
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from threading import Event
from typing import cast

import numpy as np

from ..domain.effects.settings import EffectSettings
from ..domain.imaging import ImageFrame, Pixels, check_cancel
from ..domain.stars import DetectionSettings
from .preset_json import decode_session_settings, encode_session_settings
from .tiff_io import EncodedPixels, _convert_integer_image, _encoded_rows

logger: logging.Logger = logging.getLogger("starlab")

MAGIC = b"SELB"
VERSION = 2
HEADER_SIZE = 64
STATUS_OFFSET = 28
MODE_OFFSET, SETTINGS_LENGTH_OFFSET = 32, 36
MODE_INTERACTIVE, MODE_REPEAT = 0, 1
SETTINGS_CAPACITY = 65536  # JSON area right after the pixels
STATUS_READY, STATUS_WRITTEN, STATUS_CANCELLED = 0, 1, 2
# Exit codes of `--photoshop-bridge`: the plug-in only needs to tell applied from discarded.
EXIT_APPLIED, EXIT_CANCELLED = 0, 2
LAYER_NAME = "Photoshop layer"


class PhotoshopBridge:
    """Own one plug-in shared-memory block for the application's lifetime (never unlinked)."""

    def __init__(self, name: str) -> None:
        """Attach to the plug-in's block and validate its header before any use."""
        # track=False: the resource tracker must never unlink the plug-in's block at exit.
        self.memory: SharedMemory = SharedMemory(name=name, create=False, track=False)
        try:
            self.width, self.height, self.channels, self.depth, self.maximum = self._validate()
            self.mode: int = struct.unpack_from("<I", self._buffer, MODE_OFFSET)[0]
        except ValueError:
            self.close()
            raise
        self.dtype: type[np.uint8] | type[np.uint16] = np.uint8 if self.depth == 8 else np.uint16
        logger.info(
            "Photoshop bridge %s: mode %d, %dx%d, %d channel(s), %d-bit, max %d",
            name,
            self.mode,
            self.width,
            self.height,
            self.channels,
            self.depth,
            self.maximum,
        )

    @property
    def _buffer(self) -> memoryview:
        """Return the mapped block (``None`` only after close, which is a programming error)."""
        buffer = self.memory.buf
        assert buffer is not None
        return buffer

    def _validate(self) -> tuple[int, int, int, int, int]:
        """Check magic, version, geometry, depth, white code, status, mode and block length."""
        buffer: memoryview = self._buffer
        if len(buffer) < HEADER_SIZE:
            raise ValueError("Photoshop block is too small to hold a header.")
        magic: bytes = bytes(buffer[:4])
        version, width, height, channels, depth, maximum, status, mode, length = struct.unpack_from(
            "<9I", buffer, 4
        )
        if magic != MAGIC:
            raise ValueError("Photoshop block has the wrong signature.")
        if version != VERSION:
            raise ValueError(f"Unsupported Photoshop block version {version}.")
        if width < 2 or height < 2:
            raise ValueError("Photoshop layer must be at least 2 × 2 pixels.")
        if channels not in (1, 3):
            raise ValueError(f"Unsupported Photoshop channel count {channels}.")
        if depth not in (8, 16):
            raise ValueError(f"Unsupported Photoshop bit depth {depth}.")
        if not 1 <= maximum <= (1 << depth) - 1:
            raise ValueError(f"Invalid Photoshop white code {maximum} for {depth}-bit samples.")
        if status != STATUS_READY:
            raise ValueError("Photoshop block does not hold fresh input pixels.")
        pixel_bytes: int = width * height * channels * depth // 8
        self._check_request(mode, length, len(buffer) - HEADER_SIZE - pixel_bytes)
        return width, height, channels, depth, maximum

    @staticmethod
    def _check_request(mode: int, length: int, room: int) -> None:
        """Check the mode word, the settings length and that the settings area fits."""
        if mode not in (MODE_INTERACTIVE, MODE_REPEAT):
            raise ValueError(f"Unsupported Photoshop mode {mode}.")
        if length > SETTINGS_CAPACITY or (mode == MODE_REPEAT and length == 0):
            raise ValueError(f"Invalid Photoshop settings length {length}.")
        if room < SETTINGS_CAPACITY:
            raise ValueError("Photoshop block is smaller than its header describes.")

    @property
    def status(self) -> int:
        """Return the status word currently stored in the block."""
        return int(struct.unpack_from("<I", self._buffer, STATUS_OFFSET)[0])

    def _set_status(self, value: int) -> None:
        """Publish a status word; the plug-in reads it only after this process exits."""
        struct.pack_into("<I", self._buffer, STATUS_OFFSET, value)

    def _settings_offset(self) -> int:
        """Return where the settings area starts (right after the pixels)."""
        return HEADER_SIZE + self.width * self.height * self.channels * self.depth // 8

    def settings(self) -> tuple[DetectionSettings, EffectSettings]:
        """Decode the stored settings the plug-in placed in the block (repeat mode)."""
        length: int = struct.unpack_from("<I", self._buffer, SETTINGS_LENGTH_OFFSET)[0]
        start: int = self._settings_offset()
        return decode_session_settings(bytes(self._buffer[start : start + length]))

    def store_settings(self, detection: DetectionSettings, effects: EffectSettings) -> None:
        """Write the settings JSON for the plug-in to remember (before pixels and status)."""
        raw: bytes = encode_session_settings(detection, effects)
        if len(raw) > SETTINGS_CAPACITY:
            raise ValueError("Settings are too large for the Photoshop block.")
        start: int = self._settings_offset()
        self._buffer[start : start + len(raw)] = raw
        struct.pack_into("<I", self._buffer, SETTINGS_LENGTH_OFFSET, len(raw))

    def _samples(self) -> EncodedPixels:
        """Return a writable H×W×C view of the block's pixels (no copy)."""
        return cast(
            EncodedPixels,
            np.ndarray(
                (self.height, self.width, self.channels),
                self.dtype,
                self._buffer,
                HEADER_SIZE,
            ),
        )

    def _checked(self, rows: EncodedPixels) -> EncodedPixels:
        """Reject codes above white before they index the transfer table."""
        if rows.size and int(rows.max()) > self.maximum:
            raise ValueError(f"Photoshop pixel code exceeds the declared white {self.maximum}.")
        return rows

    def read(self, path: Path | None = None, cancel: Event | None = None) -> ImageFrame:
        """Convert the layer to linear working pixels in strips; ``path`` is ignored."""
        check_cancel(cancel)
        samples: EncodedPixels = self._samples()
        return _convert_integer_image(
            lambda top, bottom: self._checked(samples[top:bottom]),
            (self.height, self.width, self.channels),
            self.maximum,
            Path(LAYER_NAME),
            "sRGB assumed from Photoshop layer",
            cancel,
        )

    def read_preview(
        self, path: Path | None = None, cancel: Event | None = None, max_edge: int = 1600
    ) -> tuple[ImageFrame, float, int, int]:
        """Return a strided thumbnail, its scale and the full size; the layer is not copied."""
        check_cancel(cancel)
        if max_edge < 2:
            raise ValueError("Preview edge must be at least two pixels.")
        longest: int = max(self.width, self.height)
        # A stride below the short side keeps ImageFrame's 2×2 minimum.
        stride: int = min(-(-longest // max_edge), min(self.width, self.height) - 1)
        stride = max(1, stride)
        samples: EncodedPixels = self._samples()
        thumbnail: EncodedPixels = samples[::stride, ::stride]
        frame: ImageFrame = _convert_integer_image(
            lambda top, bottom: self._checked(thumbnail[top:bottom]),
            cast(tuple[int, int, int], thumbnail.shape),
            self.maximum,
            Path(LAYER_NAME),
            "sRGB assumed from Photoshop layer",
            cancel,
        )
        return frame, 1 / stride, self.width, self.height

    def write(
        self,
        path: Path | None,
        frame: ImageFrame,
        pixels: Pixels | Iterable[Pixels],
        *,
        source_path: Path | None = None,
        cancel: Event | None = None,
    ) -> None:
        """Encode linear bands into the block in place, then publish status 1.

        A failure or cancellation part-way leaves status 0 and the already written rows
        modified (no full-size backup exists); the plug-in must ignore pixels unless
        status is 1. The path arguments are ignored: nothing here touches the disk.
        """
        shape: tuple[int, ...] = frame.pixels.shape
        if shape != (self.height, self.width, self.channels):
            raise ValueError("Result size differs from the Photoshop layer.")
        check_cancel(cancel)
        samples: EncodedPixels = self._samples()
        for index, row in enumerate(_encoded_rows(pixels, shape, cancel, self.maximum)):
            samples[index] = row.reshape(self.width, self.channels)
        self._set_status(STATUS_WRITTEN)

    def cancel(self) -> None:
        """Mark the layer as discarded unless a result was already published."""
        if self.status != STATUS_WRITTEN:
            self._set_status(STATUS_CANCELLED)

    @property
    def exit_code(self) -> int:
        """Return the process exit code that reports the outcome to the plug-in."""
        return EXIT_APPLIED if self.status == STATUS_WRITTEN else EXIT_CANCELLED

    def close(self) -> None:
        """Detach from the block without unlinking it (the plug-in owns it)."""
        try:
            self.memory.close()
        except BufferError:  # a stray view; the OS unmaps it at process exit
            logger.warning("Photoshop block still has exported views at close.")
