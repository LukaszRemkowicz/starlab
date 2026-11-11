"""Owner-provided Starligner ONNX model and disposable full-image mask cache."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import cast

import cv2
import numpy as np
import tifffile
from numba import get_num_threads, prange, set_num_threads

from ..domain.array_types import FloatArray, IntArray
from ..domain.imaging import ImageFrame, check_cancel, linear_to_srgb
from ..domain.jit import njit
from ..domain.sky_mask import EncodedArray, SkyMask, SkySegmenter, build_sky_mask
from .tiff_io import _mapped_pixels, _validated_page

MODEL_DIRECTORY = Path(__file__).with_name("models")
# Inference contract of the packaged sky-mask.onnx; the hash pins these values to that file.
MODEL_SHA256 = "b39141ebe7339f3aec897c7da058a935032540f80c1da04d7abf7fc1650d606d"
MODEL_TILE = 256
MODEL_OVERLAP = 32
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ImageDimensions:
    """Encoded source dimensions needed before full pixel import completes."""

    width: int
    height: int


@njit(cache=True, nogil=True, parallel=True)
def _integer_block_mean(strip: EncodedArray, factor: int) -> FloatArray:
    """Average unsigned box samples exactly using an integer accumulator."""
    height, width, channels = strip.shape
    result: FloatArray = np.empty((height // factor, width // factor, channels), np.float32)
    divisor: np.float32 = np.float32(factor * factor)
    for row in prange(result.shape[0]):
        for col in range(result.shape[1]):
            for channel in range(channels):
                total: np.uint32 = np.uint32(0)
                for y in range(factor):
                    for x in range(factor):
                        total += strip[row * factor + y, col * factor + x, channel]
                result[row, col, channel] = np.float32(total) / divisor
    return result


def _block_mean(strip: EncodedArray, factor: int) -> FloatArray:
    """Use exact integer reduction when safe, otherwise NumPy float32 averaging."""
    # Up to 16x16 uint16 samples sum exactly in float32, as in the reference mean.
    if strip.dtype.kind == "u" and strip.dtype.itemsize <= 2 and factor <= 16:
        if not strip.dtype.isnative:
            strip = strip.astype(np.uint16)
        threads: int = get_num_threads()
        try:
            set_num_threads(min(4, threads))
            return _integer_block_mean(strip, factor)
        finally:
            set_num_threads(threads)
    height, width, channels = strip.shape
    return strip.reshape(height // factor, factor, width // factor, factor, channels).mean(
        axis=(1, 3), dtype=np.float32
    )


class OnnxSkySegmenter:
    """Own the verified OpenCV network and its encoded-tile inference contract."""

    batch_size = 8

    def __init__(self) -> None:
        """Validate the packaged weight hash before opening its network graph."""
        weights: bytes = (MODEL_DIRECTORY / "sky-mask.onnx").read_bytes()
        self.identity = hashlib.sha256(weights).hexdigest()
        if self.identity != MODEL_SHA256:
            raise ValueError("Sky model does not match its recorded identity.")
        self.tile, self.overlap = MODEL_TILE, MODEL_OVERLAP
        # Load the exact bytes that were verified, not the file a second time.
        self.network = cv2.dnn.readNetFromONNX(np.frombuffer(weights, np.uint8))

    def sky_logits(self, image: FloatArray) -> FloatArray:
        """Infer one H×W raw logit plane from a normalized H×W×3 float32 tile."""
        self.network.setInput(np.ascontiguousarray(image.transpose(2, 0, 1)[None]), "image")
        return cast(FloatArray, self.network.forward("sky_logit")[0, 0])

    def sky_logits_batch(self, images: FloatArray) -> FloatArray:
        """Infer N×H×W logit planes from normalized N×H×W×3 float32 tiles."""
        self.network.setInput(np.ascontiguousarray(images.transpose(0, 3, 1, 2)), "image")
        return cast(FloatArray, self.network.forward("sky_logit")[:, 0])


class SkyMaskService:
    """One worker owns the network. Image and slider buffers remain independent."""

    def __init__(
        self,
        cache_directory: Path | str | None = None,
        segmenter: SkySegmenter | None = None,
        *,
        persistent_cache: bool = True,
    ) -> None:
        """Configure a model owner; production opts out of persisted photo masks."""
        self.persistent_cache = persistent_cache
        self.cache_directory = Path(
            cache_directory or Path(tempfile.gettempdir()) / "star-effects-lab/sky-masks"
        )
        self.segmenter: SkySegmenter | None = segmenter
        self.last_frame: weakref.ReferenceType[ImageFrame | _ImageDimensions] | None = None
        self.last_mask: SkyMask | None = None
        recipe: Path = Path(__file__).parents[1] / "domain/sky_mask.py"
        self.model_identity: str = cast(str, getattr(segmenter, "identity", MODEL_SHA256))
        self.recipe = hashlib.sha256(recipe.read_bytes() + Path(__file__).read_bytes()).hexdigest()

    def from_source(self, source: Path, cancel: Event | None = None) -> SkyMask:
        """Build a full-coordinate mask directly from an encoded TIFF source."""
        with tifffile.TiffFile(source) as tiff_file:
            page, _ = _validated_page(tiff_file)
            dimensions: _ImageDimensions = _ImageDimensions(page.imagewidth, page.imagelength)
        return self(dimensions, source, cancel)

    def _cache_path(self, source: Path | str | None) -> Path | None:
        """Identify optional persisted masks without opening their data."""
        if source is None or not self.persistent_cache:
            return None
        path: Path = Path(source)
        stat: os.stat_result = path.stat()
        identity: tuple[str, int, int, str, str] = (
            os.path.normcase(str(path.resolve())),
            stat.st_size,
            stat.st_mtime_ns,
            self.recipe,
            self.model_identity,
        )
        key: str = hashlib.sha256(repr(identity).encode()).hexdigest()
        return self.cache_directory / f"{key}.npz"

    @staticmethod
    def _load_cached(path: Path | None, shape: tuple[int, int]) -> SkyMask | None:
        """Read an optional cache entry; malformed or unavailable data is a miss."""
        if path is None:
            return None
        try:
            with np.load(path, allow_pickle=False) as stored:
                stored_shape: IntArray = stored["shape"]
                if stored_shape.shape != (2,) or not np.array_equal(stored_shape, shape):
                    return None
                return SkyMask(int(stored["cell_px"]), stored["classes"], shape)
        except OSError, ValueError, KeyError:
            return None

    def _from_encoded_source(
        self, shape: tuple[int, int], source: Path | str, cancel: Event | None
    ) -> SkyMask:
        """Run mask preprocessing on encoded TIFF samples, never working RGB."""
        assert self.segmenter is not None
        with tifffile.TiffFile(source) as tiff_file:
            page, _ = _validated_page(tiff_file)
            raw: EncodedArray = _mapped_pixels(page)
            return build_sky_mask(
                shape,
                lambda top, bottom: raw[top:bottom],
                self.segmenter,
                cancel,
                block_mean=_block_mean,
            )

    def _from_working_frame(self, frame: ImageFrame, cancel: Event | None) -> SkyMask:
        """Encode in-memory working samples when no TIFF source exists."""

        assert self.segmenter is not None

        def read(top: int, bottom: int) -> FloatArray:
            """Return encoded source-domain rows for mask preprocessing."""
            pixels: FloatArray = frame.pixels[top:bottom]
            return linear_to_srgb(pixels) if frame.color_space == "linear-srgb" else pixels

        return build_sky_mask(
            (frame.height, frame.width), read, self.segmenter, cancel, block_mean=_block_mean
        )

    @staticmethod
    def _save_cached(path: Path | None, mask: SkyMask) -> None:
        """Atomically persist an optional mask without making cache writes mandatory."""
        if path is None:
            return
        temporary: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".npz", delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(
                    stream, cell_px=mask.cell_px, classes=mask.classes, shape=mask.shape
                )
            os.replace(temporary, path)
        except OSError:
            logger.debug("Sky-mask cache could not be written.", exc_info=True)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Temporary sky-mask cache could not be removed.", exc_info=True)

    def __call__(
        self,
        frame: ImageFrame | _ImageDimensions,
        source: Path | None = None,
        cancel: Event | None = None,
    ) -> SkyMask:
        """Reuse this frame's mask or build one, keeping persistence optional."""
        check_cancel(cancel)
        if self.last_frame is not None and self.last_frame() is frame:
            assert self.last_mask is not None
            return self.last_mask
        cache: Path | None = self._cache_path(source)
        mask: SkyMask | None = self._load_cached(cache, (frame.height, frame.width))
        if mask is None:
            if self.segmenter is None:
                self.segmenter = OnnxSkySegmenter()
            if source is not None:
                mask = self._from_encoded_source((frame.height, frame.width), source, cancel)
            else:
                if not isinstance(frame, ImageFrame):
                    raise ValueError("In-memory masking requires image pixels.")
                mask = self._from_working_frame(frame, cancel)
            check_cancel(cancel)
            self._save_cached(cache, mask)
        self.last_frame, self.last_mask = weakref.ref(frame), mask
        return mask
