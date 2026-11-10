"""Owner-provided Starligner ONNX model and disposable full-image mask cache."""

import hashlib
import json
import logging
import os
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
from numba import get_num_threads, njit, prange, set_num_threads

from ..domain.imaging import check_cancel, linear_to_srgb
from ..domain.sky_mask import SkyMask, build_sky_mask
from .tiff_io import _mapped_pixels, _validated_page

MODEL_DIRECTORY = Path(__file__).with_name("models")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ImageDimensions:
    width: int
    height: int


@njit(cache=True, nogil=True, parallel=True)
def _integer_block_mean(strip, factor):
    height, width, channels = strip.shape
    result = np.empty((height // factor, width // factor, channels), np.float32)
    divisor = np.float32(factor * factor)
    for row in prange(result.shape[0]):
        for col in range(result.shape[1]):
            for channel in range(channels):
                total = np.uint32(0)
                for y in range(factor):
                    for x in range(factor):
                        total += strip[row * factor + y, col * factor + x, channel]
                result[row, col, channel] = np.float32(total) / divisor
    return result


def _block_mean(strip, factor):
    # Up to 16x16 uint16 samples sum exactly in float32, as in the reference mean.
    if strip.dtype.kind == "u" and strip.dtype.itemsize <= 2 and factor <= 16:
        if not strip.dtype.isnative:
            strip = strip.astype(np.uint16)
        threads = get_num_threads()
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
    batch_size = 8

    def __init__(self):
        model = MODEL_DIRECTORY / "sky-mask.onnx"
        meta = json.loads((MODEL_DIRECTORY / "sky-mask.json").read_text(encoding="utf-8"))
        self.identity = hashlib.sha256(model.read_bytes()).hexdigest()
        if self.identity != meta["onnx_sha256"]:
            raise ValueError("Sky model does not match its recorded identity.")
        self.tile, self.overlap = int(meta["tile"]), int(meta["overlap"])
        self.network = cv2.dnn.readNetFromONNX(str(model))

    def sky_logits(self, image):
        self.network.setInput(np.ascontiguousarray(image.transpose(2, 0, 1)[None]), "image")
        return self.network.forward("sky_logit")[0, 0]

    def sky_logits_batch(self, images):
        self.network.setInput(np.ascontiguousarray(images.transpose(0, 3, 1, 2)), "image")
        return self.network.forward("sky_logit")[:, 0]


class SkyMaskService:
    """One worker owns the network. Image and slider buffers remain independent."""

    def __init__(self, cache_directory=None, segmenter=None, *, persistent_cache=True):
        self.persistent_cache = persistent_cache
        self.cache_directory = Path(
            cache_directory or Path(tempfile.gettempdir()) / "star-effects-lab/sky-masks"
        )
        self.segmenter = segmenter
        self.last_frame = self.last_mask = None
        recipe = Path(__file__).parents[1] / "domain/sky_mask.py"
        meta = (MODEL_DIRECTORY / "sky-mask.json").read_bytes()
        self.model_identity = getattr(segmenter, "identity", json.loads(meta)["onnx_sha256"])
        self.recipe = hashlib.sha256(
            recipe.read_bytes() + Path(__file__).read_bytes() + meta
        ).hexdigest()

    def from_source(self, source, cancel=None):
        with tifffile.TiffFile(source) as tif:
            page, _ = _validated_page(tif)
            dimensions = _ImageDimensions(page.imagewidth, page.imagelength)
        return self(dimensions, source, cancel)

    def _cache_path(self, source: Path | str | None) -> Path | None:
        """Identify optional persisted masks without opening their data."""
        if source is None or not self.persistent_cache:
            return None
        path = Path(source)
        stat = path.stat()
        identity = (
            os.path.normcase(str(path.resolve())),
            stat.st_size,
            stat.st_mtime_ns,
            self.recipe,
            self.model_identity,
        )
        key = hashlib.sha256(repr(identity).encode()).hexdigest()
        return self.cache_directory / f"{key}.npz"

    def _load_cached(self, path: Path | None, shape: tuple[int, int]) -> SkyMask | None:
        """Read an optional cache entry; malformed or unavailable data is a miss."""
        if path is None:
            return None
        try:
            with np.load(path, allow_pickle=False) as stored:
                mask = SkyMask(int(stored["cell_px"]), stored["classes"], tuple(stored["shape"]))
            return mask if mask.shape == shape else None
        except (OSError, ValueError, KeyError):
            return None

    def _from_encoded_source(self, shape: tuple[int, int], source: Path | str, cancel) -> SkyMask:
        """Run mask preprocessing on encoded TIFF samples, never working RGB."""
        with tifffile.TiffFile(source) as tif:
            page, _ = _validated_page(tif)
            raw = _mapped_pixels(page)
            return build_sky_mask(
                shape,
                lambda top, bottom: raw[top:bottom],
                self.segmenter,
                cancel,
                block_mean=_block_mean,
            )

    def _from_working_frame(self, frame, cancel) -> SkyMask:
        """Encode in-memory working samples when no TIFF source exists."""

        def read(top: int, bottom: int) -> np.ndarray:
            pixels = frame.pixels[top:bottom]
            return linear_to_srgb(pixels) if frame.color_space == "linear-srgb" else pixels

        return build_sky_mask(
            (frame.height, frame.width), read, self.segmenter, cancel, block_mean=_block_mean
        )

    def _save_cached(self, path: Path | None, mask: SkyMask) -> None:
        """Atomically persist an optional mask without making cache writes mandatory."""
        if path is None:
            return
        temporary = None
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

    def __call__(self, frame, source=None, cancel=None) -> SkyMask:
        """Reuse this frame's mask or build one, keeping persistence optional."""
        check_cancel(cancel)
        if self.last_frame is not None and self.last_frame() is frame:
            return self.last_mask
        cache = self._cache_path(source)
        mask = self._load_cached(cache, (frame.height, frame.width))
        if mask is None:
            if self.segmenter is None:
                self.segmenter = OnnxSkySegmenter()
            mask = (
                self._from_encoded_source((frame.height, frame.width), source, cancel)
                if source is not None
                else self._from_working_frame(frame, cancel)
            )
            check_cancel(cancel)
            self._save_cached(cache, mask)
        self.last_frame, self.last_mask = weakref.ref(frame), mask
        return mask
