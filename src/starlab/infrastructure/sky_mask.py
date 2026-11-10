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

    def __call__(self, frame, source=None, cancel=None):
        check_cancel(cancel)
        if self.last_frame is not None and self.last_frame() is frame:
            return self.last_mask
        cache = None
        if source is not None and self.persistent_cache:
            source = Path(source)
            stat = source.stat()
            identity = (
                os.path.normcase(str(source.resolve())),
                stat.st_size,
                stat.st_mtime_ns,
                self.recipe,
                self.model_identity,
            )
            key = hashlib.sha256(repr(identity).encode()).hexdigest()
            cache = self.cache_directory / f"{key}.npz"
            try:
                with np.load(cache, allow_pickle=False) as stored:
                    mask = SkyMask(
                        int(stored["cell_px"]), stored["classes"], tuple(stored["shape"])
                    )
                if mask.shape != (frame.height, frame.width):
                    raise ValueError("Cached mask shape changed.")
                self.last_frame, self.last_mask = weakref.ref(frame), mask
                return mask
            except (OSError, ValueError, KeyError):
                pass
        if self.segmenter is None:
            self.segmenter = OnnxSkySegmenter()
        if source is not None:
            with tifffile.TiffFile(source) as tif:
                page, _ = _validated_page(tif)
                raw = _mapped_pixels(page)
                mask = build_sky_mask(
                    (frame.height, frame.width),
                    lambda top, bottom: raw[top:bottom],
                    self.segmenter,
                    cancel,
                    block_mean=_block_mean,
                )
        else:

            def read(top, bottom):
                pixels = frame.pixels[top:bottom]
                return linear_to_srgb(pixels) if frame.color_space == "linear-srgb" else pixels

            mask = build_sky_mask(
                (frame.height, frame.width), read, self.segmenter, cancel, block_mean=_block_mean
            )
        self.last_frame, self.last_mask = weakref.ref(frame), mask
        if cache is not None:
            temporary = None
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=cache.parent, suffix=".npz", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(
                        stream, cell_px=mask.cell_px, classes=mask.classes, shape=mask.shape
                    )
                os.replace(temporary, cache)
            except OSError:
                logger.debug("Sky-mask cache could not be written.", exc_info=True)
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return mask
