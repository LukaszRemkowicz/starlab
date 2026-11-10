"""Sky/foreground classes and the preprocessing used by Starligner PR388.

Adapted with the owner's permission from commit cd40fd16e3552e3c1519de9de7272b77c6333d0f.
The input rows are encoded image samples, not linear-light working pixels.
"""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Any

import numpy as np

from .imaging import check_cancel

FOREGROUND, SKY, UNKNOWN = 0, 1, 2


def _classify_logits(logits):
    # The division feeding this function produces float32. Its first value
    # whose float64 sigmoid reaches 0.6 is one ULP above float32(log(1.5));
    # the negative value is the last whose sigmoid is at most 0.4.
    boundary = np.nextafter(np.float32(np.log(1.5)), np.float32(np.inf))
    classes = np.full(logits.shape, UNKNOWN, np.uint8)
    classes[logits >= boundary] = SKY
    classes[logits <= -boundary] = FOREGROUND
    return classes


@dataclass(frozen=True, eq=False)
class SkyMask:
    cell_px: int
    classes: np.ndarray
    shape: tuple[int, int]

    def __post_init__(self):
        h, w = self.shape
        values = np.asarray(self.classes)
        if (
            not isinstance(self.cell_px, (int, np.integer))
            or self.cell_px < 1
            or h < 1
            or w < 1
            or values.shape
            != ((h + self.cell_px - 1) // self.cell_px, (w + self.cell_px - 1) // self.cell_px)
        ):
            raise ValueError("Sky mask must cover the full image.")
        if not np.isfinite(values).all() or not np.isin(values, (FOREGROUND, SKY, UNKNOWN)).all():
            raise ValueError("Invalid sky-mask classes.")
        values = np.array(values, dtype=np.uint8, copy=True)
        values.flags.writeable = False
        object.__setattr__(self, "classes", values)

    def keep(self, xy):
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        finite = np.isfinite(points).all(axis=1)
        cells = np.floor(np.where(finite[:, None], points, 0) / self.cell_px)
        rows = np.clip(cells[:, 1], 0, self.classes.shape[0] - 1).astype(np.int64)
        cols = np.clip(cells[:, 0], 0, self.classes.shape[1] - 1).astype(np.int64)
        height, width = self.shape
        inside = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        return finite & inside & (self.classes[rows, cols] != FOREGROUND)

    def allowed_at(self, x, y):
        return bool(self.keep(((x, y),))[0])

    def labels_region(self, bounds, sampling=1):
        if not isinstance(sampling, (int, np.integer)) or sampling < 1:
            raise ValueError("Mask sampling must be a positive integer pixel stride.")
        left, top, right, bottom = bounds
        rows = np.clip(
            np.arange(top, bottom, sampling) // self.cell_px, 0, self.classes.shape[0] - 1
        ).astype(np.int64)
        cols = np.clip(
            np.arange(left, right, sampling) // self.cell_px, 0, self.classes.shape[1] - 1
        ).astype(np.int64)
        return self.classes[np.ix_(rows, cols)]

    def region(self, bounds, sampling=1):
        return self.labels_region(bounds, sampling) != FOREGROUND


def _reduce_encoded_rows(
    out_h: int,
    out_w: int,
    factor: int,
    read_rows: Callable[[int, int], np.ndarray],
    cancel: Event | None,
    block_mean: Callable[[np.ndarray, int], np.ndarray] | None,
) -> np.ndarray:
    """Read bounded encoded strips and preserve the original float32 block means."""
    reduced = np.empty((out_h, out_w, 3), np.float32)
    strip_rows = max(factor, 512 // factor * factor)
    for top in range(0, out_h * factor, strip_rows):
        check_cancel(cancel)
        bottom = min(top + strip_rows, out_h * factor)
        strip = np.asarray(read_rows(top, bottom))
        if strip.ndim == 2:
            strip = strip[..., None]
        if (
            strip.ndim != 3
            or strip.shape[0] != bottom - top
            or strip.shape[1] < out_w * factor
            or strip.shape[2] not in (1, 3)
        ):
            raise ValueError("Unexpected encoded image strip for the sky model.")
        if strip.shape[2] == 1:
            strip = np.repeat(strip, 3, axis=2)
        strip = strip[:, : out_w * factor]
        if block_mean is None:
            blocks = strip.reshape(-1, factor, out_w, factor, 3)
            means = blocks.mean(axis=(1, 3), dtype=np.float32)
        else:
            means = block_mean(strip, factor)
        reduced[top // factor : bottom // factor] = means
    return reduced


def _normalize_channels(reduced: np.ndarray, cancel: Event | None) -> np.ndarray:
    """Normalize each encoded channel into the model's float16 input range."""
    normalized = np.empty_like(reduced, dtype=np.float16)
    for channel in range(3):
        check_cancel(cancel)
        sample = reduced[::4, ::4, channel]
        median = float(np.median(sample))
        high = float(np.percentile(sample, 99.5))
        normalized[..., channel] = np.clip(
            (reduced[..., channel] - median) / max(high - median, 1e-6), -1, 8
        )
    return normalized


def _infer_overlapping_logits(
    normalized: np.ndarray, segmenter: Any, cancel: Event | None
) -> np.ndarray:
    """Average validated model logits in the original tile and batch order."""
    out_h, out_w = normalized.shape[:2]
    tile, overlap = segmenter.tile, segmenter.overlap
    if not 0 <= overlap < tile:
        raise ValueError("Invalid model tile overlap.")
    ph, pw = max(out_h, tile), max(out_w, tile)
    normalized = np.pad(normalized, ((0, ph - out_h), (0, pw - out_w), (0, 0)), mode="edge")
    total = np.zeros((ph, pw), np.float32)
    count = np.zeros_like(total)
    stride = tile - overlap
    positions = [
        (top, left)
        for top in sorted({*range(0, ph - tile, stride), ph - tile})
        for left in sorted({*range(0, pw - tile, stride), pw - tile})
    ]
    batch_size = getattr(segmenter, "batch_size", 1)
    for start in range(0, len(positions), batch_size):
        check_cancel(cancel)
        batch_positions = positions[start : start + batch_size]
        parts = np.stack(
            [normalized[top : top + tile, left : left + tile] for top, left in batch_positions],
            dtype=np.float32,
        )
        if batch_size == 1:
            logits = np.asarray(segmenter.sky_logits(parts[0]), dtype=np.float32)[None]
        else:
            logits = np.asarray(segmenter.sky_logits_batch(parts), dtype=np.float32)
        check_cancel(cancel)
        if logits.shape != (len(parts), tile, tile) or not np.isfinite(logits).all():
            raise ValueError("Sky model returned invalid logits.")
        for (top, left), tile_logits in zip(batch_positions, logits):
            total[top : top + tile, left : left + tile] += tile_logits
            count[top : top + tile, left : left + tile] += 1
    return (total / count)[:out_h, :out_w]


def build_sky_mask(shape, read_rows, segmenter, cancel=None, block_mean=None):
    """Bounded row reads, robust encoded-channel normalization, overlap-averaged logits."""
    height, width = shape
    factor = max(2, round(max(shape) / 4000))
    out_h, out_w = height // factor, width // factor
    if min(out_h, out_w) < 1:
        raise ValueError("Image is too small for sky segmentation.")
    reduced = _reduce_encoded_rows(out_h, out_w, factor, read_rows, cancel, block_mean)
    normalized = _normalize_channels(reduced, cancel)
    del reduced
    classes = _classify_logits(_infer_overlapping_logits(normalized, segmenter, cancel))
    gh, gw = (height + factor - 1) // factor, (width + factor - 1) // factor
    classes = np.pad(classes, ((0, gh - out_h), (0, gw - out_w)), mode="edge")
    check_cancel(cancel)
    return SkyMask(factor, classes, shape)
