"""Sky/foreground classes and the preprocessing used by Starligner PR388.

Adapted with the owner's permission from commit cd40fd16e3552e3c1519de9de7272b77c6333d0f.
The input rows are encoded image samples, not linear-light working pixels.
"""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .array_types import BoolArray, Bounds, ByteArray, DoubleArray, FloatArray, IntArray
from .imaging import check_cancel

FOREGROUND, SKY, UNKNOWN = 0, 1, 2

type EncodedArray = ByteArray | NDArray[np.uint16] | FloatArray | DoubleArray
type HalfArray = NDArray[np.float16]
type RowReader = Callable[[int, int], EncodedArray]
type BlockMean = Callable[[EncodedArray, int], FloatArray]


class SkySegmenter(Protocol):
    """Model contract for normalized RGB tiles and same-size raw sky logits."""

    tile: int
    overlap: int

    def sky_logits(self, tile: FloatArray, /) -> FloatArray:
        """Return one raw logit per input pixel, without applying a sigmoid."""
        ...


class BatchedSkySegmenter(SkySegmenter, Protocol):
    """Optional inference interface selected only by a model's batch-size opt-in."""

    batch_size: int

    def sky_logits_batch(self, tiles: FloatArray, /) -> FloatArray:
        """Return one same-size logit plane per normalized RGB tile."""
        ...


def _classify_logits(logits: FloatArray) -> ByteArray:
    # The division feeding this function produces float32. Its first value
    # whose float64 sigmoid reaches 0.6 is one ULP above float32(log(1.5));
    # the negative value is the last whose sigmoid is at most 0.4.
    """Classify float32 logits at the exact 0.4/0.6 sigmoid rounding boundaries."""
    boundary: np.float32 = np.nextafter(np.float32(np.log(1.5)), np.float32(np.inf))
    classes: ByteArray = np.full(logits.shape, UNKNOWN, np.uint8)
    classes[logits >= boundary] = SKY
    classes[logits <= -boundary] = FOREGROUND
    return classes


@dataclass(frozen=True, eq=False)
class SkyMask:
    """Own immutable cell classes covering an image in native pixel coordinates."""

    cell_px: int
    classes: ByteArray
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        """Require full coverage and integral labels before copying to uint8 storage."""
        height, width = self.shape
        values: ByteArray = np.asarray(self.classes)
        if (
            not isinstance(self.cell_px, (int, np.integer))
            or self.cell_px < 1
            or height < 1
            or width < 1
            or values.shape
            != (
                (height + self.cell_px - 1) // self.cell_px,
                (width + self.cell_px - 1) // self.cell_px,
            )
        ):
            raise ValueError("Sky mask must cover the full image.")
        # Integer labels need only a range check; isin/isfinite would add image-sized temporaries.
        valid: bool = (
            bool(FOREGROUND <= values.min() and values.max() <= UNKNOWN)
            if values.dtype.kind in "biu"
            else bool(
                np.isfinite(values).all() and np.isin(values, (FOREGROUND, SKY, UNKNOWN)).all()
            )
        )
        if not valid:
            raise ValueError("Invalid sky-mask classes.")
        values = np.array(values, dtype=np.uint8, copy=True)
        values.flags.writeable = False
        object.__setattr__(self, "classes", values)

    def keep(self, positions: ArrayLike) -> BoolArray:
        """Return in-image sky/unknown membership for each floating-point source center."""
        points: DoubleArray = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        finite: BoolArray = np.isfinite(points).all(axis=1)
        cells: DoubleArray = np.floor(np.where(finite[:, None], points, 0) / self.cell_px)
        rows: IntArray = np.clip(cells[:, 1], 0, self.classes.shape[0] - 1).astype(np.int64)
        columns: IntArray = np.clip(cells[:, 0], 0, self.classes.shape[1] - 1).astype(np.int64)
        height, width = self.shape
        inside: BoolArray = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        return cast(BoolArray, finite & inside & (self.classes[rows, columns] != FOREGROUND))

    def allowed_at(self, x: float, y: float) -> bool:
        """Test one source-coordinate point without admitting off-image positions."""
        return bool(self.keep(((x, y),))[0])

    def labels_region(self, bounds: Bounds, sampling: int = 1) -> ByteArray:
        """Sample cell labels on the requested native-pixel stride grid."""
        if not isinstance(sampling, (int, np.integer)) or sampling < 1:
            raise ValueError("Mask sampling must be a positive integer pixel stride.")
        left, top, right, bottom = bounds
        rows: IntArray = np.clip(
            np.arange(top, bottom, sampling) // self.cell_px, 0, self.classes.shape[0] - 1
        ).astype(np.int64)
        columns: IntArray = np.clip(
            np.arange(left, right, sampling) // self.cell_px, 0, self.classes.shape[1] - 1
        ).astype(np.int64)
        return self.classes[np.ix_(rows, columns)]

    def region(self, bounds: Bounds, sampling: int = 1) -> BoolArray:
        """Return the sky/unknown keep plane on a sampled image region."""
        return cast(BoolArray, self.labels_region(bounds, sampling) != FOREGROUND)


def _reduce_encoded_rows(
    output_height: int,
    output_width: int,
    factor: int,
    read_rows: RowReader,
    cancel: Event | None,
    block_mean: BlockMean | None,
) -> FloatArray:
    """Read bounded encoded strips and preserve the original float32 block means."""
    reduced: FloatArray = np.empty((output_height, output_width, 3), np.float32)
    strip_rows: int = max(factor, 64 // factor * factor)
    for top in range(0, output_height * factor, strip_rows):
        check_cancel(cancel)
        bottom: int = min(top + strip_rows, output_height * factor)
        strip: EncodedArray = cast(EncodedArray, np.asarray(read_rows(top, bottom)))
        if strip.ndim == 2:
            strip = strip[..., None]
        if (
            strip.ndim != 3
            or strip.shape[0] != bottom - top
            or strip.shape[1] < output_width * factor
            or strip.shape[2] not in (1, 3)
        ):
            raise ValueError("Unexpected encoded image strip for the sky model.")
        if strip.shape[2] == 1:
            strip = cast(EncodedArray, np.repeat(strip, 3, axis=2))
        strip = strip[:, : output_width * factor]
        if block_mean is None:
            blocks: EncodedArray = strip.reshape(-1, factor, output_width, factor, 3)
            means: FloatArray = np.mean(blocks, axis=(1, 3), dtype=np.float32)
        else:
            means = block_mean(strip, factor)
        reduced[top // factor : bottom // factor] = means
    return reduced


type _Statistics = list[tuple[float, float]]


def _channel_statistics(reduced: FloatArray, cancel: Event | None) -> _Statistics:
    """Return each encoded channel's median and 99.5th percentile from the sampled grid."""
    statistics: _Statistics = []
    for channel in range(3):
        check_cancel(cancel)
        sample: FloatArray = reduced[::4, ::4, channel]
        statistics.append((float(np.median(sample)), float(np.percentile(sample, 99.5))))
    return statistics


def _normalized_tile(reduced: FloatArray, statistics: _Statistics) -> FloatArray:
    """Normalize one region into the model's float16 input range, returned as float32."""
    normalized: HalfArray = np.empty_like(reduced, dtype=np.float16)
    for channel, (median, high) in enumerate(statistics):
        normalized[..., channel] = np.clip(
            (reduced[..., channel] - median) / max(high - median, 1e-6), -1, 8
        )
    return normalized.astype(np.float32)


def _model_logits(
    segmenter: SkySegmenter, parts: FloatArray, batch_size: int, cancel: Event | None
) -> FloatArray:
    """Run one batch of normalized tiles and validate the returned logits."""
    if batch_size == 1:
        logits: FloatArray = np.asarray(segmenter.sky_logits(parts[0]), dtype=np.float32)[None]
    else:
        logits = np.asarray(
            cast(BatchedSkySegmenter, segmenter).sky_logits_batch(parts), dtype=np.float32
        )
    check_cancel(cancel)
    if logits.shape != (len(parts), *parts.shape[1:3]) or not np.isfinite(logits).all():
        raise ValueError("Sky model returned invalid logits.")
    return logits


def _infer_overlapping_logits(
    reduced: FloatArray, segmenter: SkySegmenter, cancel: Event | None
) -> ByteArray:
    """Classify overlap-averaged logits, normalizing tiles and releasing finished rows lazily."""
    output_height: int
    output_width: int
    output_height, output_width = reduced.shape[:2]
    overlap: int
    tile: int
    tile, overlap = segmenter.tile, segmenter.overlap
    if not 0 <= overlap < tile:
        raise ValueError("Invalid model tile overlap.")
    statistics: _Statistics = _channel_statistics(reduced, cancel)
    padded_height: int
    padded_width: int
    padded_height, padded_width = max(output_height, tile), max(output_width, tile)
    if (padded_height, padded_width) != (output_height, output_width):
        reduced = np.pad(
            reduced,
            ((0, padded_height - output_height), (0, padded_width - output_width), (0, 0)),
            mode="edge",
        )
    stride: int = tile - overlap
    tops: list[int] = sorted({*range(0, padded_height - tile, stride), padded_height - tile})
    lefts: list[int] = sorted({*range(0, padded_width - tile, stride), padded_width - tile})
    positions: list[tuple[int, int]] = [(top, left) for top in tops for left in lefts]
    batch_size: int = getattr(segmenter, "batch_size", 1)
    # Rows before the next tile row's top are final; a batch spans at most this many rows.
    window: int = tile + stride * -(-batch_size // len(lefts))
    total: FloatArray = np.zeros((window, padded_width), np.float32)
    count: FloatArray = np.zeros_like(total)
    classes: ByteArray = np.empty((output_height, output_width), np.uint8)
    base: int = 0

    def finish(limit: int) -> None:
        nonlocal base
        limit = min(limit, padded_height)
        if limit <= base:
            return
        rows: int = limit - base
        # NumPy preserves these float32 operands; its division stub widens the dtype.
        done: FloatArray = cast(FloatArray, total[:rows] / count[:rows])
        keep: int = max(0, min(limit, output_height) - base)
        classes[base : base + keep] = _classify_logits(done[:keep, :output_width])
        total[: window - rows] = total[rows:]
        count[: window - rows] = count[rows:]
        total[window - rows :] = 0
        count[window - rows :] = 0
        base = limit

    for start in range(0, len(positions), batch_size):
        check_cancel(cancel)
        batch_positions: list[tuple[int, int]] = positions[start : start + batch_size]
        parts: FloatArray = np.stack(
            [
                _normalized_tile(reduced[top : top + tile, left : left + tile], statistics)
                for top, left in batch_positions
            ]
        )
        logits: FloatArray = _model_logits(segmenter, parts, batch_size, cancel)
        for (top, left), tile_logits in zip(batch_positions, logits):
            total[top - base : top - base + tile, left : left + tile] += tile_logits
            count[top - base : top - base + tile, left : left + tile] += 1
        completed: int = (start + len(batch_positions)) // len(lefts)
        finish(tops[completed] if completed < len(tops) else padded_height)
    return classes


def build_sky_mask(
    shape: tuple[int, int],
    read_rows: RowReader,
    segmenter: SkySegmenter,
    cancel: Event | None = None,
    block_mean: BlockMean | None = None,
) -> SkyMask:
    """Bounded row reads, robust encoded-channel normalization, overlap-averaged logits."""
    height, width = shape
    factor: int = max(2, round(max(shape) / 4000))
    output_height: int
    output_width: int
    output_height, output_width = height // factor, width // factor
    if min(output_height, output_width) < 1:
        raise ValueError("Image is too small for sky segmentation.")
    reduced: FloatArray = _reduce_encoded_rows(
        output_height, output_width, factor, read_rows, cancel, block_mean
    )
    classes: ByteArray = _infer_overlapping_logits(reduced, segmenter, cancel)
    grid_height: int
    grid_width: int
    grid_height, grid_width = (height + factor - 1) // factor, (width + factor - 1) // factor
    classes = np.pad(
        classes, ((0, grid_height - output_height), (0, grid_width - output_width)), mode="edge"
    )
    check_cancel(cancel)
    return SkyMask(factor, classes, shape)
