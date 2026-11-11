"""Quantized analysis proxies for linear-sRGB working images.

Packaged transfer tables provide deterministic transforms. See docs/assets.md
for their contents and integrity metadata."""

import io
import os
import struct
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import lru_cache
from threading import Event
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms

from ..array_types import BoolArray, ByteArray, FloatArray, IntArray
from ..imaging import check_cancel, linear_to_srgb
from ..jit import njit
from ..sky_mask import SkyMask
from .lookup_assets import ColorTransform, load_analysis_tables
from .numeric import LUMA
from .records import BitsArray, CodesArray, IndexArray

UIntT = TypeVar("UIntT", np.uint8, np.uint16)

READ, WRITE = load_analysis_tables()

COARSE_WORKERS = min(8, os.cpu_count() or 1)
_COLOR_TRANSFORM = ColorTransform()


@lru_cache(maxsize=16)
def transform(thread_id: int | None = None) -> ImageCms.ImageCmsTransform:
    """Build a thread-specific sRGB-to-gamma-2.4 RGBA transform with unchanged alpha."""
    source: ImageCms.ImageCmsProfile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    profile: bytearray = bytearray(source.tobytes())
    count: int = struct.unpack_from(">I", profile, 128)[0]
    for tag_index in range(count):
        tag, offset, size = struct.unpack_from(">4sII", profile, 132 + tag_index * 12)
        if tag in (b"rTRC", b"gTRC", b"bTRC"):
            if profile[offset : offset + 4] != b"para" or size < 16:
                raise ValueError("Unsupported LCMS sRGB profile representation")
            struct.pack_into(">H", profile, offset + 8, 0)
            struct.pack_into(">i", profile, offset + 12, round(2.4 * 65536))
            struct.pack_into(">I", profile, 132 + tag_index * 12 + 8, 16)
    profile[84:100] = bytes(16)
    target: ImageCms.ImageCmsProfile = ImageCms.ImageCmsProfile(io.BytesIO(profile))
    return ImageCms.buildTransformFromOpenProfiles(
        source,
        target,
        "RGBA",
        "RGBA",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        flags=ImageCms.Flags.FORCE_CLUT | ImageCms.Flags.COPY_ALPHA,
    )


@lru_cache(maxsize=1)
def _encode_transfer() -> tuple[BitsArray, CodesArray]:
    """First float32 input bit pattern for each uint16 output transition.

    Build from the original NumPy arithmetic, including ties-to-even. Float
    buckets bound a short binary search; source pixels retain every input bit.
    """
    targets: CodesArray = np.arange(1, 65536, dtype=np.uint16)
    lower: BitsArray = np.zeros(len(targets), np.uint32)
    upper: BitsArray = np.full(len(targets), 0x3F800000, np.uint32)
    for _ in range(30):
        middle: BitsArray = (lower + upper) // 2
        codes: FloatArray = np.rint(np.clip(linear_to_srgb(middle.view(np.float32)), 0, 1) * 65535)
        above: BoolArray = codes >= targets
        upper = np.where(above, middle, upper)
        lower = np.where(above, lower, middle + 1)
    starts: NDArray[np.uint32 | np.int64] = np.arange((0x3F800000 >> 8) + 2, dtype=np.uint32) << 8
    buckets: CodesArray = np.searchsorted(lower, starts, side="right").astype(np.uint16)
    lower.flags.writeable = buckets.flags.writeable = False
    return lower, buckets


@njit(cache=True, nogil=True, inline="always")
def _encoded_value(
    value: int | np.uint32,
    boundaries: BitsArray,
    buckets: ByteArray | CodesArray,
    shift: int = 8,
    maximum: int = 65535,
) -> int | np.int64:
    """Find the quantized code for one original float32 bit pattern using bucketed bounds."""
    if value & 0x80000000 or value > 0x7F800000:
        return 0
    if value >= 0x3F800000:
        return maximum
    bucket: int | np.uint32 = value >> shift
    high: np.int64
    low: np.int64
    low, high = np.int64(buckets[bucket]), np.int64(buckets[bucket + 1])
    while low < high:
        middle: np.int64 = (low + high) // 2
        if value >= boundaries[middle]:
            low = middle + 1
        else:
            high = middle
    return low


@njit(cache=True, nogil=True)
def _encode_bits(
    bits: BitsArray,
    boundaries: BitsArray,
    buckets: NDArray[UIntT],
    result: NDArray[UIntT],
    shift: int = 8,
    maximum: int = 65535,
) -> NDArray[UIntT]:
    """Encode H×W×1/3 float32 bits into a supplied RGB integer array, expanding gray."""
    height, width, channels = bits.shape
    for row in range(height):
        for column in range(width):
            for channel in range(channels):
                result[row, column, channel] = _encoded_value(
                    bits[row, column, channel], boundaries, buckets, shift, maximum
                )
            if channels == 1:
                result[row, column, 1] = result[row, column, 0]
                result[row, column, 2] = result[row, column, 0]
    return result


def encode16(linear: FloatArray) -> CodesArray:
    """Encode H×W×1/3 linear samples into exact ties-to-even RGB uint16 codes."""
    boundaries, buckets = _encode_transfer()
    values: FloatArray = np.asarray(linear, dtype=np.float32)
    result: CodesArray = np.empty((*values.shape[:2], 3), np.uint16)
    return _encode_bits(values.view(np.uint32), boundaries, buckets, result)


def convert_codes(encoded: ByteArray) -> ByteArray:
    """Transform H×W×3 RGB8 samples through the complete coupled RGB color transform."""
    rgba: ByteArray = np.empty((*encoded.shape[:2], 4), np.uint8)
    rgba[:, :, :3] = encoded
    rgba[:, :, 3] = 255
    return np.asarray(
        ImageCms.applyTransform(Image.fromarray(rgba), transform(threading.get_ident()))
    )[:, :, :3].copy()


@lru_cache(maxsize=1)
def _full_transfer() -> tuple[BitsArray, ByteArray]:
    """Return bit transitions/buckets for high bytes of the rounded uint16 encoding."""
    boundaries: BitsArray = _encode_transfer()[0][255::256].copy()
    starts: NDArray[np.uint32 | np.int64] = np.arange((0x3F800000 >> 12) + 2, dtype=np.uint32) << 12
    buckets: ByteArray = np.searchsorted(boundaries, starts, side="right").astype(np.uint8)
    boundaries.flags.writeable = buckets.flags.writeable = False
    return boundaries, buckets


def full_codes(linear: FloatArray) -> ByteArray:
    """Encode linear gray/RGB through uint16 rounding, high-byte selection, and color conversion."""
    boundaries, buckets = _full_transfer()
    values: FloatArray = np.asarray(linear, dtype=np.float32)
    result: ByteArray = np.empty((*values.shape[:2], 3), np.uint8)
    encoded: ByteArray = _encode_bits(values.view(np.uint32), boundaries, buckets, result, 12, 255)
    return convert_codes(encoded)


def decode(codes: ByteArray, denominator: float) -> FloatArray:
    """Read analysis levels, optionally applying the image-wide normalization/12-bit requantization."""
    values: FloatArray = READ[codes]
    if 0 < denominator < 0.98:
        indices: IndexArray = np.rint(np.minimum(values / denominator, 1) * 4095).astype(np.int32)
        values = READ[WRITE[indices]]
    return values


def maximum_luminance(
    codes: ByteArray, cancel: Event | None = None, allowed: BoolArray | None = None
) -> float:
    """Find the global allowed luminance maximum in cancellable RGB8 image strips."""
    peak: float = 0.0
    for y in range(0, len(codes), 128):
        check_cancel(cancel)
        values: FloatArray = READ[codes[y : y + 128]]
        lum: FloatArray = (
            values[:, :, 0] * np.float32(LUMA[0]) + values[:, :, 2] * np.float32(LUMA[2])
        ) + values[:, :, 1] * np.float32(LUMA[1])
        if allowed is not None:
            lum = lum[allowed[y : y + 128]]
        peak = max(peak, float(lum.max(initial=0)))
    return peak


def axis_mapping(source_size: int, destination_size: int) -> tuple[int, IntArray]:
    """Return native encoded-box width and source starts for one downsampled axis."""
    width, step = coarse_step(source_size, destination_size)
    return width, (0.5 + np.arange(destination_size) * step).astype(np.int64)


def coarse_step(source_size: int, destination_size: int) -> tuple[int, float]:
    """Return the encoded-box width and the source distance between coarse samples."""
    width: int = round(float(np.float32(source_size) / np.float32(destination_size)))
    step: float = (
        (source_size - width) / (destination_size - 1) if destination_size > 1 else source_size
    )
    return width, step


@njit(cache=True, nogil=True)
def _coarse_bits(
    bits: BitsArray,
    source_ys: IntArray,
    source_xs: IntArray,
    box_height: int,
    box_width: int,
    allowed: BoolArray,
    boundaries: BitsArray,
    buckets: CodesArray,
) -> ByteArray:
    """Average encoded uint16 box samples before high-byte extraction; an empty mask means all allowed."""
    result: ByteArray = np.empty((len(source_ys), len(source_xs), 3), np.uint8)
    channels: int = bits.shape[2]
    reciprocal: np.float32 = np.float32(1 / (box_height * box_width))
    totals: IntArray = np.empty(3, np.int64)
    for row in range(len(source_ys)):
        for column in range(len(source_xs)):
            totals[0] = totals[1] = totals[2] = 0
            count: int = 0
            for dy in range(box_height):
                y: np.int64 = source_ys[row] + dy
                for dx in range(box_width):
                    x: np.int64 = source_xs[column] + dx
                    if allowed.size and not allowed[y, x]:
                        continue
                    count += 1
                    for channel in range(channels):
                        totals[channel] += _encoded_value(bits[y, x, channel], boundaries, buckets)
            for channel in range(3):
                total: np.float32 = np.float32(totals[0 if channels == 1 else channel])
                if allowed.size:
                    average: np.float32 = (
                        np.float32(total / np.float32(count)) if count else np.float32(0)
                    )
                else:
                    average = np.float32(total * reciprocal)
                result[row, column, channel] = np.uint16(np.rint(average)) >> 8
    return result


def coarse_codes(
    pixels: FloatArray, cancel: Event | None = None, sky_mask: SkyMask | None = None
) -> ByteArray:
    """Encoded box mean, integer ties-to-even, then >>8, BEFORE color conversion."""
    height, width = pixels.shape[:2]
    coarse_height, coarse_width = max(1, height // 4), max(1, width // 4)
    box_height, source_ys = axis_mapping(height, coarse_height)
    box_width, source_xs = axis_mapping(width, coarse_width)
    output: ByteArray = np.empty((coarse_height, coarse_width, 3), np.uint8)
    boundaries, buckets = _encode_transfer()
    bits: BitsArray = np.asarray(pixels, dtype=np.float32).view(np.uint32)
    token: Event = cancel if cancel is not None else threading.Event()

    def process(y: int) -> None:
        """Convert one coarse-row chunk using a thread-local color transform and shared output."""
        check_cancel(token)
        chunk_ys: IntArray = source_ys[y : y + 128]
        top, bottom = int(chunk_ys[0]), int(chunk_ys[-1]) + box_height
        allowed: BoolArray = (
            np.empty((0, 0), np.bool_)
            if sky_mask is None
            else sky_mask.region((0, top, width, bottom))
        )
        encoded: ByteArray = _coarse_bits(
            bits[top:bottom],
            chunk_ys - top,
            source_xs,
            box_height,
            box_width,
            allowed,
            boundaries,
            buckets,
        )
        converted: ByteArray = convert_codes(encoded)
        check_cancel(token)
        output[y : y + 128] = converted

    chunks: range = range(0, coarse_height, 128)
    if len(chunks) < 2 or COARSE_WORKERS == 1:
        for y in chunks:
            process(y)
    else:
        with ThreadPoolExecutor(
            max_workers=COARSE_WORKERS, thread_name_prefix="coarse-proxy"
        ) as pool:
            futures: list[Future[None]] = [pool.submit(process, y) for y in chunks]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                token.set()
                for future in futures:
                    future.cancel()
                raise
    return output


def _full_lcms() -> ByteArray:
    """Load the lossless universal CMM cube once, including concurrent first use."""
    return _COLOR_TRANSFORM.table()


@njit(cache=True, nogil=True)
def _full_proxy_bits(
    bits: BitsArray, boundaries: BitsArray, buckets: ByteArray, cube: ByteArray, levels: FloatArray
) -> FloatArray:
    """Fuse RGB encoding, universal CMM lookup, and normalized analysis levels for one tile."""
    height, width, channels = bits.shape
    result: FloatArray = np.empty((height, width, 3), np.float32)
    for row in range(height):
        for column in range(width):
            red: int | np.int64 = _encoded_value(bits[row, column, 0], boundaries, buckets, 12, 255)
            if channels == 1:
                blue: int | np.int64
                green: int | np.int64
                green, blue = red, red
            else:
                green = _encoded_value(bits[row, column, 1], boundaries, buckets, 12, 255)
                blue = _encoded_value(bits[row, column, 2], boundaries, buckets, 12, 255)
            for channel in range(3):
                result[row, column, channel] = levels[cube[red, green, blue, channel]]
    return result


def full_proxy(linear: FloatArray, denominator: float) -> FloatArray:
    """Exact encode16>>8, full RGB CMM, READ and global normalization in one pass."""
    boundaries, buckets = _full_transfer()
    # Evaluating all 256 possible CMM outputs with the original NumPy arithmetic
    # preserves its float32 division/multiplication and ties-to-even exactly.
    levels: FloatArray = decode(np.arange(256, dtype=np.uint8), denominator)
    values: FloatArray = np.asarray(linear, dtype=np.float32)
    return _full_proxy_bits(values.view(np.uint32), boundaries, buckets, _full_lcms(), levels)
