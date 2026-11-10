"""Quantized analysis proxies for linear-sRGB working images.

Packaged transfer tables provide deterministic transforms. See docs/assets.md
for their contents and integrity metadata."""

import hashlib
import io
import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image, ImageCms

from ..imaging import check_cancel, linear_to_srgb
from .numeric import LUMA

with np.load(Path(__file__).with_name("analysis_luts.npz")) as tables:
    READ = tables["read"].copy()
    WRITE = tables["write"].copy()

COARSE_WORKERS = min(8, os.cpu_count() or 1)
_LCMS_TABLE = None
_LCMS_LOCK = threading.Lock()


@lru_cache(maxsize=16)
def transform(thread_id=None):
    source = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    profile = bytearray(source.tobytes())
    count = struct.unpack_from(">I", profile, 128)[0]
    for i in range(count):
        tag, offset, size = struct.unpack_from(">4sII", profile, 132 + i * 12)
        if tag in (b"rTRC", b"gTRC", b"bTRC"):
            if profile[offset : offset + 4] != b"para" or size < 16:
                raise ValueError("Unsupported LCMS sRGB profile representation")
            struct.pack_into(">H", profile, offset + 8, 0)
            struct.pack_into(">i", profile, offset + 12, round(2.4 * 65536))
            struct.pack_into(">I", profile, 132 + i * 12 + 8, 16)
    profile[84:100] = bytes(16)
    target = ImageCms.ImageCmsProfile(io.BytesIO(profile))
    return ImageCms.buildTransformFromOpenProfiles(
        source,
        target,
        "RGBA",
        "RGBA",
        renderingIntent=1,
        flags=ImageCms.Flags.FORCE_CLUT | ImageCms.Flags.COPY_ALPHA,
    )


@lru_cache(maxsize=1)
def _encode_transfer():
    """First float32 input bit pattern for each uint16 output transition.

    Build from the original NumPy arithmetic, including ties-to-even. Float
    buckets bound a short binary search; source pixels retain every input bit.
    """
    targets = np.arange(1, 65536, dtype=np.uint16)
    lower = np.zeros(len(targets), np.uint32)
    upper = np.full(len(targets), 0x3F800000, np.uint32)
    for _ in range(30):
        middle = (lower + upper) // 2
        codes = np.rint(np.clip(linear_to_srgb(middle.view(np.float32)), 0, 1) * 65535)
        above = codes >= targets
        upper = np.where(above, middle, upper)
        lower = np.where(above, lower, middle + 1)
    starts = np.arange((0x3F800000 >> 8) + 2, dtype=np.uint32) << 8
    buckets = np.searchsorted(lower, starts, side="right").astype(np.uint16)
    lower.flags.writeable = buckets.flags.writeable = False
    return lower, buckets


@njit(cache=True, nogil=True, inline="always")
def _encoded_value(value, boundaries, buckets, shift=8, maximum=65535):
    if value & 0x80000000 or value > 0x7F800000:
        return 0
    if value >= 0x3F800000:
        return maximum
    bucket = value >> shift
    low, high = np.int64(buckets[bucket]), np.int64(buckets[bucket + 1])
    while low < high:
        middle = (low + high) // 2
        if value >= boundaries[middle]:
            low = middle + 1
        else:
            high = middle
    return low


@njit(cache=True, nogil=True)
def _encode_bits(bits, boundaries, buckets, result, shift=8, maximum=65535):
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


def encode16(linear):
    boundaries, buckets = _encode_transfer()
    values = np.asarray(linear, dtype=np.float32)
    result = np.empty((*values.shape[:2], 3), np.uint16)
    return _encode_bits(values.view(np.uint32), boundaries, buckets, result)


def convert_codes(encoded):
    rgba = np.empty((*encoded.shape[:2], 4), np.uint8)
    rgba[:, :, :3] = encoded
    rgba[:, :, 3] = 255
    return np.asarray(
        ImageCms.applyTransform(Image.fromarray(rgba), transform(threading.get_ident()))
    )[:, :, :3].copy()


@lru_cache(maxsize=1)
def _full_transfer():
    # rint16(x)>>8 changes only at uint16 codes 256,512,...,65280.
    # This deliberately differs from rounding the gamma result to 8 bits.
    boundaries = _encode_transfer()[0][255::256].copy()
    starts = np.arange((0x3F800000 >> 12) + 2, dtype=np.uint32) << 12
    buckets = np.searchsorted(boundaries, starts, side="right").astype(np.uint8)
    boundaries.flags.writeable = buckets.flags.writeable = False
    return boundaries, buckets


def full_codes(linear):
    boundaries, buckets = _full_transfer()
    values = np.asarray(linear, dtype=np.float32)
    result = np.empty((*values.shape[:2], 3), np.uint8)
    encoded = _encode_bits(values.view(np.uint32), boundaries, buckets, result, 12, 255)
    return convert_codes(encoded)


def decode(codes, denominator):
    values = READ[codes]
    if 0 < denominator < 0.98:
        indices = np.rint(np.minimum(values / denominator, 1) * 4095).astype(np.int32)
        values = READ[WRITE[indices]]
    return values


def maximum_luminance(codes, cancel=None, allowed=None):
    peak = 0.0
    for y in range(0, len(codes), 128):
        check_cancel(cancel)
        values = READ[codes[y : y + 128]]
        lum = (
            values[:, :, 0] * np.float32(LUMA[0]) + values[:, :, 2] * np.float32(LUMA[2])
        ) + values[:, :, 1] * np.float32(LUMA[1])
        if allowed is not None:
            lum = lum[allowed[y : y + 128]]
        peak = max(peak, float(lum.max(initial=0)))
    return peak


def axis_mapping(source_size, destination_size):
    width = round(float(np.float32(source_size) / np.float32(destination_size)))
    step = (source_size - width) / (destination_size - 1) if destination_size > 1 else source_size
    return width, (0.5 + np.arange(destination_size) * step).astype(np.int64)


@njit(cache=True, nogil=True)
def _coarse_bits(bits, ys, xs, bh, bw, allowed, boundaries, buckets):
    result = np.empty((len(ys), len(xs), 3), np.uint8)
    channels = bits.shape[2]
    reciprocal = np.float32(1 / (bh * bw))
    totals = np.empty(3, np.int64)
    for row in range(len(ys)):
        for column in range(len(xs)):
            totals[0] = totals[1] = totals[2] = 0
            count = 0
            for dy in range(bh):
                y = ys[row] + dy
                for dx in range(bw):
                    x = xs[column] + dx
                    if allowed.size and not allowed[y, x]:
                        continue
                    count += 1
                    for channel in range(channels):
                        totals[channel] += _encoded_value(bits[y, x, channel], boundaries, buckets)
            for channel in range(3):
                total = np.float32(totals[0 if channels == 1 else channel])
                if allowed.size:
                    average = np.float32(total / np.float32(count)) if count else np.float32(0)
                else:
                    average = np.float32(total * reciprocal)
                result[row, column, channel] = np.uint16(np.rint(average)) >> 8
    return result


def coarse_codes(pixels, cancel=None, sky_mask=None):
    """Encoded box mean, integer ties-to-even, then >>8, BEFORE color conversion."""
    height, width = pixels.shape[:2]
    dh, dw = max(1, height // 4), max(1, width // 4)
    bh, ys = axis_mapping(height, dh)
    bw, xs = axis_mapping(width, dw)
    output = np.empty((dh, dw, 3), np.uint8)
    boundaries, buckets = _encode_transfer()
    bits = np.asarray(pixels, dtype=np.float32).view(np.uint32)
    token = cancel if cancel is not None else threading.Event()

    def process(y):
        check_cancel(token)
        chunk_ys = ys[y : y + 128]
        top, bottom = int(chunk_ys[0]), int(chunk_ys[-1]) + bh
        allowed = (
            np.empty((0, 0), np.bool_)
            if sky_mask is None
            else sky_mask.region((0, top, width, bottom))
        )
        encoded = _coarse_bits(
            bits[top:bottom], chunk_ys - top, xs, bh, bw, allowed, boundaries, buckets
        )
        converted = convert_codes(encoded)
        check_cancel(token)
        output[y : y + 128] = converted

    chunks = range(0, dh, 128)
    if len(chunks) < 2 or COARSE_WORKERS == 1:
        for y in chunks:
            process(y)
    else:
        with ThreadPoolExecutor(
            max_workers=COARSE_WORKERS, thread_name_prefix="coarse-proxy"
        ) as pool:
            futures = [pool.submit(process, y) for y in chunks]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                token.set()
                for future in futures:
                    future.cancel()
                raise
    return output


def _full_lcms():
    """Load the lossless universal CMM cube once, including concurrent first use."""
    global _LCMS_TABLE
    with _LCMS_LOCK:
        if _LCMS_TABLE is None:
            path = Path(__file__).with_name("full_lcms_lut.npz")
            metadata = json.loads(path.with_suffix(".json").read_text())
            with np.load(path, allow_pickle=False) as stored:
                cube = stored["delta"]
                np.add(cube[..., 0], stored["red"][:, None, None], out=cube[..., 0])
                np.add(cube[..., 1], stored["green"][None, :, None], out=cube[..., 1])
                np.add(cube[..., 2], stored["blue"][None, None, :], out=cube[..., 2])
            if hashlib.sha256(memoryview(cube)).hexdigest() != metadata["table_sha256"]:
                raise ValueError("Universal LCMS cube does not match its recorded identity.")
            cube.flags.writeable = False
            _LCMS_TABLE = cube
        return _LCMS_TABLE


@njit(cache=True, nogil=True)
def _full_proxy_bits(bits, boundaries, buckets, cube, levels):
    height, width, channels = bits.shape
    result = np.empty((height, width, 3), np.float32)
    for row in range(height):
        for column in range(width):
            red = _encoded_value(bits[row, column, 0], boundaries, buckets, 12, 255)
            if channels == 1:
                green, blue = red, red
            else:
                green = _encoded_value(bits[row, column, 1], boundaries, buckets, 12, 255)
                blue = _encoded_value(bits[row, column, 2], boundaries, buckets, 12, 255)
            for channel in range(3):
                result[row, column, channel] = levels[cube[red, green, blue, channel]]
    return result


def full_proxy(linear, denominator):
    """Exact encode16>>8, full RGB CMM, READ and global normalization in one pass."""
    boundaries, buckets = _full_transfer()
    # Evaluating all 256 possible CMM outputs with the original NumPy arithmetic
    # preserves its float32 division/multiplication and ties-to-even exactly.
    levels = decode(np.arange(256, dtype=np.uint8), denominator)
    values = np.asarray(linear, dtype=np.float32)
    return _full_proxy_bits(values.view(np.uint32), boundaries, buckets, _full_lcms(), levels)
