"""Validated TIFF/ICC boundaries and atomic, original-protecting image writes."""

from __future__ import annotations

import io
import os
import struct
from functools import lru_cache
from pathlib import Path
from threading import Event
from typing import cast

import numpy as np
import tifffile
from numpy.typing import NDArray
from PIL import ImageCms
from PIL.ImageCms import ImageCmsProfile
from tifffile import TiffPage, TiffTag

from ..domain.array_types import DoubleArray, FloatArray
from ..domain.imaging import ImageFrame, Pixels, check_cancel, linear_to_srgb, srgb_to_linear
from ..domain.jit import njit

type EncodedPixels = NDArray[np.uint8] | NDArray[np.uint16]
type ProfileCharacteristics = tuple[bytes, dict[bytes, bytes]]

ROWS_PER_BLOCK = 128


def _profile_characteristics(profile: bytes) -> tuple[bytes, dict[bytes, bytes]]:
    """Extract bounded ICC matrix/shaper tags, ignoring descriptive metadata.

    ICC header/table layout: https://www.color.org/icc34.pdf, clause 6.2.
    """
    if len(profile) < 132 or struct.unpack_from(">I", profile)[0] != len(profile):
        raise ValueError("Invalid ICC size.")
    if profile[36:40] != b"acsp" or profile[8] not in (2, 4):
        raise ValueError("Unsupported ICC header.")
    count: int = struct.unpack_from(">I", profile, 128)[0]
    table_end: int = 132 + count * 12
    if table_end > len(profile):
        raise ValueError("Invalid ICC tag table.")
    tags: dict[bytes, bytes] = {}
    for index in range(count):
        signature, offset, size = struct.unpack_from(">4sII", profile, 132 + index * 12)
        if offset < table_end or offset + size > len(profile) or signature in tags:
            raise ValueError("Invalid ICC tag bounds.")
        tags[signature] = profile[offset : offset + size]
    for metadata in (
        b"desc",
        b"cprt",
        b"dmnd",
        b"dmdd",
        b"vued",
        b"view",
        b"lumi",
        b"meas",
        b"tech",
    ):
        tags.pop(metadata, None)
    # Version, class, RGB/PCS signatures and PCS illuminant affect interpretation.
    return profile[8:24] + profile[68:80], tags


def _matches_fixed_tag(value: bytes, expected: bytes) -> bool:
    """Compare fixed-point ICC tag values within the supported matrix tolerance."""
    if len(value) != len(expected) or value[:8] != expected[:8]:
        return False
    actual_values: DoubleArray = np.frombuffer(value[8:], dtype=">i4").astype(np.float64) / 65536
    expected_values: DoubleArray = (
        np.frombuffer(expected[8:], dtype=">i4").astype(np.float64) / 65536
    )
    return bool(np.allclose(actual_values, expected_values, atol=0.0005, rtol=0))


@lru_cache(maxsize=16)
def _matches_srgb_curve(value: bytes, expected: bytes) -> bool:
    """Accept the reference curve or a numerically equivalent monotone sampled curve."""
    if value == expected:
        return True
    # Accept sampled v2 curveType, alongside the reference's parametric curve.
    # Compare its piecewise-linear evaluation at EVERY supported 16-bit input,
    # within one 16-bit linear-light output code. No names or profile hashes.
    if len(value) < 12 or value[:8] != b"curv\0\0\0\0":
        return False
    count: int = struct.unpack_from(">I", value, 8)[0]
    if not 2 <= count <= 65536 or len(value) != 12 + count * 2:
        return False
    curve: DoubleArray = np.frombuffer(value[12:], dtype=">u2").astype(np.float64) / 65535
    if np.any(np.diff(curve) < 0):
        return False
    codes: DoubleArray = np.arange(65536, dtype=np.float64) / 65535
    decoded: DoubleArray = np.interp(codes, np.linspace(0, 1, count), curve)
    srgb: DoubleArray = np.where(codes <= 0.04045, codes / 12.92, ((codes + 0.055) / 1.055) ** 2.4)
    return bool(np.max(np.abs(decoded - srgb)) <= 1 / 65535)


def _matches_optional_tags(tags: dict[bytes, bytes], reference_tags: dict[bytes, bytes]) -> bool:
    """Validate optional adaptation/chromaticity and a zero black point."""
    for signature in (b"chad", b"chrm"):
        if signature in tags:
            if signature == b"chrm":
                if tags[signature] != reference_tags[signature]:
                    return False
            elif not _matches_fixed_tag(tags[signature], reference_tags[signature]):
                return False
    if b"bkpt" in tags and tags[b"bkpt"] != b"XYZ \0\0\0\0" + bytes(12):
        return False
    return True


def _matches_white_point(
    header: bytes, tags: dict[bytes, bytes], reference_tags: dict[bytes, bytes]
) -> bool:
    """Accept reference white or legacy v2 D65 without an adaptation tag."""
    if _matches_fixed_tag(tags[b"wtpt"], reference_tags[b"wtpt"]):
        return True
    # Legacy v2 sRGB uses D65 wtpt with D50-adapted colorants and no chad.
    # https://registry.color.org/rgb-registry/srgbprofiles
    d65: bytes = b"XYZ \0\0\0\0" + struct.pack(">iii", 62289, 65536, 71372)
    return header[0] == 2 and b"chad" not in tags and _matches_fixed_tag(tags[b"wtpt"], d65)


def _matches_srgb(
    characteristics: ProfileCharacteristics, reference: ProfileCharacteristics
) -> bool:
    """Check matrix/shaper compatibility without trusting descriptive profile names."""
    header, tags = characteristics
    reference_header, reference_tags = reference
    if header[4:] != reference_header[4:]:
        return False
    required: set[bytes] = {b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC", b"gTRC", b"bTRC", b"wtpt"}
    if not required <= tags.keys() or tags.keys() - required - {b"chad", b"chrm", b"bkpt"}:
        return False  # Never ignore LUT or unknown transform tags.
    for channel in (b"r", b"g", b"b"):
        if not _matches_fixed_tag(tags[channel + b"XYZ"], reference_tags[channel + b"XYZ"]):
            return False
        if not _matches_srgb_curve(tags[channel + b"TRC"], reference_tags[channel + b"TRC"]):
            return False
    return _matches_optional_tags(tags, reference_tags) and _matches_white_point(
        header, tags, reference_tags
    )


def _check_profile(profile: bytes | None, rgb: bool) -> str:
    """Validate color interpretation and return the import note or raise ValueError."""
    if not profile:
        return (
            "Untagged image: sRGB assumed." if rgb else "Untagged grayscale: sRGB tone curve assumed."
        )
    try:
        parsed: ImageCmsProfile = ImageCms.ImageCmsProfile(io.BytesIO(profile))
        name: str = ImageCms.getProfileName(parsed).strip()
        characteristics: tuple[bytes, dict[bytes, bytes]] = _profile_characteristics(profile)
    except Exception as exc:
        raise ValueError("Invalid ICC profile. Convert the image to sRGB in your editor.") from exc
    reference: bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    reference_characteristics: tuple[bytes, dict[bytes, bytes]] = _profile_characteristics(
        reference
    )
    if not rgb or not _matches_srgb(characteristics, reference_characteristics):
        raise ValueError(
            f"Profile '{name}' is not supported by this prototype's strict ICC check. "
            "Convert to sRGB in your editor, then export an untagged RGB TIFF. "
            "Removing a profile alone does not convert colors."
        )
    suffix: str = (
        " (validated sRGB matrix/TRC variant)"
        if characteristics != reference_characteristics
        else ""
    )
    return f"Embedded sRGB profile: {name}{suffix}"


def _validated_page(tiff_file: tifffile.TiffFile) -> tuple[tifffile.TiffPage, str]:
    """Accept one supported page and describe its validated color interpretation."""
    if len(tiff_file.pages) != 1:
        raise ValueError("Use a single-page, flattened TIFF (no image stacks).")
    page: TiffPage = cast(tifffile.TiffPage, tiff_file.pages[0])
    orientation: TiffTag | None = page.tags.get(274)
    if orientation and orientation.value != 1:
        raise ValueError(
            "Unsupported TIFF orientation. Normalize rotation/mirroring in your editor."
        )
    # 12-bit or 4-bit samples sit in 16/8-bit containers and would load far too dark.
    if page.dtype not in (np.dtype("uint8"), np.dtype("uint16")) or page.bitspersample not in (
        8,
        16,
    ):
        raise ValueError("Supported input: unsigned 8/16-bit TIFF. HDR is a later milestone.")
    if page.imagedepth > 1:
        raise ValueError("Use a single-page, flattened TIFF (no image stacks).")
    if page.photometric not in (1, 2) or page.samplesperpixel not in (1, 3) or page.extrasamples:
        raise ValueError("Use RGB or grayscale TIFF without alpha, palette or CMYK.")
    if min(page.imagelength, page.imagewidth) < 2:
        raise ValueError("Expected an image of at least 2 × 2 pixels.")
    tag: TiffTag | None = page.tags.get(34675)
    return page, _check_profile(tag.value if tag else None, page.photometric == 2)


def _mapped_pixels(page: tifffile.TiffPage) -> EncodedPixels:
    """Expose validated unsigned samples as H×W×C, mapping/decoding only as needed."""
    # Uncompressed contiguous TIFFs map the source file directly. Everything else decodes
    # in RAM: tifffile's "memmap" fallback would write the photo to a temporary file.
    data: EncodedPixels = page.asarray(out="memmap" if page.is_memmappable else None)
    if page.planarconfig == 2 and data.ndim == 3:
        data = np.moveaxis(data, 0, -1)
    if data.ndim == 2:
        data = data[..., None]
    return data


@lru_cache(maxsize=4)
def _integer_transfer(maximum: int) -> Pixels:
    """Build an immutable sRGB-decoding table for uint8/uint16 input codes.

    Grayscale uses the same tone curve as RGB, like most editors treat untagged gray,
    so effects blend in linear light for both.
    """
    codes: Pixels | DoubleArray = np.arange(maximum + 1, dtype=np.float32) / maximum
    values: Pixels | DoubleArray = srgb_to_linear(codes)
    values.flags.writeable = False
    # Weak-scalar division preserves float32; NumPy stubs conservatively include float64.
    return cast(Pixels, values)


@njit(cache=True, nogil=True)
def _transfer_integer_block(source: EncodedPixels, table: Pixels, destination: Pixels) -> None:
    """Convert one H×W×C integer strip into its owned linear-float destination."""
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            for channel in range(source.shape[2]):
                destination[y, x, channel] = table[source[y, x, channel]]


def _convert_integer_image(
    data: EncodedPixels, path: Path, note: str, cancel: Event | None
) -> ImageFrame:
    """Convert validated samples in cancellable strips, then transfer buffer ownership."""
    normalized: FloatArray = np.empty(data.shape, np.float32)
    table: FloatArray = _integer_transfer(np.iinfo(data.dtype).max)
    for row in range(0, data.shape[0], ROWS_PER_BLOCK):
        check_cancel(cancel)
        source: EncodedPixels = np.asarray(
            data[row : row + ROWS_PER_BLOCK], dtype=data.dtype.newbyteorder("=")
        )
        _transfer_integer_block(source, table, normalized[row : row + ROWS_PER_BLOCK])
    check_cancel(cancel)
    return ImageFrame._from_owned_validated(
        normalized,
        path.name,
        "linear-srgb",
        note,
    )


def read_tiff(path: Path | str, cancel: Event | None = None) -> ImageFrame:
    """Read one TIFF into immutable working pixels, validating format and ICC first."""
    check_cancel(cancel)
    path = Path(path)
    with tifffile.TiffFile(path) as tiff_file:
        page, note = _validated_page(tiff_file)
        data: EncodedPixels = _mapped_pixels(page)
        return _convert_integer_image(data, path, note, cancel)


def read_tiff_preview(
    path: Path | str, cancel: Event | None = None, max_edge: int = 1600
) -> tuple[ImageFrame, float, int, int]:
    """Return a quick strided preview, scale, original width and original height.

    Compatible uncompressed files are sampled directly from their map before
    float conversion. Compressed TIFFs are decoded in memory first.
    This is an opening preview, not an antialiased export or detection input.
    """
    check_cancel(cancel)
    if max_edge < 2:
        raise ValueError("Preview edge must be at least two pixels.")
    path = Path(path)
    with tifffile.TiffFile(path) as tiff_file:
        page, note = _validated_page(tiff_file)
        width, height = page.imagewidth, page.imagelength
        stride: int = max(1, (max(width, height) + max_edge - 1) // max_edge)
        data: EncodedPixels = _mapped_pixels(page)[::stride, ::stride]
        # Keep ImageFrame's 2×2 minimum for exceptionally thin panoramas.
        if min(data.shape[:2]) < 2:
            data = cast(EncodedPixels, np.repeat(data, 2, axis=0 if data.shape[0] == 1 else 1))
        return _convert_integer_image(data, path, note, cancel), 1 / stride, width, height


def write_tiff(
    path: Path | str,
    frame: ImageFrame,
    pixels: Pixels,
    source_path: Path | str | None = None,
    cancel: Event | None = None,
) -> None:
    """Write atomically. A cancellation before replacement preserves the existing destination."""
    path = Path(path)
    if source_path is not None:
        source: Path = Path(source_path)
        if path.resolve() == source.resolve() or (
            path.exists() and source.exists() and path.samefile(source)
        ):
            raise ValueError("Choose a new output file; the original image is protected.")
    if pixels.shape != frame.pixels.shape:
        raise ValueError("Invalid output pixels.")
    check_cancel(cancel)
    rgb: bool = pixels.shape[2] == 3
    output: NDArray[np.uint16] = np.empty(pixels.shape, np.uint16)
    for row in range(0, pixels.shape[0], ROWS_PER_BLOCK):
        check_cancel(cancel)
        block: FloatArray = pixels[row : row + ROWS_PER_BLOCK]
        # Checked per block so a huge export needs no extra full-image boolean buffer.
        if not np.isfinite(block).all() or block.min() < 0:
            raise ValueError("Invalid output pixels.")
        encoded: FloatArray = linear_to_srgb(block)
        output[row : row + ROWS_PER_BLOCK] = np.rint(np.clip(encoded, 0, 1) * 65535).astype(
            np.uint16
        )
    profile: bytes | None = (
        ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes() if rgb else None
    )
    tags: list[tuple[int, str, int, bytes, bool]] = (
        [(34675, "B", len(profile), profile, False)] if profile else []
    )
    # Unlike mkstemp (0600), touch honours the umask so exports stay readable like other files.
    temporary: Path = path.parent / f".{path.stem}-{os.urandom(8).hex()}.tif"
    temporary.touch(exist_ok=False)
    try:
        tifffile.imwrite(
            temporary,
            output if rgb else output[..., 0],
            photometric="rgb" if rgb else "minisblack",
            metadata=None,
            extratags=tags,
        )
        with temporary.open("rb+") as written:
            os.fsync(written.fileno())  # a power cut must not leave a truncated export
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o7777)  # replacing keeps its permissions
        check_cancel(cancel)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_samples(path: Path | str, samples: NDArray[np.integer], compress: bool) -> None:
    """Write already-encoded 8/16-bit samples as an untagged TIFF (used by the warm-up)."""
    gray: bool = samples.shape[2] == 1
    tifffile.imwrite(
        path,
        samples[..., 0] if gray else samples,
        photometric="minisblack" if gray else "rgb",
        compression="zlib" if compress else None,
        metadata=None,
    )
