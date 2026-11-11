"""Validated TIFF/ICC boundaries and atomic, original-protecting image writes."""

from __future__ import annotations

import io
import os
import struct
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
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
            "Untagged image: sRGB assumed."
            if rgb
            else "Untagged grayscale: sRGB tone curve assumed."
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


class EncodedRows:
    """Read encoded unsigned samples as H×W×C rows without holding a decoded file.

    Uncompressed contiguous pages are memory-mapped. Every other page decodes, per
    request, only the strips or tiles that intersect the requested rows (in parallel,
    never writing to disk) and keeps just the last partially used strip row, so
    sequential bands decode each segment once. A band is released after use. A
    TIFF stored as ONE compressed strip (or planar with one strip per plane) must
    be decoded whole to read any row, so it needs about one decoded image of RAM.
    """

    def __init__(self, page: TiffPage) -> None:
        """Describe the segment grid of a validated page."""
        self.page = page
        self.height, self.width = page.imagelength, page.imagewidth
        self.channels: int = page.samplesperpixel
        self.dtype = cast(
            np.dtype[np.uint8] | np.dtype[np.uint16], np.dtype(page.dtype).newbyteorder("=")
        )
        self.mapped: EncodedPixels | None = None
        if page.is_memmappable:
            # Mapping the source never writes: tifffile's "memmap" fallback for other pages would.
            mapped: EncodedPixels = page.asarray(out="memmap")
            if page.planarconfig == 2 and mapped.ndim == 3:
                mapped = np.moveaxis(mapped, 0, -1)
            self.mapped = mapped.reshape(self.height, self.width, self.channels)
        self.planar: bool = page.planarconfig == 2 and self.channels > 1
        # Segment height/width: tile, or strip rows (clipped, a lone strip is rowsperstrip=2**32-1).
        self.seg_rows: int = page.tile[0] if page.tile else min(page.rowsperstrip, self.height)
        self.seg_cols: int = page.tile[1] if page.tile else self.width
        self.grid_rows: int = -(-self.height // self.seg_rows)
        self.grid_cols: int = -(-self.width // self.seg_cols)
        self.kept: tuple[int, EncodedPixels] | None = None  # (strip-row index, its decoded rows)
        self.pool: ThreadPoolExecutor | None = None

    def __enter__(self) -> EncodedRows:
        """Use as a context manager so the decode threads end with the reader."""
        return self

    def __exit__(self, *_: object) -> None:
        """Stop the decode threads, if any were started."""
        self.close()

    def close(self) -> None:
        """Shut the reusable decode executor down; later reads start a new one."""
        pool, self.pool = self.pool, None
        if pool is not None:
            pool.shutdown()

    def _place(
        self, out: EncodedPixels, top: int, row0: int, col0: int, plane: int, data: EncodedPixels
    ) -> None:
        """Copy the part of one decoded segment that falls inside the output rows."""
        lo, hi = max(top, row0), min(top + out.shape[0], row0 + data.shape[1], self.height)
        width: int = min(data.shape[2], self.width - col0)
        if lo < hi:
            part: EncodedPixels = data[0, lo - row0 : hi - row0, :width]
            if self.planar:
                out[lo - top : hi - top, col0 : col0 + width, plane] = part[..., 0]
            else:
                out[lo - top : hi - top, col0 : col0 + width] = part

    def _decode(self, indices: list[int]) -> Iterator[tuple[int, EncodedPixels | None]]:
        """Decode the given segments a few at a time, yielding each with its index."""
        page, handle = self.page, self.page.parent.filehandle
        workers: int = max(1, page.maxworkers)
        args: dict[str, object] = {"_fullsize": page.is_tiled}
        if page.compression in {6, 7, 34892, 33007}:  # JPEG
            args["jpegtables"] = page.jpegtables
            args["jpegheader"] = page.jpegheader

        def decode(item: tuple[bytes | None, int]) -> tuple[int, EncodedPixels | None]:
            result = page.decode(*item, **args)
            return item[1], result[0]

        if self.pool is None:
            self.pool = ThreadPoolExecutor(workers)
        for start in range(0, len(indices), 2 * workers):
            items: list[tuple[bytes | None, int]] = []
            with handle.lock:
                for index in indices[start : start + 2 * workers]:
                    count: int = page.databytecounts[index]
                    handle.seek(page.dataoffsets[index])
                    items.append((handle.read(count) if count else None, index))
            yield from self.pool.map(decode, items)

    def rows(self, top: int, bottom: int, step: int = 1) -> EncodedPixels:
        """Return rows top, top+step, ... below `bottom`; only those are read or decoded."""
        if self.mapped is not None:
            return np.asarray(self.mapped[top:bottom:step], dtype=self.dtype)
        out: EncodedPixels = np.zeros((bottom - top, self.width, self.channels), self.dtype)
        kept, self.kept = self.kept, None
        last: int = (bottom - 1) // self.seg_rows
        partial: bool = (last + 1) * self.seg_rows > bottom
        wanted: list[int] = []
        for band in range(top // self.seg_rows, last + 1):
            lo, hi = band * self.seg_rows, (band + 1) * self.seg_rows
            if -(-(lo - top) // step) * step + top >= hi:
                continue  # no sampled row inside this strip row
            if kept is not None and kept[0] == band:
                start, stop = max(top, lo), min(bottom, hi)
                out[start - top : stop - top] = kept[1][start - lo : stop - lo]
                self.kept = kept if band == last and partial else None
                continue
            for plane in range(self.channels if self.planar else 1):
                first: int = (plane * self.grid_rows + band) * self.grid_cols
                wanted += range(first, first + self.grid_cols)
        retained: EncodedPixels | None = None
        for index, data in self._decode(wanted):
            if data is None:
                continue
            plane, rest = divmod(index, self.grid_rows * self.grid_cols)
            band, col = divmod(rest, self.grid_cols)
            self._place(out, top, band * self.seg_rows, col * self.seg_cols, plane, data)
            if band == last and partial:
                retained = self._extend(retained, plane, col, data)
        if retained is not None:
            self.kept = (last, retained)
        return out[::step]

    def _extend(
        self, band: EncodedPixels | None, plane: int, col: int, data: EncodedPixels
    ) -> EncodedPixels:
        """Add a decoded segment to the retained strip row (a plain view if it spans the row)."""
        if band is None and not self.planar and self.grid_cols == 1:
            return np.asarray(data[0, :, : self.width], dtype=self.dtype)
        if band is None:
            band = np.zeros((data.shape[1], self.width, self.channels), self.dtype)
        left: int = col * self.seg_cols
        width: int = min(data.shape[2], self.width - left)
        rows: int = min(data.shape[1], band.shape[0])
        if self.planar:
            band[:rows, left : left + width, plane] = data[0, :rows, :width, 0]
        else:
            band[:rows, left : left + width] = data[0, :rows, :width]
        return band


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
    read_rows: Callable[[int, int], EncodedPixels],
    shape: tuple[int, int, int],
    maximum: int,
    path: Path,
    note: str,
    cancel: Event | None,
    block: int = ROWS_PER_BLOCK,
) -> ImageFrame:
    """Convert encoded row bands in cancellable strips, then transfer buffer ownership."""
    normalized: FloatArray = np.empty(shape, np.float32)
    table: FloatArray = _integer_transfer(maximum)
    for row in range(0, shape[0], block):
        check_cancel(cancel)
        source: EncodedPixels = read_rows(row, min(row + block, shape[0]))
        _transfer_integer_block(source, table, normalized[row : row + block])
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
        with EncodedRows(page) as reader:
            shape: tuple[int, int, int] = (reader.height, reader.width, reader.channels)
            return _convert_integer_image(
                reader.rows, shape, int(np.iinfo(reader.dtype).max), path, note, cancel
            )


def _slice_rows(array: EncodedPixels) -> Callable[[int, int], EncodedPixels]:
    """Serve row bands of an in-memory array through the reader interface."""
    return lambda top, bottom: array[top:bottom]


def read_tiff_preview(
    path: Path | str, cancel: Event | None = None, max_edge: int = 1600
) -> tuple[ImageFrame, float, int, int]:
    """Return a quick strided preview, scale, original width and original height.

    Rows are read band by band (see EncodedRows): uncompressed files are sampled
    from their map; compressed ones decode only strips/tiles holding a sampled row
    and keep none of them. This is an opening preview, not an antialiased export
    or detection input.
    """
    check_cancel(cancel)
    if max_edge < 2:
        raise ValueError("Preview edge must be at least two pixels.")
    path = Path(path)
    with tifffile.TiffFile(path) as tiff_file:
        page, note = _validated_page(tiff_file)
        width, height = page.imagewidth, page.imagelength
        stride: int = max(1, (max(width, height) + max_edge - 1) // max_edge)
        with EncodedRows(page) as reader:
            shape: tuple[int, int, int] = (
                -(-height // stride),
                -(-width // stride),
                reader.channels,
            )

            def sampled(top: int, bottom: int) -> EncodedPixels:
                """Return strided output rows [top, bottom) of the preview grid."""
                rows: EncodedPixels = reader.rows(
                    top * stride, min((bottom - 1) * stride + 1, height), stride
                )
                return rows[:, ::stride]

            block: int = max(1, ROWS_PER_BLOCK // stride)
            # Keep ImageFrame's 2×2 minimum for exceptionally thin panoramas.
            if min(shape[:2]) < 2:
                thin: EncodedPixels = cast(
                    EncodedPixels,
                    np.concatenate(
                        [
                            sampled(row, min(row + block, shape[0]))
                            for row in range(0, shape[0], block)
                        ]
                    ),
                )
                thin = cast(EncodedPixels, np.repeat(thin, 2, axis=0 if thin.shape[0] == 1 else 1))
                shape = (thin.shape[0], thin.shape[1], shape[2])
                read_rows = _slice_rows(thin)
            else:
                read_rows = sampled
            frame: ImageFrame = _convert_integer_image(
                read_rows, shape, int(np.iinfo(reader.dtype).max), path, note, cancel, block
            )
            return frame, 1 / stride, width, height


def _encoded_rows(
    pixels: Pixels | Iterable[Pixels], shape: tuple[int, ...], cancel: Event | None
) -> Iterator[NDArray[np.uint16]]:
    """Validate and encode float bands, yielding one uint16 row at a time."""
    bands: Iterable[Pixels] = (
        (pixels[row : row + ROWS_PER_BLOCK] for row in range(0, shape[0], ROWS_PER_BLOCK))
        if isinstance(pixels, np.ndarray)
        else pixels
    )
    written: int = 0
    for band in bands:
        check_cancel(cancel)
        if band.ndim != 3 or band.shape[1:] != shape[1:] or written + band.shape[0] > shape[0]:
            raise ValueError("Invalid output pixels.")
        # Checked per band so a huge export needs no extra full-image boolean buffer.
        if not np.isfinite(band).all() or band.min() < 0:
            raise ValueError("Invalid output pixels.")
        for start in range(0, band.shape[0], ROWS_PER_BLOCK):
            encoded: FloatArray = linear_to_srgb(band[start : start + ROWS_PER_BLOCK])
            block: NDArray[np.uint16] = np.rint(np.clip(encoded, 0, 1) * 65535).astype(np.uint16)
            yield from (block if shape[2] == 3 else block[..., 0])
        written += band.shape[0]
    if written != shape[0]:
        raise ValueError("Invalid output pixels.")


def write_tiff(
    path: Path | str,
    frame: ImageFrame,
    pixels: Pixels | Iterable[Pixels],
    source_path: Path | str | None = None,
    cancel: Event | None = None,
) -> None:
    """Write atomically from an array or full-width float row bands, top to bottom.

    A cancellation before replacement preserves the existing destination.
    """
    path = Path(path)
    if source_path is not None:
        source: Path = Path(source_path)
        if path.resolve() == source.resolve() or (
            path.exists() and source.exists() and path.samefile(source)
        ):
            raise ValueError("Choose a new output file; the original image is protected.")
    shape: tuple[int, ...] = frame.pixels.shape
    if isinstance(pixels, np.ndarray) and pixels.shape != shape:
        raise ValueError("Invalid output pixels.")
    check_cancel(cancel)
    rgb: bool = shape[2] == 3
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
            _encoded_rows(pixels, shape, cancel),
            shape=shape if rgb else shape[:2],
            dtype=np.uint16,
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
