"""Matrix/TRC RGB profiles convert to linear sRGB like LittleCMS (relative, no BPC)."""

import struct

import numpy as np
import pytest
import tifffile
from PIL import ImageCms

from starlab.domain.imaging import linear_to_srgb
from starlab.infrastructure.sky_mask import SkyMaskService
from starlab.infrastructure.tiff_io import (
    _convert_integer_image,
    _evaluate_curve,
    read_tiff,
    read_tiff_preview,
)

PROPHOTO = [(0.7977, 0.2880, 0.0), (0.1352, 0.7119, 0.0001), (0.0313, 0.0001, 0.8249)]
ADOBE = [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)]
D50 = (0.9642, 1.0, 0.8249)


def _fixed(values):
    return b"".join(struct.pack(">i", round(v * 65536)) for v in values)


def _xyz(values):
    return b"XYZ \0\0\0\0" + _fixed(values)


def gamma_curve(gamma):
    return b"curv\0\0\0\0" + struct.pack(">IH", 1, round(gamma * 256)) + b"\0\0"


def make_profile(colorants, curve, extra=None, name="Test RGB"):
    """Build a minimal ICC v2.1 monitor RGB matrix/TRC profile."""
    ascii_name = name.encode() + b"\0"
    desc = b"desc\0\0\0\0" + struct.pack(">I", len(ascii_name)) + ascii_name
    desc += bytes(4 + 4 + 2 + 1 + 67)
    tags = {b"desc": desc, b"cprt": b"text\0\0\0\0x\0", b"wtpt": _xyz(D50)}
    for signature, colorant in zip((b"rXYZ", b"gXYZ", b"bXYZ"), colorants, strict=True):
        tags[signature] = _xyz(colorant)
    for signature in (b"rTRC", b"gTRC", b"bTRC"):
        tags[signature] = curve
    tags.update(extra or {})
    offset = 132 + 12 * len(tags)
    table, body = b"", b""
    for signature, data in tags.items():
        padded = data + bytes(-len(data) % 4)
        table += struct.pack(">4sII", signature, offset + len(body), len(data))
        body += padded
    header = bytearray(128)
    struct.pack_into(">I", header, 8, 0x02100000)
    header[12:24] = b"mntrRGB XYZ "
    header[36:40] = b"acsp"
    header[68:80] = _fixed((0.9642, 1.0, 0.8249))
    profile = bytearray(header + struct.pack(">I", len(tags)) + table + body)
    struct.pack_into(">I", profile, 0, len(profile))
    return bytes(profile)


def write(path, data, profile, **options):
    tifffile.imwrite(
        path,
        data,
        photometric="rgb",
        metadata=None,
        extratags=[(34675, "B", len(profile), profile, False)],
        **options,
    )


def numpy_oracle(codes, maximum, colorants, gamma):
    """Decode gamma, map ROMM/Adobe-like colorants to sRGB primaries, clip."""
    reference = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    target = (
        np.array(
            [
                struct.unpack_from(">3i", reference, _tag(reference, s) + 8)
                for s in (b"rXYZ", b"gXYZ", b"bXYZ")
            ]
        ).T
        / 65536
    )
    matrix = np.linalg.solve(target, np.round(np.array(colorants).T * 65536) / 65536)
    linear = (codes / maximum) ** (round(gamma * 256) / 256)
    return linear @ matrix.T


def _tag(profile, wanted):
    for index in range(struct.unpack_from(">I", profile, 128)[0]):
        signature, offset, _ = struct.unpack_from(">4sII", profile, 132 + index * 12)
        if signature == wanted:
            return offset
    raise AssertionError(wanted)


def test_gamma_parametric_and_sampled_curves():
    x = np.linspace(0, 1, 11)
    assert np.array_equal(_evaluate_curve(b"curv\0\0\0\0\0\0\0\0", x), x)  # identity
    np.testing.assert_allclose(
        _evaluate_curve(gamma_curve(1.8), x), x ** (461 / 256), atol=1e-12
    )  # u8Fixed8 gamma
    sampled = b"curv\0\0\0\0" + struct.pack(">I3H", 3, 0, 16384, 65535)  # piecewise linear
    np.testing.assert_allclose(
        _evaluate_curve(sampled, np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
        [0, 0.125, 0.25, 0.625, 1],
        atol=1e-4,
    )
    # parametric type 3 with sRGB parameters
    parameters = _fixed([2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045])
    srgb_like = b"para\0\0\0\0" + struct.pack(">HH", 3, 0) + parameters
    expected = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    np.testing.assert_allclose(
        _evaluate_curve(srgb_like, x), expected, atol=5e-4
    )  # s15Fixed16 parameters
    with pytest.raises(ValueError):
        _evaluate_curve(b"para\0\0\0\0" + struct.pack(">HH", 9, 0), x)
    with pytest.raises(ValueError):
        _evaluate_curve(b"curv\0\0\0\0" + struct.pack(">I", 50), x)


@pytest.mark.parametrize(("colorants", "gamma"), [(PROPHOTO, 1.8), (ADOBE, 2.19921875)])
def test_conversion_matches_littlecms_for_in_gamut_colors(tmp_path, colorants, gamma):
    profile = make_profile(colorants, gamma_curve(gamma))
    levels = np.arange(8, 256, 24)
    grid = (
        np.stack(np.meshgrid(levels, levels, levels), axis=-1).reshape(-1, 3).astype(np.uint8)[:-1]
    )  # even count
    image = grid.reshape(-1, 2, 3)
    path = tmp_path / "grid.tif"
    write(path, image, profile)
    ours = np.rint(linear_to_srgb(read_tiff(path).pixels) * 255).reshape(-1, 3)
    source = ImageCms.ImageCmsProfile(__import__("io").BytesIO(profile))
    from PIL import Image

    cms = ImageCms.profileToProfile(
        Image.fromarray(image),
        source,
        ImageCms.createProfile("sRGB"),
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        outputMode="RGB",
    )
    reference = np.asarray(cms).reshape(-1, 3).astype(float)
    unclipped = numpy_oracle(grid.astype(float), 255, colorants, gamma)
    inside = np.all((unclipped > 0.002) & (unclipped < 0.998), axis=1)
    assert inside.sum() > 50
    difference = np.abs(ours[inside] - reference[inside])
    assert difference.max() <= 2, (difference.max(), difference.mean())


def test_read_path_matches_numpy_oracle_streamed_and_thumbnail(tmp_path):
    profile = make_profile(PROPHOTO, gamma_curve(1.8), name="ProPhoto test")
    data = np.random.default_rng(5).integers(0, 65536, (53, 71, 3), dtype=np.uint16)
    path = tmp_path / "wide.tif"
    write(path, data, profile, compression="zlib", rowsperstrip=7)
    frame = read_tiff(path)
    expected = np.clip(numpy_oracle(data.astype(float), 65535, PROPHOTO, 1.8), 0, 1)
    np.testing.assert_allclose(frame.pixels, expected, atol=1e-6)
    assert frame.note == (
        "Converted from 'ProPhoto test' to sRGB (relative colorimetric); "
        "colors outside sRGB were clipped."
    )
    preview, scale, width, height = read_tiff_preview(path, max_edge=20)
    stride = round(1 / scale)
    assert (width, height) == (71, 53)
    np.testing.assert_allclose(preview.pixels, expected[::stride, ::stride], atol=1e-6)
    # Band size must not matter.
    from starlab.infrastructure.tiff_io import EncodedRows, _validated_page

    with tifffile.TiffFile(path) as tiff_file:
        page, note, conversion = _validated_page(tiff_file)
        with EncodedRows(page) as reader:
            small = _convert_integer_image(
                reader.rows, (53, 71, 3), 65535, path, note, None, 5, conversion
            )
    np.testing.assert_array_equal(small.pixels, frame.pixels)


def test_sky_mask_input_is_converted_to_srgb_encoding(tmp_path):
    profile = make_profile(ADOBE, gamma_curve(2.19921875))
    small = np.random.default_rng(6).integers(8000, 60000, (53, 71, 3), dtype=np.uint16)
    data = np.repeat(np.repeat(small, 2, 0), 2, 1)  # constant 2x2 blocks: reduction commutes
    wide = tmp_path / "wide.tif"
    write(wide, data, profile)
    converted = np.clip(numpy_oracle(data.astype(float), 65535, ADOBE, 2.19921875), 0, 1)
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(
        plain,
        np.rint(linear_to_srgb(converted.astype(np.float32)) * 65535).astype(np.uint16),
        photometric="rgb",
        metadata=None,
    )

    class Network:
        tile, overlap, identity = 8, 2, "test-model"

        def __init__(self):
            self.tiles = []

        def sky_logits(self, tile):
            self.tiles.append(tile.copy())
            return np.ones(tile.shape[:2], np.float32)

    seen = []
    for path in (wide, plain):
        network = Network()
        SkyMaskService(segmenter=network, persistent_cache=False)._from_encoded_source(
            (106, 142), path, None
        )
        seen.append(np.stack(network.tiles))
    np.testing.assert_allclose(seen[0], seen[1], atol=0.02)


def test_lut_based_or_unknown_tag_profiles_are_still_rejected(tmp_path):
    for extra in ({b"A2B0": b"mft2" + bytes(60)}, {b"zzzz": bytes(8)}):
        path = tmp_path / "lut.tif"
        write(path, np.zeros((4, 4, 3), np.uint8), make_profile(PROPHOTO, gamma_curve(1.8), extra))
        with pytest.raises(ValueError, match="not supported"):
            read_tiff(path)
    path = tmp_path / "gray.tif"
    tifffile.imwrite(
        path,
        np.zeros((4, 4), np.uint8),
        photometric="minisblack",
        extratags=[(34675, "B", 10, bytes(10), False)],
    )
    with pytest.raises(ValueError):
        read_tiff(path)
