"""Regression profile from the user's RGB16 TIFF, with metadata and sampled ICC v2 TRCs."""

import base64
import io
import struct
import zlib

import numpy as np
import pytest
import tifffile
from PIL import ImageCms

from starlab.infrastructure.tiff_io import read_tiff, read_tiff_preview

# Exact 3144-byte IEC61966-2.1 profile from laczenie_pelne.tif (no image pixels).
_SRGB_V2 = (
    "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"
    "8eNv94Tz+w=="
)


def _profile():
    return bytearray(zlib.decompress(base64.b64decode(_SRGB_V2)))


def _tag(profile, wanted):
    for index in range(struct.unpack_from(">I", profile, 128)[0]):
        signature, offset, size = struct.unpack_from(">4sII", profile, 132 + index * 12)
        if signature == wanted:
            return offset, size
    raise AssertionError(wanted)


def _write_rgb16(path, profile):
    data = np.array([0, 1, 128, 4096, 32768, 65535], np.uint16).reshape(2, 3)
    tifffile.imwrite(
        path,
        np.repeat(data[..., None], 3, axis=2),
        photometric="rgb",
        metadata=None,
        extratags=[(34675, "B", len(profile), bytes(profile), False)],
    )


@pytest.mark.parametrize("preview", [False, True])
def test_real_srgb_v2_sampled_profile_imports_without_losing_rgb16_precision(tmp_path, preview):
    path = tmp_path / "srgb-v2.tif"
    _write_rgb16(path, _profile())
    frame = read_tiff_preview(path)[0] if preview else read_tiff(path)
    np.testing.assert_allclose(
        frame.pixels[..., 0],
        [[0, 0.00000118103885, 0.000151172972], [0.00515576883, 0.2140482023, 1]],
        rtol=3e-7,
        atol=1e-10,
    )
    assert "Embedded sRGB" in frame.note
    assert frame.pixels.dtype == np.float32


@pytest.mark.parametrize(
    "change", ["curve", "localized-curve", "matrix", "white", "black", "v4-white", "lut"]
)
def test_srgb_description_does_not_hide_changed_colorimetry(tmp_path, change):
    profile = _profile()
    if change == "curve":
        offset, _ = _tag(profile, b"rTRC")
        profile[offset + 12 :] = np.rint(np.linspace(0, 65535, 1024)).astype(">u2").tobytes()
    elif change == "localized-curve":
        offset, _ = _tag(profile, b"rTRC")
        position = offset + 12 + 777 * 2
        old = struct.unpack_from(">H", profile, position)[0]
        struct.pack_into(">H", profile, position, old + 10)
    elif change in ("matrix", "white", "black"):
        offset, _ = _tag(profile, {"matrix": b"rXYZ", "white": b"wtpt", "black": b"bkpt"}[change])
        struct.pack_into(">i", profile, offset + 8, 65536)
    elif change == "v4-white":
        profile[8] = 4
    else:
        for index in range(struct.unpack_from(">I", profile, 128)[0]):
            position = 132 + index * 12
            if profile[position : position + 4] == b"dmnd":
                profile[position : position + 4] = b"A2B0"
    assert "srgb" in ImageCms.getProfileName(ImageCms.ImageCmsProfile(io.BytesIO(profile))).lower()
    path = tmp_path / "changed-srgb.tif"
    _write_rgb16(path, profile)
    with pytest.raises(ValueError, match="not supported"):
        read_tiff(path)
