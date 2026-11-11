import os
from dataclasses import replace

import numpy as np
import pytest
import tifffile


@pytest.mark.parametrize("height", [16, 300])
def test_rgb16_round_trip_preserves_pixels_and_profile(tmp_path, height):
    from starlab.infrastructure.tiff_io import read_tiff, write_tiff

    data = np.arange(height * 18 * 3, dtype=np.uint16).reshape(height, 18, 3) * 30
    source, target = tmp_path / "input.tif", tmp_path / "output.tif"
    tifffile.imwrite(source, data, photometric="rgb")
    frame = read_tiff(source)
    assert "assumed" in frame.note.lower()
    write_tiff(target, frame, frame.pixels, source_path=source)
    np.testing.assert_allclose(tifffile.imread(target), data, atol=1)
    with tifffile.TiffFile(target) as tf:
        assert tf.pages[0].tags[34675].value
    reread = read_tiff(target)
    np.testing.assert_allclose(reread.pixels, frame.pixels, atol=1e-5)


def test_gray_round_trip_and_source_protection(tmp_path):
    from starlab.infrastructure.tiff_io import read_tiff, write_tiff

    source = tmp_path / "gray.tif"
    data = np.full((12, 13), 12345, np.uint16)
    tifffile.imwrite(source, data)
    frame = read_tiff(source)
    assert frame.pixels.shape == (12, 13, 1)
    with pytest.raises(ValueError):
        write_tiff(source, frame, frame.pixels, source_path=source)
    np.testing.assert_array_equal(tifffile.imread(source), data)
    write_tiff(tmp_path / "out.tif", frame, frame.pixels)
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "out.tif"), data)


@pytest.mark.parametrize("data", [np.zeros((8, 9, 4), np.uint8), np.zeros((8, 9), np.float32)])
def test_unsupported_image_is_rejected(tmp_path, data):
    from starlab.infrastructure.tiff_io import read_tiff

    source = tmp_path / "unsupported.tif"
    tifffile.imwrite(source, data)
    with pytest.raises(ValueError):
        read_tiff(source)


def test_bad_profile_is_rejected(tmp_path):
    from starlab.infrastructure.tiff_io import read_tiff

    source = tmp_path / "profile.tif"
    tifffile.imwrite(
        source,
        np.zeros((8, 9, 3), np.uint16),
        photometric="rgb",
        extratags=[(34675, "B", 4, b"oops", False)],
    )
    with pytest.raises(ValueError):
        read_tiff(source)


def test_preset_roundtrip_and_strict_validation(tmp_path):
    from starlab.domain.effects.settings import EffectSettings
    from starlab.infrastructure.preset_json import load_preset, save_preset

    settings = EffectSettings()
    settings = replace(settings, ring=replace(settings.ring, enabled=True, size=73))
    target = tmp_path / "preset.json"
    save_preset(target, settings)
    assert load_preset(target) == settings
    for bad in ["{}", '{"schema_version":999}', '{"schema_version":1,"settings":{}}']:
        target.write_text(bad)
        with pytest.raises(ValueError):
            load_preset(target)


def test_invalid_export_does_not_destroy_existing_file(tmp_path):
    from starlab.domain.imaging import ImageFrame
    from starlab.infrastructure.tiff_io import write_tiff

    target = tmp_path / "existing.tif"
    target.write_bytes(b"keep this file")
    frame = ImageFrame(np.zeros((8, 8, 3), np.float32))
    with pytest.raises(ValueError):
        write_tiff(target, frame, np.full((8, 8, 3), np.nan, np.float32))
    assert target.read_bytes() == b"keep this file"


@pytest.mark.parametrize("orientation", [2, 6])
def test_nonstandard_orientation_rejected(tmp_path, orientation):
    from starlab.infrastructure.tiff_io import read_tiff

    source = tmp_path / "oriented.tif"
    tifffile.imwrite(
        source, np.zeros((8, 9), np.uint16), extratags=[(274, "H", 1, orientation, False)]
    )
    with pytest.raises(ValueError, match="orientation"):
        read_tiff(source)


def test_profile_named_srgb_with_altered_transfer_curve_rejected(tmp_path):
    import io
    import struct

    from PIL import ImageCms

    from starlab.infrastructure.tiff_io import read_tiff

    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    count = struct.unpack_from(">I", profile, 128)[0]
    for index in range(count):
        signature, offset, size = struct.unpack_from(">4sII", profile, 132 + index * 12)
        if signature == b"rTRC":
            assert profile[offset : offset + 4] == b"para"
            struct.pack_into(">i", profile, offset + 12, 65536)  # gamma 1, same description
    parsed = ImageCms.ImageCmsProfile(io.BytesIO(profile))
    assert "srgb" in ImageCms.getProfileName(parsed).lower()
    source = tmp_path / "wrong-trc.tif"
    tifffile.imwrite(
        source,
        np.full((8, 9, 3), 32768, np.uint16),
        photometric="rgb",
        extratags=[(34675, "B", len(profile), bytes(profile), False)],
    )
    with pytest.raises(ValueError, match="not supported"):
        read_tiff(source)


def test_multipage_tiff_rejected(tmp_path):
    from starlab.infrastructure.tiff_io import read_tiff

    source = tmp_path / "stack.tif"
    with tifffile.TiffWriter(source) as writer:
        writer.write(np.zeros((8, 9), np.uint16))
        writer.write(np.zeros((8, 9), np.uint16))
    with pytest.raises(ValueError, match="single-page"):
        read_tiff(source)


def test_cancelled_export_preserves_destination(tmp_path):
    from threading import Event

    from starlab.domain.imaging import Cancelled, ImageFrame
    from starlab.infrastructure.tiff_io import write_tiff

    target = tmp_path / "existing.tif"
    target.write_bytes(b"existing image")
    frame = ImageFrame(np.zeros((8, 9, 3), np.float32))
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        write_tiff(target, frame, frame.pixels, cancel=cancel)
    assert target.read_bytes() == b"existing image"


def test_tiff_above_40_megapixels_loads_without_resizing(tmp_path):
    from starlab.infrastructure.tiff_io import read_tiff

    source = tmp_path / "large.tif"
    mapped = tifffile.memmap(source, shape=(6400, 6400), dtype=np.uint16, photometric="minisblack")
    mapped[:] = 0
    mapped[6399, 6398] = 32768
    mapped.flush()
    del mapped
    frame = read_tiff(source)
    assert frame.pixels.shape == (6400, 6400, 1)
    assert frame.pixels[6399, 6398, 0] == pytest.approx(32768 / 65535)


@pytest.mark.parametrize("delta, accepted", [(20, True), (2000, False)])
def test_srgb_matrix_variants_keep_strict_transfer_curve_check(tmp_path, delta, accepted):
    import struct

    from PIL import ImageCms

    from starlab.infrastructure.tiff_io import read_tiff

    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    for index in range(struct.unpack_from(">I", profile, 128)[0]):
        signature, offset, size = struct.unpack_from(">4sII", profile, 132 + index * 12)
        if signature == b"chad":
            old = struct.unpack_from(">i", profile, offset + 8)[0]
            struct.pack_into(">i", profile, offset + 8, old + delta)
    source = tmp_path / "variant.tif"
    tifffile.imwrite(
        source,
        np.full((600, 9, 3), 32768, np.uint16),
        photometric="rgb",
        extratags=[(34675, "B", len(profile), bytes(profile), False)],
    )
    if accepted:
        frame = read_tiff(source)
        assert frame.pixels[599, 8, 0] == pytest.approx(0.214048, abs=1e-5)
    else:
        with pytest.raises(ValueError, match="not supported"):
            read_tiff(source)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("rgb", [True, False])
def test_integer_import_matches_existing_transfer_for_every_code(tmp_path, dtype, rgb):
    from starlab.domain.imaging import srgb_to_linear
    from starlab.infrastructure.tiff_io import read_tiff

    maximum = np.iinfo(dtype).max
    codes = np.arange(maximum + 1, dtype=dtype).reshape(-1, 16)
    data = np.repeat(codes[..., None], 3, axis=2) if rgb else codes
    path = tmp_path / "all-codes.tif"
    tifffile.imwrite(path, data, photometric="rgb" if rgb else "minisblack")
    frame = read_tiff(path)
    expected = data.astype(np.float32) / maximum
    expected = srgb_to_linear(expected) if rgb else expected[..., None]
    np.testing.assert_array_equal(frame.pixels, expected)
    assert frame.pixels.flags.owndata and not frame.pixels.flags.writeable


@pytest.mark.parametrize("compression", [None, "deflate"])
@pytest.mark.parametrize("planar", [False, True])
def test_preview_samples_before_conversion_and_preserves_metadata(tmp_path, compression, planar):
    from starlab.domain.imaging import srgb_to_linear
    from starlab.infrastructure.tiff_io import read_tiff_preview

    data = np.arange(43 * 71 * 3, dtype=np.uint16).reshape(43, 71, 3) * 7
    path = tmp_path / "preview.tif"
    tifffile.imwrite(
        path,
        np.moveaxis(data, -1, 0) if planar else data,
        photometric="rgb",
        planarconfig="separate" if planar else "contig",
        compression=compression,
    )
    thumb, scale, width, height = read_tiff_preview(path, max_edge=24)
    assert (scale, width, height) == (1 / 3, 71, 43)
    assert thumb.name == path.name and "assumed" in thumb.note.lower()
    np.testing.assert_array_equal(
        thumb.pixels, srgb_to_linear(data[::3, ::3].astype(np.float32) / 65535)
    )


def test_preview_rejects_orientation_and_checks_cancellation(tmp_path):
    from threading import Event

    from starlab.domain.imaging import Cancelled
    from starlab.infrastructure.tiff_io import read_tiff_preview

    path = tmp_path / "orientation.tif"
    tifffile.imwrite(path, np.zeros((8, 9), np.uint16), extratags=[(274, "H", 1, 6, False)])
    with pytest.raises(ValueError, match="orientation"):
        read_tiff_preview(path)
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        read_tiff_preview(path, cancel)


@pytest.mark.parametrize("shape", [(2, 70), (70, 2), (2, 2)])
def test_preview_accepts_thin_valid_images(tmp_path, shape):
    from starlab.infrastructure.tiff_io import read_tiff_preview

    path = tmp_path / "thin.tif"
    tifffile.imwrite(path, np.full(shape, 8192, np.uint16))
    frame, _, width, height = read_tiff_preview(path, max_edge=24)
    assert (height, width) == shape
    assert min(frame.pixels.shape[:2]) >= 2
    assert max(frame.pixels.shape[:2]) <= 24
    np.testing.assert_array_equal(frame.pixels, np.float32(8192 / 65535))


def test_reader_owned_buffer_transfer_does_not_weaken_public_copy_contract():
    from starlab.domain.imaging import ImageFrame

    data = np.zeros((8, 9, 3), np.float32)
    public = ImageFrame(data)
    data[0, 0] = 1
    assert public.pixels[0, 0].max() == 0
    trusted = ImageFrame._from_owned_validated(data, "reader", "linear-srgb", "")
    assert trusted.pixels is data and not data.flags.writeable
    with pytest.raises(ValueError, match="owned"):
        ImageFrame._from_owned_validated(data[::2], "view", "linear-srgb", "")


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_export_and_preset_files_follow_umask(tmp_path):
    from starlab.domain.effects.settings import EffectSettings
    from starlab.infrastructure.preset_json import save_preset
    from starlab.infrastructure.tiff_io import read_tiff, write_tiff

    source = tmp_path / "in.tif"
    tifffile.imwrite(source, np.zeros((4, 4, 3), np.uint16), photometric="rgb")
    frame = read_tiff(source)
    umask = os.umask(0o022)
    try:
        write_tiff(tmp_path / "out.tif", frame, frame.pixels, source_path=source)
        save_preset(tmp_path / "preset.json", EffectSettings())
    finally:
        os.umask(umask)
    assert (tmp_path / "out.tif").stat().st_mode & 0o777 == 0o644
    assert (tmp_path / "preset.json").stat().st_mode & 0o777 == 0o644
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.tif", "out.tif", "preset.json"]


def test_compressed_tiff_decodes_in_memory_without_temp_files(tmp_path, monkeypatch):
    import tempfile as tempfile_module

    from starlab.infrastructure.tiff_io import read_tiff

    source = tmp_path / "compressed.tif"
    data = np.arange(32 * 40 * 3, dtype=np.uint16).reshape(32, 40, 3)
    tifffile.imwrite(source, data, photometric="rgb", compression="zlib")

    def no_temp_files(*args, **kwargs):
        raise AssertionError("photo data must not be written to a temporary file")

    monkeypatch.setattr(tempfile_module, "NamedTemporaryFile", no_temp_files)
    frame = read_tiff(source)
    assert frame.pixels.shape == (32, 40, 3)
