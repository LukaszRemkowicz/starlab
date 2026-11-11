"""Streaming TIFF decode must match a full tifffile decode for every storage layout."""

import numpy as np
import pytest
import tifffile

from starlab.domain.sky_mask import build_sky_mask
from starlab.infrastructure.sky_mask import SkyMaskService, _block_mean
from starlab.infrastructure.tiff_io import (
    EncodedRows,
    _integer_transfer,
    read_tiff,
    read_tiff_preview,
)

H, W = 53, 71
LAYOUTS = {
    "plain": {},
    "zlib-strips": {"compression": "zlib", "rowsperstrip": 7},
    "lzw-strips": {"compression": "lzw", "rowsperstrip": 5},
    "zlib-one-strip": {"compression": "zlib", "rowsperstrip": H},
    "zlib-tiled": {"compression": "zlib", "tile": (16, 32)},
    "planar-zlib": {"compression": "zlib", "rowsperstrip": 6, "planarconfig": "separate"},
    "planar-plain": {"planarconfig": "separate"},
    "big-endian-zlib": {"compression": "zlib", "rowsperstrip": 9, "byteorder": ">"},
    "big-endian-plain": {"byteorder": ">"},
}


def _write(tmp_path, name, dtype, channels):
    rng = np.random.default_rng(1)
    top = np.iinfo(dtype).max
    data = rng.integers(0, top, (H, W, channels), dtype=dtype, endpoint=True)
    kwargs = dict(LAYOUTS[name])
    if channels == 1:
        kwargs.pop("planarconfig", None)
    stored = np.moveaxis(data, -1, 0) if kwargs.get("planarconfig") else data
    path = tmp_path / f"{name}.tif"
    tifffile.imwrite(
        path,
        stored[..., 0] if channels == 1 else stored,
        photometric="rgb" if channels == 3 else "minisblack",
        metadata=None,
        **kwargs,
    )
    return path, data


def _linear(data):
    return _integer_transfer(int(np.iinfo(data.dtype).max))[data]


CASES = [(n, d, c) for n in LAYOUTS for d in (np.uint8, np.uint16) for c in (1, 3)]
IDS = [f"{n}-{np.dtype(d).name}-{c}" for n, d, c in CASES]


@pytest.mark.parametrize(("name", "dtype", "channels"), CASES, ids=IDS)
def test_streamed_frame_rows_and_preview_match_full_decode(tmp_path, name, dtype, channels):
    path, data = _write(tmp_path, name, dtype, channels)
    reference = tifffile.imread(path).reshape(
        (channels, H, W) if "planar" in name and channels == 3 else (H, W, channels)
    )
    reference = np.moveaxis(reference, 0, -1) if "planar" in name and channels == 3 else reference
    assert np.array_equal(reference, data)
    assert np.array_equal(read_tiff(path).pixels, _linear(data))
    for edge in (20, 1600):
        frame, scale, width, height = read_tiff_preview(path, max_edge=edge)
        stride = round(1 / scale)
        assert (width, height) == (W, H)
        assert np.array_equal(frame.pixels, _linear(data[::stride, ::stride]))
    with tifffile.TiffFile(path) as tiff_file:
        reader = EncodedRows(tiff_file.pages[0])
        # Sequential, overlapping, repeated and strided requests all agree with the reference.
        for top, bottom, step in [(0, 10, 1), (10, 30, 1), (28, 53, 1), (0, 53, 1), (3, 40, 4)]:
            got = reader.rows(top, bottom, step)
            assert np.array_equal(got[::step], data[top:bottom][::step])
            assert got.dtype.isnative


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_streamed_mask_input_matches_full_decode(tmp_path, name):
    path, data = _write(tmp_path, name, np.uint16, 3)
    big = np.repeat(np.repeat(data, 2, 0), 2, 1)  # 106 x 142 so the mask grid has several tiles
    path = tmp_path / "big.tif"
    layout = dict(LAYOUTS[name])
    stored = np.moveaxis(big, -1, 0) if layout.get("planarconfig") else big
    tifffile.imwrite(path, stored, photometric="rgb", metadata=None, **layout)

    class Network:
        tile, overlap, identity = 8, 2, "test-model"

        def __init__(self):
            self.tiles = []

        def sky_logits(self, tile):
            self.tiles.append(tile.copy())
            return np.ones(tile.shape[:2], np.float32)

    streamed, reference = Network(), Network()
    got = SkyMaskService(segmenter=streamed, persistent_cache=False)._from_encoded_source(
        big.shape[:2], path, None
    )
    want = build_sky_mask(
        big.shape[:2], lambda top, bottom: big[top:bottom], reference, block_mean=_block_mean
    )
    assert len(streamed.tiles) == len(reference.tiles) > 1
    assert all(np.array_equal(a, b) for a, b in zip(streamed.tiles, reference.tiles, strict=True))
    assert np.array_equal(got.classes, want.classes)


def test_only_segments_with_requested_rows_are_decoded(tmp_path):
    path, data = _write(tmp_path, "zlib-strips", np.uint16, 3)  # strips of 7 rows
    with tifffile.TiffFile(path) as tiff_file:
        page = tiff_file.pages[0]
        reader = EncodedRows(page)
        decoded = []
        original = page.decode
        page.decode = lambda blob, index, **kw: (
            decoded.append(index),
            original(blob, index, **kw),
        )[1]
        reader.rows(0, 10)
        assert sorted(decoded) == [0, 1]
        decoded.clear()
        reader.rows(10, 20)  # strip 1 (rows 7-13) is reused from the previous band
        assert sorted(decoded) == [2]
        decoded.clear()
        reader.rows(0, 53, 21)  # sampled rows 0, 21, 42 live in strips 0, 3, 6
        assert sorted(decoded) == [0, 3, 6]
