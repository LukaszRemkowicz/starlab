from pathlib import Path

import numpy as np
import pytest

from starlab.domain.detection.candidate import Image
from starlab.domain.detection.peaks import _local_mean, _select, peak_positions
from starlab.domain.detection.pipeline import (
    peak_positions_reference as reference_positions,
)


@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("mask", [None, "random", "empty", "half"])
@pytest.mark.parametrize("shape", [(1, 1), (2, 4), (31, 47), (137, 143)])
def test_sparse_peaks_preserve_reference(shape, mask, coarse):
    rng = np.random.default_rng(423)
    rgb = rng.random((*shape, 3), dtype=np.float32)
    allowed = None
    if mask is not None:
        allowed = rng.random(shape) > 0.3
        if mask == "empty":
            allowed[:] = False
        elif mask == "half":
            allowed[shape[0] // 2 :] = False
    image = Image(rgb, origin=(19, 27), allowed=allowed)
    for threshold in (0.0, 0.2, 0.75, 1.0):
        core = (19, 27, shape[1] + 19, shape[0] + 27)
        assert peak_positions(image, threshold, core, coarse) == list(
            reference_positions(image, threshold, core, coarse)
        )


@pytest.mark.parametrize("coarse", [False, True])
def test_sparse_peaks_keep_plateau_bbox_and_halo_topology(coarse):
    rgb = np.zeros((145, 167, 3), np.float32)
    rgb[62:82, 65:93] = 0.8
    # Connected ring: its bounding-box center is a hole and must be checked
    # after labeling, rather than filtering each selected point beforehand.
    rgb[65:79, 69:89] = 0
    rgb[3:6, 3:6] = 1
    allowed = np.ones(rgb.shape[:2], np.bool_)
    allowed[71:74, 77:82] = False
    for mask in (None, allowed):
        image = Image(rgb, origin=(10, 20), allowed=mask)
        for core in ((10, 20, 177, 165), (80, 90, 125, 135)):
            assert peak_positions(image, 0.2, core, coarse) == list(
                reference_positions(image, 0.2, core, coarse)
            )


@pytest.mark.parametrize("value", [0.0, 0.13, 0.6, 0.75, 1.0])
def test_flat_tiles_do_not_create_peaks(value):
    image = Image(np.full((61, 71, 3), value, np.float32))
    for coarse in (False, True):
        assert peak_positions(image, 0.2, (0, 0, 71, 61), coarse) == []


@pytest.mark.parametrize("masked", [False, True])
def test_sparse_floor_finds_dark_pixel_missed_by_nine_probes(monkeypatch, masked):
    from starlab.domain.detection import peaks

    rgb = np.full((145, 151, 3), 0.7, np.float32)
    rgb[72, 75] = 0.8
    rgb[83, 92] = 0.1
    allowed = np.ones(rgb.shape[:2], np.bool_) if masked else None
    if masked:
        allowed[83, 92] = False
        rgb[87, 96] = 0.1
    image = Image(rgb, allowed=allowed)
    core = (0, 0, image.w, image.h)
    # Only the center exceeds threshold; all nine probes miss the dark pixel.
    _, unresolved, count, _ = _select(
        image.rgb,
        image.lum,
        allowed,
        np.float32(0.78),
        np.float32(0.039),
        False,
        image.lum.min(),
    )
    assert count == 1
    assert tuple(unresolved[0]) == (72, 75)
    expected = list(reference_positions(image, 0.78, core))
    assert expected == [(75, 72)]

    def unexpected_full_minimum(*args, **kwargs):
        raise AssertionError("one unresolved candidate must use sparse floor search")

    monkeypatch.setattr(peaks.ndimage, "minimum_filter", unexpected_full_minimum)
    assert peak_positions(image, 0.78, core) == expected


def test_colored_low_luminance_candidates_skip_impossible_floor_search(monkeypatch):
    from starlab.domain.detection import peaks

    rgb = np.zeros((141, 151, 3), np.float32)
    rgb[35:105, 35:115, 2] = 0.9
    image = Image(rgb)
    core = (0, 0, image.w, image.h)
    assert image.lum.max() < np.float32(0.13)
    assert list(reference_positions(image, 0.2, core)) == []

    def unexpected_floor_search(*args, **kwargs):
        raise AssertionError("global lower bound must reject impossible contrast")

    monkeypatch.setattr(peaks.ndimage, "minimum_filter", unexpected_floor_search)
    monkeypatch.setattr(peaks, "_resolve_floor", unexpected_floor_search)
    assert peak_positions(image, 0.2, core) == []


def test_sparse_uniform_preserves_axis_rounding():
    from scipy import ndimage

    rng = np.random.default_rng(923)
    lum = rng.random((37, 41), dtype=np.float32)
    allowed = rng.random(lum.shape) > 0.3
    for mask in (None, allowed):
        values = lum if mask is None else np.where(mask, lum, 0)
        mean = ndimage.uniform_filter(values, size=7, mode="nearest")
        square = ndimage.uniform_filter(values**2, size=7, mode="nearest")
        for y, x in ((0, 0), (3, 9), (18, 20), (36, 40)):
            m, s, _ = _local_mean(lum, mask, y, x, 3)
            assert m == mean[y, x]
            assert s == square[y, x]


@pytest.mark.parametrize("shape", [(1, 1), (2, 4), (31, 47), (137, 143)])
@pytest.mark.parametrize("kind", ["positive", "negative", "exponents", "ties"])
@pytest.mark.parametrize("mask", ["none", "random", "empty"])
def test_rolling_uniform_matches_scipy_at_all_centers(shape, kind, mask):
    from scipy import ndimage

    from starlab.domain.detection.peak_filters import uniform7_at

    rng = np.random.default_rng(913)
    lum = rng.random(shape, dtype=np.float32)
    if kind == "negative":
        lum = lum * np.float32(2) - np.float32(1)
    elif kind == "exponents":
        lum *= np.exp2(rng.integers(-35, 35, shape)).astype(np.float32)
    elif kind == "ties":
        lum = np.nextafter(
            np.full(shape, 0.2, np.float32), np.where(lum > 0.5, np.float32(1), np.float32(0))
        )
    allowed = None if mask == "none" else rng.random(shape) > 0.3
    if mask == "empty":
        allowed[:] = False
    values = lum if allowed is None else np.where(allowed, lum, 0)
    expected = np.stack(
        [
            ndimage.uniform_filter(values, size=7, mode="nearest"),
            ndimage.uniform_filter(values**2, size=7, mode="nearest"),
            np.ones(shape, np.float32)
            if allowed is None
            else ndimage.uniform_filter(allowed.astype(np.float32), size=7, mode="nearest"),
        ],
        axis=-1,
    )
    centers = np.indices(shape).reshape(2, -1).T
    # Arbitrary center order, duplicates and edges all retain caller order.
    rng.shuffle(centers)
    centers = np.concatenate((centers, centers[:3]))
    actual = uniform7_at(lum, allowed, centers)
    y, x = centers.T
    np.testing.assert_array_equal(actual.view(np.uint32), expected[y, x].view(np.uint32))
    sparse = np.array(
        [(shape[0] // 2, shape[1] // 3), (0, 0), (shape[0] // 3, shape[1] // 5)],
        dtype=np.int64,
    )
    actual = uniform7_at(lum, allowed, sparse)
    y, x = sparse.T
    np.testing.assert_array_equal(actual.view(np.uint32), expected[y, x].view(np.uint32))


@pytest.mark.parametrize("name", ["pano-sky-middle.tif", "grid-1536.tif"])
def test_real_photo_crop_preserves_peak_positions(name):
    from starlab.domain.detection.proxy import decode, full_codes, maximum_luminance
    from starlab.infrastructure.tiff_io import read_tiff

    frame = read_tiff(Path("tests/fixtures/detection") / name)
    codes = full_codes(frame.pixels[:512, :640])
    rgb = decode(codes, maximum_luminance(codes, None))
    for masked in (False, True):
        allowed = None
        if masked:
            y, x = np.indices(rgb.shape[:2])
            allowed = y < (rgb.shape[0] * 0.7 + x * 0.1)
        image = Image(rgb, origin=(71, 93), allowed=allowed)
        core = (71, 93, 71 + image.w, 93 + image.h)
        for coarse in (False, True):
            assert peak_positions(image, 0.2, core, coarse) == list(
                reference_positions(image, 0.2, core, coarse)
            )
