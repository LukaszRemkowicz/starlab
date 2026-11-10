import numpy as np
import pytest

from starlab.domain.detection.candidate import Image, candidate


def test_compiled_candidate_preserves_reference_decisions_and_values():
    from starlab.domain.detection.compiled import make_image, qualify

    y, x = np.mgrid[:64, :80]
    gray = 0.03 + 0.9 * np.exp(-((x - 35.3) ** 2 + (y - 27.7) ** 2) / 12)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    allowed = y < 48
    reference = Image(rgb, allowed=allowed)
    fast = make_image(reference)
    excluded = np.empty((0, 3), np.float64)
    for yy in range(23, 33):
        for xx in range(31, 41):
            expected = candidate(reference, xx + 0.5, yy + 0.5)
            actual = qualify(fast, xx + 0.5, yy + 0.5, False, excluded)
            assert (actual is None) == (expected is None), (xx, yy)
            if actual is not None:
                np.testing.assert_allclose(
                    actual[:4],
                    [expected[k] for k in ("x", "y", "size", "brightness")],
                    atol=1e-7,
                    rtol=0,
                )
                np.testing.assert_allclose(actual[4:7], expected["rgb"], atol=1e-7, rtol=0)
    reference.clear_cache()


@pytest.mark.parametrize("budget", [1, 262144])
def test_compiled_full_scan_matches_python_catalog(monkeypatch, budget):
    from pathlib import Path

    from starlab.domain.detection import compiled, pipeline
    from starlab.domain.stars import detect_stars
    from starlab.infrastructure.tiff_io import read_tiff

    frame = read_tiff(Path("tests/fixtures/detection/pano-sky-middle.tif"))
    monkeypatch.setattr(pipeline, "COMPILED", False)
    expected = detect_stars(frame)
    monkeypatch.setattr(pipeline, "COMPILED", True)
    monkeypatch.setattr(compiled, "PARALLEL_MIN_PEAKS", 0)
    monkeypatch.setattr(compiled, "MAX_CANDIDATE_COORDS", budget)
    actual = detect_stars(frame)
    assert len(actual) == len(expected) == 254
    for a, b in zip(actual, expected):
        assert (a.x, a.y, a.radius) == (b.x, b.y, b.radius)
        np.testing.assert_allclose(a.color, b.color, atol=2e-7, rtol=0)


def test_parallel_tiles_preserve_serial_catalog_across_shared_edges(monkeypatch):
    from pathlib import Path

    from starlab.domain.detection import pipeline
    from starlab.domain.stars import detect_stars
    from starlab.infrastructure.tiff_io import read_tiff

    frame = read_tiff(Path("tests/fixtures/detection/grid-1536.tif"))
    monkeypatch.setattr(pipeline, "TILE", 256)
    monkeypatch.setattr(pipeline, "WORKERS", 1)
    expected = detect_stars(frame)
    monkeypatch.setattr(pipeline, "WORKERS", 4)
    assert detect_stars(frame) == expected


def test_compiled_coarse_color_mask_and_nonzero_origin_match_reference():
    from starlab.domain.detection.compiled import make_image, qualify

    y, x = np.mgrid[:96, :112]
    gray = 0.03 + 0.96 * np.exp(-((x - 53.3) ** 2 + (y - 43.7) ** 2) / 100)
    rgb = np.stack((gray, gray * 0.94, gray * 0.9), axis=2).astype(np.float32)
    image = Image(rgb, origin=(200, 300), allowed=y < 70)
    expected = candidate(image, 253.5, 343.5, True)
    actual = qualify(make_image(image), 253.5, 343.5, True, np.empty((0, 3), np.float64))
    assert expected is not None and actual is not None
    np.testing.assert_allclose(
        actual[:4], [expected[k] for k in ("x", "y", "size", "brightness")], rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(actual[4:7], expected["rgb"], rtol=0, atol=1e-7)
    image.clear_cache()
