"""Production contract and controlled original-catalog regressions."""

import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from starlab.domain.detection import pipeline
from starlab.domain.detection.candidate import Image
from starlab.domain.detection.numeric import LUMA
from starlab.domain.imaging import Cancelled
from starlab.domain.stars import DetectionSettings, detect_stars
from starlab.infrastructure.tiff_io import read_tiff

EVIDENCE = Path(__file__).resolve().parents[1] / "tests/fixtures/detection"


def native_matches(stars, native, tolerance=1):
    if not stars or not native:
        return []
    actual = np.array([(s.x + 0.5, s.y + 0.5) for s in stars])
    expected = np.array([row["xy_20"] for row in native])
    distance = np.sum((actual[:, None] - expected[None]) ** 2, axis=2)
    valid = distance < tolerance**2
    # One forbidden edge costs more than every possible valid edge combined:
    # the assignment maximizes distinct in-tolerance matches before distance.
    cost = np.where(valid, distance / tolerance**2, max(distance.shape) + 1)
    rows, columns = linear_sum_assignment(cost)
    return [(stars[i], native[j]) for i, j in zip(rows, columns) if valid[i, j]]


def test_native_grid_geometry_color_and_global_brightness():
    frame = read_tiff(EVIDENCE / "grid-1536.tif")
    native = json.loads((EVIDENCE / "grid-catalog.json").read_text())["records"]
    stats = {}
    stars = detect_stars(frame, diagnostics=stats)
    assert len(stars) == len(native) == 8
    matched = native_matches(stars, native)
    assert len(matched) == len(native)
    for star, record in matched:
        assert (star.x, star.y) == tuple(v - 0.5 for v in record["xy_20"])
        assert star.radius == record["size_30"]
        np.testing.assert_allclose(star.color, record["rgb_00"], atol=2e-6, rtol=0)
        assert star.brightness == pytest.approx(np.dot(star.color, LUMA), abs=1e-7)
    assert min(s.brightness for s in stars) < max(s.brightness for s in stars) * 0.3
    assert stats["full"]["candidate_seeds"] < frame.width * frame.height // 1000
    assert DetectionSettings().max_stars is None


def test_tile_size_does_not_normalize_or_rank_each_tile(monkeypatch):
    frame = read_tiff(EVIDENCE / "grid-1536.tif")
    baseline = detect_stars(frame)
    monkeypatch.setattr(pipeline, "TILE", 256)
    assert detect_stars(frame) == baseline
    assert len(detect_stars(frame, DetectionSettings(threshold_sigma=30))) == 4
    assert len(detect_stars(frame, DetectionSettings(max_stars=3))) == 3


def test_positive_coarse_pass_and_fast_seed_coverage():
    stats = {}
    stars = detect_stars(read_tiff(EVIDENCE / "coarse-1024.tif"), diagnostics=stats)
    assert len(stars) == 2 and stats["coarse_count"] == 1
    native = json.loads((EVIDENCE / "coarse-catalog.json").read_text())["records"]
    matched = native_matches(stars, native)
    assert len(matched) == len(stars) == len(native)
    for star, record in matched:
        assert (star.x, star.y) == tuple(v - 0.5 for v in record["xy_20"])
        assert abs(star.radius - record["size_30"]) < 0.126
    # Explicit approximate-seeding tradeoff on a crowded real sky crop:
    # the saved original has256; the fast path must retain at least99%.
    sky = detect_stars(read_tiff(EVIDENCE / "pano-sky-middle.tif"))
    native = json.loads((EVIDENCE / "sky-catalog.json").read_text())["records"]
    matched = native_matches(sky, native)
    assert len(matched) / len(native) >= 0.99
    assert len(matched) == len(sky), "Detections must not reuse references or add false sources."
    for side in ("left", "right"):
        assert len(detect_stars(read_tiff(EVIDENCE / f"pano-sky-{side}.tif"))) == 13


def test_fast_seeds_reject_low_contrast_texture_on_a_bright_background():
    # A remote black border passes the old distant-minimum screen. The local
    # texture has no star-shaped contrast; its entire range is only0.04.
    rng = np.random.default_rng(8)
    values = np.zeros((96, 96), dtype=np.float32)
    values[8:88, 8:88] = rng.uniform(0.28, 0.32, size=(80, 80))
    image = Image(np.repeat(values[:, :, None], 3, axis=2))
    try:
        assert list(pipeline.peak_positions(image, 0.2, (24, 24, 72, 72))) == []
    finally:
        image.clear_cache()


def test_detector_cancel_before_allocating_proxy():
    token = Event()
    token.set()
    with pytest.raises(Cancelled):
        detect_stars(read_tiff(EVIDENCE / "single-1024.tif"), cancel=token)
