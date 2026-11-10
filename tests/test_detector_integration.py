"""Production contract and controlled original-catalog regressions."""

import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from starlab.domain.detection import pipeline
from starlab.domain.detection.candidate import Image
from starlab.domain.detection.numeric import LUMA
from starlab.domain.imaging import Cancelled
from starlab.domain.stars import DetectionSettings, detect_stars
from starlab.infrastructure.tiff_io import read_tiff

EVIDENCE = Path(__file__).resolve().parents[1] / "tests/fixtures/detection"


def test_native_grid_geometry_color_and_global_brightness():
    frame = read_tiff(EVIDENCE / "grid-1536.tif")
    native = json.loads((EVIDENCE / "grid-catalog.json").read_text())["records"]
    stats = {}
    stars = detect_stars(frame, diagnostics=stats)
    assert len(stars) == len(native) == 8
    for star in stars:
        record = min(
            native,
            key=lambda r: (r["xy_20"][0] - 0.5 - star.x) ** 2 + (r["xy_20"][1] - 0.5 - star.y) ** 2,
        )
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
    for star in stars:
        record = min(
            native,
            key=lambda r: (r["xy_20"][0] - 0.5 - star.x) ** 2 + (r["xy_20"][1] - 0.5 - star.y) ** 2,
        )
        assert (star.x, star.y) == tuple(v - 0.5 for v in record["xy_20"])
        assert abs(star.radius - record["size_30"]) < 0.126
    # Explicit approximate-seeding tradeoff on a crowded real sky crop:
    # the saved original has256; the fast path must retain at least99%.
    sky = detect_stars(read_tiff(EVIDENCE / "pano-sky-middle.tif"))
    native = json.loads((EVIDENCE / "sky-catalog.json").read_text())["records"]
    matched = sum(
        any((s.x + 0.5 - r["xy_20"][0]) ** 2 + (s.y + 0.5 - r["xy_20"][1]) ** 2 < 1 for s in sky)
        for r in native
    )
    assert matched >= 254
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
