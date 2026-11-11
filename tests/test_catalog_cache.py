from dataclasses import replace

import numpy as np

from starlab.domain.imaging import ImageFrame
from starlab.domain.stars import DetectionSettings, Star
from starlab.infrastructure import catalog_cache


def test_cache_invalidates_file_parameters_and_engine(tmp_path, monkeypatch):
    source = tmp_path / "source.tif"
    source.write_bytes(b"image")
    target = tmp_path / "catalog.json"
    monkeypatch.setattr(catalog_cache, "cache_path", lambda _, settings=None: target)
    settings = DetectionSettings()
    stars = (Star(1, 10.25, 12.5, 0.87, (0.87, 0.87, 0.87)),)
    catalog_cache.save_catalog(source, settings, stars)
    assert catalog_cache.load_catalog(source, settings) == stars
    assert catalog_cache.load_catalog(source, replace(settings, threshold_sigma=10)) is None
    source.write_bytes(b"changed-image")
    assert catalog_cache.load_catalog(source, settings) is None
    catalog_cache.save_catalog(source, settings, stars)
    monkeypatch.setattr(catalog_cache, "engine_signature", lambda: "new-code")
    assert catalog_cache.load_catalog(source, settings) is None


def test_corrupt_cache_and_failed_write_do_not_break_analysis(tmp_path, monkeypatch):
    source = tmp_path / "source.tif"
    source.write_bytes(b"image")
    target = tmp_path / "catalog.json"
    target.write_text("invalid json")
    monkeypatch.setattr(catalog_cache, "cache_path", lambda _, settings=None: target)
    assert catalog_cache.load_catalog(source, DetectionSettings()) is None
    monkeypatch.setattr(
        catalog_cache, "cache_path", lambda _, settings=None: source / "unwritable.json"
    )
    catalog_cache.save_catalog(source, DetectionSettings(), ())


def test_analysis_uses_cache_and_manual_request_forces_detection(tmp_path, monkeypatch):
    from starlab.application import use_cases

    frame = ImageFrame(np.zeros((8, 8, 3), np.float32))
    source = tmp_path / "source.tif"
    source.write_bytes(b"image")
    cached = (Star(1, 2, 3, 0.5, (0.5, 0.5, 0.5)),)
    calls = []
    monkeypatch.setattr(use_cases, "detect_stars", lambda *args: calls.append("detect") or ())
    service = use_cases.ImageServices(
        None, None, lambda *_: cached, lambda *_: calls.append("save")
    )
    assert service.analyze(frame, source) == cached
    assert not calls
    assert service.analyze(frame, source, force=True) == ()
    assert calls == ["detect", "save"]


def test_jit_configuration_change_invalidates_persisted_catalog(tmp_path, monkeypatch):
    """A compiler-option change must not reuse a previously computed source catalog."""
    from pathlib import Path

    source, target = tmp_path / "source.tif", tmp_path / "catalog.json"
    source.write_bytes(b"image")
    monkeypatch.setattr(catalog_cache, "cache_path", lambda _, settings=None: target)
    original_read = Path.read_bytes
    changed = False

    def read_bytes(path):
        data = original_read(path)
        return (
            data + b"\n# Changed JIT configuration" if changed and path.name == "jit.py" else data
        )

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    settings = DetectionSettings()
    stars = (Star(1, 2, 3, 0.5, (0.5, 0.5, 0.5)),)
    catalog_cache.engine_signature.cache_clear()
    try:
        catalog_cache.save_catalog(source, settings, stars)
        assert catalog_cache.load_catalog(source, settings) == stars
        changed = True
        catalog_cache.engine_signature.cache_clear()  # Simulate a fresh process.
        assert catalog_cache.load_catalog(source, settings) is None
    finally:
        catalog_cache.engine_signature.cache_clear()
