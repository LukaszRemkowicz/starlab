import tempfile
from pathlib import Path

from starlab.application.warmup import _synthetic_sky, warm_up
from starlab.bootstrap import create_services


def test_warm_up_runs_the_production_path_and_leaves_no_files():
    from starlab.infrastructure.tiff_io import write_samples

    before = set(Path(tempfile.gettempdir()).glob("star-effects-lab-*"))
    assert warm_up(create_services(sky_mask=True), write_samples) > 0
    assert set(Path(tempfile.gettempdir()).glob("star-effects-lab-*")) == before


def test_open_and_detection_wait_for_background_compilation(tmp_path):
    services = create_services(sky_mask=False)
    calls = []
    services.ready = calls.append
    source = tmp_path / "sky.tif"
    frame = _synthetic_sky(64, 64, 20)
    services.write(source, frame, frame.pixels)
    frame, _ = services.open_image(source)
    services.analyze(frame, source)
    assert calls == [None, None]


def test_background_preparation_runs_once_per_program_build(tmp_path, monkeypatch, caplog):
    from starlab.bootstrap import start_preparation
    from starlab.preparation import code_signature

    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba"))  # keep the dev cache intact

    stamp = tmp_path / "prepared-code"
    process, ready = start_preparation(tmp_path / "preparation.log", stamp)
    ready(None)
    assert process.wait(timeout=600) == 0
    assert stamp.read_text(encoding="utf-8") == code_signature()
    assert "Background preparation" not in caplog.text
    process, ready = start_preparation(tmp_path / "preparation.log", stamp)
    assert process.wait(timeout=10) == 0  # already prepared: returns almost at once


def test_background_preparation_stops_when_the_app_goes_away(tmp_path, monkeypatch):
    from starlab.bootstrap import start_preparation

    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))  # cold: a full warm-up takes ~30 s
    process, _ = start_preparation(tmp_path / "preparation.log")
    process.stdin.close()  # what the operating system does when the app exits or crashes
    assert process.wait(timeout=8) == 0


def test_failed_background_preparation_is_logged(tmp_path, monkeypatch, caplog):
    from starlab import bootstrap

    monkeypatch.setattr(bootstrap, "PREPARE", "import sys; sys.exit(3)")
    log = tmp_path / "preparation.log"
    process, ready = bootstrap.start_preparation(log)
    ready(None)
    assert process.returncode == 3
    assert "exit code 3" in caplog.text and str(log) in caplog.text


def test_unwritable_preparation_log_raises(tmp_path):
    import pytest

    from starlab.bootstrap import start_preparation

    blocked = tmp_path / "file"
    blocked.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        start_preparation(blocked / "preparation.log")


def test_new_build_clears_stale_compiled_code(tmp_path, monkeypatch):
    from starlab import preparation

    cache = tmp_path / "cache"
    stale = cache / "Users" / "starlab" / "domain" / "kernel.f-1.py314.nbi"
    other = cache / "Users" / "otherapp" / "kernel.f-1.py314.nbi"
    for path in (stale, other):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"old machine code")
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache))
    preparation.clear_compiled_code()
    assert not stale.exists()
    assert other.exists()  # other programs sharing the cache folder are left alone
