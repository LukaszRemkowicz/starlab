import tempfile
from pathlib import Path

from starlab.application.warmup import warm_up
from starlab.bootstrap import create_services


def test_warm_up_runs_the_production_path_and_leaves_no_files():
    before = set(Path(tempfile.gettempdir()).glob("star-effects-lab-*"))
    assert warm_up(create_services(sky_mask=True)) > 0
    assert set(Path(tempfile.gettempdir()).glob("star-effects-lab-*")) == before


def test_open_and_detection_wait_for_background_compilation(tmp_path):
    services = create_services(sky_mask=False)
    calls = []
    services.ready = calls.append
    source = tmp_path / "sky.tif"
    from starlab.application.warmup import _synthetic_sky

    frame = _synthetic_sky()
    services.write(source, frame, frame.pixels)
    frame, _ = services.open_image(source)
    services.analyze(frame, source)
    assert calls == [None, None]


def test_background_preparation_succeeds_and_logs_nothing(tmp_path, caplog):
    from starlab.bootstrap import start_preparation

    process, ready = start_preparation(tmp_path / "preparation.log")
    ready(None)
    assert process.wait(timeout=300) == 0
    assert "Background preparation" not in caplog.text


def test_background_preparation_stops_when_the_app_goes_away(tmp_path, monkeypatch):
    from starlab.bootstrap import start_preparation

    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))  # cold: a full warm-up takes ~20 s
    process, _ = start_preparation()
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
