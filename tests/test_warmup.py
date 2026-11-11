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
