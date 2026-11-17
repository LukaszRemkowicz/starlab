import json
import os
import struct
import tempfile
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import tifffile

from starlab.domain.effects.settings import EffectSettings
from starlab.domain.imaging import Cancelled
from starlab.domain.stars import DetectionSettings
from starlab.infrastructure.photoshop_bridge import SETTINGS_CAPACITY, PhotoshopBridge
from starlab.infrastructure.preset_json import decode_session_settings, encode_session_settings

CASES = [(8, 255, 3), (16, 65535, 3), (16, 32768, 3), (8, 255, 1), (16, 32768, 1)]


@pytest.fixture
def make_block():
    """Create plug-in-style blocks and always unlink them."""
    created: list[SharedMemory] = []
    bridges: list[PhotoshopBridge] = []

    def make(
        width=40,
        height=30,
        channels=3,
        depth=8,
        maximum=255,
        samples=None,
        header=None,
        extra=SETTINGS_CAPACITY,
    ):
        dtype = np.uint8 if depth == 8 else np.uint16
        if samples is None:
            samples = np.random.default_rng(1).integers(
                0, maximum + 1, (height, width, channels), dtype=dtype
            )
        size = 64 + samples.nbytes + extra
        block = SharedMemory(name="selb" + uuid.uuid4().hex[:12], create=True, size=size)
        created.append(block)
        fields = {
            "magic": b"SELB",
            "version": 2,
            "width": width,
            "height": height,
            "channels": channels,
            "depth": depth,
            "maximum": maximum,
            "status": 0,
            "mode": 0,
            "length": 0,
        } | (header or {})
        struct.pack_into(
            "<4s9I",
            block.buf,
            0,
            *[
                fields[k]
                for k in "magic version width height channels depth maximum status mode length".split()
            ],
        )
        block.buf[64 : 64 + samples.nbytes] = samples.tobytes()
        return block, samples

    def attach(block):
        bridge = PhotoshopBridge(block.name)
        bridges.append(bridge)
        return bridge

    make.attach = attach
    yield make
    for bridge in bridges:
        bridge.close()
    for block in created:
        block.close()
        block.unlink()


def _codes(block, samples):
    return np.frombuffer(block.buf, samples.dtype, samples.size, 64).reshape(samples.shape)


@pytest.mark.parametrize(("depth", "maximum", "channels"), CASES)
def test_round_trip_is_exact(make_block, depth, maximum, channels):
    """Table decode then encode must return every code: float32 error is far below half a code."""
    block, samples = make_block(channels=channels, depth=depth, maximum=maximum)
    bridge = make_block.attach(block)
    frame = bridge.read()
    assert frame.pixels.shape == samples.shape and frame.name == "Photoshop layer"
    bridge.write(None, frame, frame.pixels)
    assert np.array_equal(_codes(block, samples), samples)
    assert bridge.status == 1 and bridge.exit_code == 0


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ({"magic": b"XXXX"}, "signature"),
        ({"version": 1}, "version"),
        ({"mode": 2}, "mode"),
        ({"mode": 1}, "settings length"),  # repeat without settings
        ({"length": SETTINGS_CAPACITY + 1}, "settings length"),
        ({"channels": 4}, "channel"),
        ({"depth": 12}, "depth"),
        ({"maximum": 256}, "white"),
        ({"maximum": 0}, "white"),
        ({"width": 1}, "2 × 2"),
        ({"status": 1}, "fresh"),
        ({"width": 400}, "smaller"),
    ],
)
def test_header_validation(make_block, header, message):
    block, _ = make_block(header=header)
    with pytest.raises(ValueError, match=message):
        PhotoshopBridge(block.name)


def test_block_without_settings_area_is_rejected(make_block):
    block, _ = make_block(extra=0)
    with pytest.raises(ValueError, match="smaller"):
        PhotoshopBridge(block.name)


def test_settings_json_round_trip_and_strict_validation():
    detection = DetectionSettings(threshold_sigma=7.5, min_distance=9, max_stars=100, sky_only=True)
    effects = EffectSettings()
    raw = encode_session_settings(detection, effects)
    assert len(raw) < SETTINGS_CAPACITY
    assert decode_session_settings(raw) == (detection, effects)
    good = json.loads(raw)
    for mutate in (
        lambda d: d.update(version=2),
        lambda d: d.update(extra=1),
        lambda d: d["detection"].update(method="bogus"),
        lambda d: d["detection"].update(threshold_sigma=99),
        lambda d: d["detection"].update(min_distance=5.0),
        lambda d: d["detection"].update(sky_only=1),
        lambda d: d["detection"].pop("method"),
        lambda d: d["detection"].update(unknown=1),
        lambda d: d["effects"]["settings"].pop("soft"),
    ):
        data = json.loads(json.dumps(good))
        mutate(data)
        with pytest.raises(ValueError):
            decode_session_settings(json.dumps(data).encode())
    for bad in (b"\xff\xfe", b"{", b"[]", b"null"):
        with pytest.raises(ValueError):
            decode_session_settings(bad)


def test_store_and_read_settings(make_block):
    block, _ = make_block()
    bridge = make_block.attach(block)
    detection = DetectionSettings(threshold_sigma=6.0)
    bridge.store_settings(detection, EffectSettings())
    length = struct.unpack_from("<I", block.buf, 36)[0]
    assert 0 < length <= SETTINGS_CAPACITY
    assert bridge.settings() == (detection, EffectSettings())
    assert bridge.status == 0


def test_code_above_white_is_rejected(make_block):
    samples = np.full((30, 40, 3), 100, np.uint16)
    samples[7, 3, 1] = 32769
    block, _ = make_block(depth=16, maximum=32768, samples=samples)
    bridge = make_block.attach(block)
    with pytest.raises(ValueError, match="exceeds"):
        bridge.read()
    with pytest.raises(ValueError, match="exceeds"):
        bridge.read_preview()


def test_status_is_set_only_after_the_last_band(make_block):
    block, samples = make_block()
    bridge = make_block.attach(block)
    frame = bridge.read()
    seen = []

    def bands():
        for top in range(0, 30, 10):
            seen.append(bridge.status)
            yield frame.pixels[top : top + 10]

    bridge.write(None, frame, bands())
    assert seen == [0, 0, 0] and bridge.status == 1


def test_aborted_write_never_reports_a_result(make_block):
    block, samples = make_block()
    bridge = make_block.attach(block)
    frame = bridge.read()

    def short():
        yield frame.pixels[:10]  # missing rows

    with pytest.raises(ValueError):
        bridge.write(None, frame, short())
    assert bridge.status == 0
    event = Event()
    event.set()
    with pytest.raises(Cancelled):
        bridge.write(None, frame, frame.pixels, cancel=event)
    assert bridge.status == 0
    bridge.cancel()
    assert bridge.status == 2 and bridge.exit_code == 2


def test_cancel_after_result_keeps_status(make_block):
    block, _ = make_block()
    bridge = make_block.attach(block)
    frame = bridge.read()
    bridge.write(None, frame, frame.pixels)
    bridge.cancel()
    assert bridge.status == 1


def test_preview_scale_and_size(make_block):
    samples = np.zeros((300, 4001, 3), np.uint8)
    samples[::10] = 200
    block, _ = make_block(width=4001, height=300, samples=samples)
    bridge = make_block.attach(block)
    frame, scale, width, height = bridge.read_preview(max_edge=1000)
    assert (width, height) == (4001, 300) and scale == 1 / 5
    assert frame.pixels.shape == (60, 801, 3)
    assert frame.pixels[2, 0, 0] > 0.5 and frame.pixels[1, 0, 0] == 0  # rows 0, 5, 10 sampled
    thin = np.full((2, 5000, 1), 9, np.uint8)
    block, _ = make_block(width=5000, height=2, channels=1, samples=thin)
    frame, scale, *_ = make_block.attach(block).read_preview(max_edge=1000)
    assert frame.pixels.shape[0] == 2 and scale == 1


def test_bridge_round_trip_writes_nothing_to_disk(make_block, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("TIFF access during a bridge round trip")

    monkeypatch.setattr(tifffile, "imwrite", forbidden)
    monkeypatch.setattr(tifffile, "TiffFile", forbidden)
    monkeypatch.chdir(tmp_path)
    temp = Path(tempfile.gettempdir())
    before = sorted(os.listdir(temp))
    block, samples = make_block()
    bridge = make_block.attach(block)
    frame = bridge.read()
    bridge.read_preview()
    bridge.write(tmp_path / "ignored.tif", frame, frame.pixels, source_path=tmp_path / "x")
    assert not list(tmp_path.iterdir())
    assert sorted(os.listdir(temp)) == before


def test_gui_apply_writes_result_and_cancel_marks_cancelled(qtbot, make_block, monkeypatch):
    from starlab.bootstrap import create_window
    from starlab.domain.stars import Star

    y, x = np.mgrid[:128, :160]
    gray = 0.02 + 0.6 * np.exp(-((x - 70.3) ** 2 + (y - 53.7) ** 2) / 7)
    samples = np.repeat(np.rint(gray[..., None] * 32768).astype(np.uint16), 3, axis=2)
    block, _ = make_block(width=160, height=128, depth=16, maximum=32768, samples=samples)
    bridge = make_block.attach(block)
    window = create_window(backend="cpu", bridge=bridge)
    qtbot.addWidget(window)
    catalog = (Star(1, 70.3, 53.7, 0.6, (0.6, 0.6, 0.6)),)
    monkeypatch.setattr(window.controller.services, "analyze", lambda *a, **k: catalog)
    window.show()
    controller = window.controller
    controller.open_layer()
    qtbot.waitUntil(
        lambda: controller.session.frame is not None and not controller.runner.busy,
        timeout=10000,
    )
    assert controller.source is None and "Photoshop" in window.windowTitle()
    assert window.open_button.isHidden() and window.export_button.text() == "Apply"
    assert not window.canvas.acceptDrops() and not window.image_view.acceptDrops()
    controller.open_path("other.tif")  # refused
    assert controller.session.frame.name == "Photoshop layer"
    window.export_button.click()
    qtbot.waitUntil(lambda: bridge.status == 1, timeout=10000)
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=10000)
    assert bridge.exit_code == 0 and not window.error_label.isVisible()
    assert np.any(_codes(block, samples) > samples)


def test_gui_cancel_marks_cancelled(qtbot, make_block):
    from starlab.bootstrap import create_window

    block, samples = make_block()
    bridge = make_block.attach(block)
    window = create_window(backend="cpu", bridge=bridge)
    qtbot.addWidget(window)
    window.show()
    window.controller.open_layer()
    qtbot.waitUntil(lambda: window.controller.session.frame is not None, timeout=10000)
    qtbot.waitUntil(lambda: not window.controller.runner.busy, timeout=10000)
    window.bridge_cancel_button.click()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=10000)
    bridge.cancel()  # __main__ does this after the event loop
    assert bridge.status == 2 and bridge.exit_code == 2
    assert np.array_equal(_codes(block, samples), samples)


def _stored(make_block, detection, **kwargs):
    """Make a repeat-mode block carrying the given settings, as the plug-in would."""
    raw = encode_session_settings(detection, EffectSettings())
    block, samples = make_block(header={"mode": 1, "length": len(raw)}, **kwargs)
    start = 64 + samples.nbytes
    block.buf[start : start + len(raw)] = raw
    return block, samples


def test_gui_apply_writes_settings(qtbot, make_block, monkeypatch):
    from starlab.bootstrap import create_window
    from starlab.domain.stars import Star

    block, _ = make_block()
    bridge = make_block.attach(block)
    window = create_window(backend="cpu", bridge=bridge)
    qtbot.addWidget(window)
    catalog = (Star(1, 20.0, 15.0, 0.6, (0.6, 0.6, 0.6)),)
    monkeypatch.setattr(window.controller.services, "analyze", lambda *a, **k: catalog)
    window.show()
    window.controller.open_layer()
    qtbot.waitUntil(
        lambda: window.controller.session.frame is not None and not window.controller.runner.busy,
        timeout=10000,
    )
    window.export_button.click()
    qtbot.waitUntil(lambda: bridge.status == 1, timeout=10000)
    detection, effects = bridge.settings()
    assert effects == window.controller.session.settings
    assert detection == window.controller.session.detection_settings


def test_repeat_runs_headless_end_to_end(make_block, monkeypatch, tmp_path):
    from starlab import __main__ as entry
    from starlab.domain.stars import Star

    y, x = np.mgrid[:30, :40]
    gray = 0.02 + 0.6 * np.exp(-((x - 20.3) ** 2 + (y - 15.7) ** 2) / 7)
    samples = np.repeat(np.rint(gray[..., None] * 255).astype(np.uint8), 3, axis=2)
    seen = []
    catalog = (Star(1, 20.3, 15.7, 0.6, (0.6, 0.6, 0.6)),)

    def analyze(self, frame, source, settings, cancel=None, *, sky_mask=None):
        seen.append(settings)
        return catalog

    from starlab.application.use_cases import ImageServices

    monkeypatch.setattr(ImageServices, "analyze", analyze)
    detection = DetectionSettings(threshold_sigma=8.0, min_distance=6)
    block, _ = _stored(make_block, detection, samples=samples)
    bridge = make_block.attach(block)
    assert bridge.mode == 1
    shown = []
    from starlab import bootstrap

    monkeypatch.setattr(bootstrap, "create_window", lambda *a, **k: shown.append(1))
    monkeypatch.setattr(entry.QMessageBox, "critical", lambda *a: shown.append("dialog"))
    app = type(
        "A",
        (),
        {
            "platformName": lambda self: "offscreen",
            "aboutToQuit": type("S", (), {"connect": lambda self, f: None})(),
        },
    )()
    assert entry._run_repeat(app, bridge, tmp_path, entry.logging.getLogger("starlab"), True) == 0
    assert seen == [detection] and not shown
    assert bridge.status == 1 and np.any(_codes(block, samples) > samples)


def test_repeat_failure_shows_dialog_and_reports_cancelled(make_block, monkeypatch, tmp_path):
    from starlab import __main__ as entry

    block, samples = _stored(make_block, DetectionSettings())
    bridge = make_block.attach(block)
    block.buf[64 + samples.nbytes : 64 + samples.nbytes + 4] = b"nope"  # corrupt settings
    shown = []
    monkeypatch.setattr(entry.QMessageBox, "critical", lambda *a: shown.append(a[2]))
    app = type(
        "A",
        (),
        {
            "platformName": lambda self: "offscreen",
            "aboutToQuit": type("S", (), {"connect": lambda self, f: None})(),
        },
    )()
    assert entry._run_repeat(app, bridge, tmp_path, entry.logging.getLogger("starlab"), True) == 2
    assert shown and bridge.status == 0
    assert np.array_equal(_codes(block, samples), samples)
