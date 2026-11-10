"""Universal transform ownership under concurrent first use and load failure."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import numpy as np
import pytest

from starlab.domain.detection import lookup_assets


def test_concurrent_first_use_initializes_one_shared_immutable_transform(monkeypatch):
    start = Barrier(4)
    entered, release = Event(), Event()
    calls = []
    expected = np.zeros((2, 2, 2, 3), np.uint8)
    expected.flags.writeable = False

    def load():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return expected

    monkeypatch.setattr(lookup_assets, "load_color_cube", load)
    owner = lookup_assets.ColorTransform()

    def acquire():
        start.wait(5)
        return owner.table()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(acquire) for _ in range(4)]
        try:
            assert entered.wait(5)
        finally:
            release.set()
        results = [future.result(5) for future in futures]
    assert calls == [1]
    assert all(result is expected for result in results)
    assert not results[0].flags.writeable


def test_failed_initialization_can_retry_without_retaining_partial_state(monkeypatch):
    calls = []
    expected = np.zeros((2, 2, 2, 3), np.uint8)

    def load():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("Corrupt transform")
        return expected

    monkeypatch.setattr(lookup_assets, "load_color_cube", load)
    owner = lookup_assets.ColorTransform()
    with pytest.raises(ValueError, match="Corrupt transform"):
        owner.table()
    assert owner.table() is expected
    assert owner.table() is expected
    assert len(calls) == 2
