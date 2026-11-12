from copy import deepcopy

import numpy as np

from starlab.domain.detection.catalog import join_catalogs


def test_compiled_join_preserves_native_snapshot_and_all_overlaps():
    from starlab.domain.detection.join_fast import join_catalogs_fast

    rng = np.random.default_rng(9)

    def rows(count):
        return [
            dict(
                x=float(rng.uniform(0, 64)),
                y=float(rng.uniform(0, 64)),
                size=float(rng.uniform(0.25, 6)),
                brightness=float(rng.uniform(0.1, 1)),
                rgb=rng.random(3).tolist(),
                count=int(rng.integers(1, 256)),
            )
            for _ in range(count)
        ]

    expected = rows(160)
    actual = deepcopy(expected)
    for incoming in (rows(180), [], rows(300)):
        join_catalogs(expected, deepcopy(incoming))
        join_catalogs_fast(actual, deepcopy(incoming))
        assert actual == expected

    expected, actual = [], []
    maximum = 0.0
    for _ in range(4):
        incoming = rows(150)
        join_catalogs(expected, deepcopy(incoming))
        maximum = join_catalogs_fast(actual, deepcopy(incoming), maximum)
        assert actual == expected
