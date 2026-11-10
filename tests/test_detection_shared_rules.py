"""Frozen merge/geometry values and boundary decisions for shared detector rules."""

from copy import deepcopy

import numpy as np
import pytest

from starlab.domain.detection import candidate, catalog, compiled, numeric, rays
from starlab.domain.detection.records import CompiledStats


def _sources():
    old = dict(x=10.0, y=20.0, size=2.0, brightness=0.25, rgb=[0.25, 0.5, 0.75], count=254)
    new = dict(x=10.25, y=20.25, size=3.0, brightness=0.75, rgb=[0.8, 0.6, 0.4], count=99)
    return old, new


def _measurements(row):
    return [row["x"], row["y"], row["size"], row["brightness"], *row["rgb"], row["count"]]


def test_fresh_candidate_adds_one_observation_even_when_incoming_count_is_larger():
    old, new = _sources()
    rows = [old]
    assert candidate.merge_candidate(rows, deepcopy(new), 2) == old["size"]
    np.testing.assert_array_equal(
        _measurements(old),
        [
            10.000980377197266,
            20.000980377197266,
            2.0039215087890625,
            0.25196078419685364,
            0.2521568536758423,
            0.5003921389579773,
            0.7486274242401123,
            255,
        ],
    )
    assert len(rows) == 1 and new["count"] == 99


def test_catalog_merge_keeps_incoming_observation_weight_at_saturation():
    old, new = _sources()
    catalog.combine_records(old, new)
    np.testing.assert_array_equal(
        _measurements(old),
        [
            10.097058296203613,
            20.097057342529297,
            2.388235092163086,
            0.44411763548851013,
            0.46352940797805786,
            0.5388234853744507,
            0.6141176223754883,
            255,
        ],
    )


@pytest.mark.parametrize("backend", ["python", "numba"])
@pytest.mark.parametrize(
    "radius, success, expected_center, expected_radius",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            True,
            (20.25, 28.197715759277344),
            [5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ),
        ([2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0], True, (20.25, 30.75), [2.0] * 8),
        (
            [1.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            False,
            (20.25, 30.75),
            [1.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ],
)
def test_radius_repair_preserves_frozen_center_values_and_input(
    backend, radius, success, expected_center, expected_radius
):
    original = np.array(radius, np.float64) if backend == "numba" else list(radius)
    operation = compiled.repair if backend == "numba" else rays.repair_recenter
    actual_success, center, repaired = operation((20.25, 30.75), original)
    assert actual_success == success and center == expected_center
    np.testing.assert_array_equal(repaired, expected_radius)
    np.testing.assert_array_equal(original, radius)


@pytest.mark.parametrize("stats_type", [numeric.Stats, CompiledStats])
@pytest.mark.parametrize(
    "size, direction, background, inner, edge, missing, expected",
    [
        (4.0, 0, 0.0, 0.6, 0.5, False, (3.5, -1, False)),
        (4.0, -1, 0.0, 1.0, 1.0, False, (4.5, -1, True)),
        (4.0, 0, 0.0, 0.8, 0.5, False, (4.0, 0, False)),
        (128.0, 0, 0.0, 1.0, 1.0, False, (128.0, 0, False)),
        (4.0, 0, 1.0, 1.0, 1.0, False, (4.0, 0, False)),
        (4.0, 0, 0.0, 0.6, 0.5, True, (4.0, 0, True)),
    ],
)
def test_large_radius_decision_handles_limits_missing_support_and_growth_reversal(
    stats_type, size, direction, background, inner, edge, missing, expected
):
    outer = None if missing else stats_type(background, 0.0, 1.0, 0.0, 0.0)
    actual = numeric.large_size_step(
        size,
        0.5,
        direction,
        outer,
        stats_type(0.0, 0.0, 1.0, 0.0, inner),
        stats_type(0.0, 0.0, 1.0, 0.0, edge),
    )
    assert actual == expected
