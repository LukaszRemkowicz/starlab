"""IEEE rounding anchors for reference and compiled use of the same precision policy."""

import math

import pytest

from starlab.domain.detection.precision import round_to_float32
from starlab.domain.jit import njit


@njit(cache=True)
def _round_in_kernel(value: float) -> float:
    return round_to_float32(value)


@pytest.mark.parametrize("operation", [round_to_float32, _round_in_kernel], ids=["python", "numba"])
@pytest.mark.parametrize(
    "value, expected",
    [
        (1 + 2**-24, 1.0),
        (1 + 3 * 2**-24, 1 + 2**-22),
        (2**24 + 1, float(2**24)),
        (2**24 + 3, float(2**24 + 4)),
        (2**-150, 0.0),
        (3 * 2**-150, 2**-148),
        (-0.0, -0.0),
        (math.inf, math.inf),
        (-math.inf, -math.inf),
    ],
)
def test_float32_rounding_preserves_ties_subnormals_and_signed_zero(operation, value, expected):
    result = operation(value)
    assert result == expected
    assert math.copysign(1.0, result) == math.copysign(1.0, expected)


@pytest.mark.parametrize("operation", [round_to_float32, _round_in_kernel], ids=["python", "numba"])
def test_float32_rounding_preserves_nan(operation):
    assert math.isnan(operation(math.nan))
