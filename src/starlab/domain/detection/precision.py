"""Explicit float32 rounding shared by Python and Numba detector calculations."""

import struct
from collections.abc import Callable
from typing import cast

import numpy as np
from numba.extending import overload


def round_to_float32(value: float) -> float:
    """Round one intermediate through IEEE float32 and return its Python scalar."""
    packed_value: bytes = struct.pack("<f", value)
    rounded_value: float = cast(float, struct.unpack("<f", packed_value)[0])
    return rounded_value


# Numba compares overload/implementation Parameters, including annotations.
# Both value parameters must stay unannotated despite their different runtime roles.
@overload(round_to_float32)
def _round_to_float32_overload(value) -> Callable[[float], np.float64]:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Provide the float32-rounding overload for a Numba scalar type."""

    def round_scalar(value) -> np.float64:  # type: ignore[no-untyped-def]  # noqa: ANN001
        """Round a compiled scalar to float32 and widen its storage to float64."""
        return np.float64(np.float32(value))

    return round_scalar
