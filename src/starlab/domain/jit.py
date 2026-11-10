"""Preserve Python signatures across Numba's dynamically typed dispatcher boundary."""

from collections.abc import Callable
from typing import Literal, ParamSpec, TypeVar, cast

import numba

P = ParamSpec("P")
R = TypeVar("R")


def njit(
    *,
    cache: bool = False,
    nogil: bool = False,
    parallel: bool = False,
    inline: Literal["never", "always"] = "never",
    error_model: Literal["python", "numpy"] = "python",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Apply Numba unchanged while exposing the wrapped signature to static checks.

    The runtime object remains Numba's CPUDispatcher, including its compilation
    and cache behavior. The cast bridges that external dynamic decorator; it
    does not hide parameter or result errors in the numerical function body.
    """
    return cast(
        Callable[[Callable[P, R]], Callable[P, R]],
        numba.njit(
            cache=cache,
            nogil=nogil,
            parallel=parallel,
            inline=inline,
            error_model=error_model,
        ),
    )
