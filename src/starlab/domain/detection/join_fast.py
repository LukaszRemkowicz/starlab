"""Native snapshot join without per-record NumPy temporary arrays."""

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from operator import itemgetter
from typing import overload

import numpy as np

from ..array_types import BoolArray, CandidateRecord, CoordinateKey, DoubleArray
from ..jit import njit

Y_KEY = itemgetter("y")


@njit(cache=True, inline="always")
def rounded(value: float) -> np.float64:
    """Round through float32 while retaining double-width tuple storage."""
    return np.float64(np.float32(value))


@njit(cache=True, nogil=True)
def merge_snapshot(snapshot: DoubleArray, incoming: DoubleArray) -> tuple[BoolArray, BoolArray]:
    """Merge N×8 raw rows in place; return incoming-append and old-update masks."""
    append = np.ones(len(incoming), np.bool_)
    updated = np.zeros(len(snapshot), np.bool_)
    xy = snapshot[:, :2].astype(np.float32)
    sizes = np.empty(len(snapshot), np.float32)
    for i in range(len(snapshot)):
        sizes[i] = rounded(snapshot[i, 2] * snapshot[i, 2])
    for row in range(len(incoming)):
        new = incoming[row]
        x, y = np.float32(new[0]), np.float32(new[1])
        size = rounded(new[2] * new[2])
        for i in range(len(snapshot)):
            dy = np.float32(xy[i, 1] - y)
            radius = max(sizes[i], size)
            if np.float32(dy * dy) >= radius:
                continue
            dx = np.float32(xy[i, 0] - x)
            if np.float32(np.float32(dx * dx) + np.float32(dy * dy)) >= radius:
                continue
            append[row] = False
            updated[i] = True
            old = snapshot[i]
            count = min(max(old[7], 1) + max(new[7], 1), 255.0)
            weight = rounded(max(new[7], 1) / count)
            remaining = rounded(1 - weight)
            for channel in range(7):
                old[channel] = rounded(
                    rounded(old[channel] * remaining) + rounded(new[channel] * weight)
                )
            old[7] = count
            xy[i, 0], xy[i, 1] = old[0], old[1]
            sizes[i] = rounded(old[2] * old[2])
    return append, updated


def packed(rows: Sequence[CandidateRecord]) -> DoubleArray:
    """Pack records into N×8 float64 [x,y,size,brightness,r,g,b,count] rows."""
    return np.array(
        [(r["x"], r["y"], r["size"], r["brightness"], *r["rgb"], r["count"]) for r in rows],
        np.float64,
    ).reshape(-1, 8)


@overload
def join_catalogs_fast(
    destination: list[CandidateRecord], incoming: list[CandidateRecord], maximum: float
) -> float:
    """Merge with a known running maximum and return its updated value."""
    ...


@overload
def join_catalogs_fast(
    destination: list[CandidateRecord], incoming: list[CandidateRecord], maximum: None = None
) -> float | None:
    """Merge without a running maximum; an empty incoming list may return None."""
    ...


def join_catalogs_fast(
    destination: list[CandidateRecord],
    incoming: list[CandidateRecord],
    maximum: float | None = None,
) -> float | None:
    """Optional running maximum enables the already-sorted streaming path."""
    incoming.sort(key=Y_KEY)
    if not destination:
        destination.extend(incoming)
        return max((r["size"] for r in incoming), default=0.0)
    if not incoming:
        return maximum
    sorted_input = maximum is not None
    previous = maximum if maximum is not None else max(r["size"] for r in destination)
    maximum = max(previous, max(r["size"] for r in incoming))
    lower = min(r["y"] for r in incoming) - maximum - 1
    upper = max(r["y"] for r in incoming) + maximum + 1
    left = min(r["x"] for r in incoming) - maximum - 1
    right = max(r["x"] for r in incoming) + maximum + 1
    candidates = (
        destination[
            bisect_left(destination, lower, key=Y_KEY) : bisect_right(destination, upper, key=Y_KEY)
        ]
        if sorted_input
        else destination
    )
    snapshot = [r for r in candidates if lower <= r["y"] <= upper and left <= r["x"] <= right]
    data = packed(snapshot)
    append, updated = merge_snapshot(data, packed(incoming))
    for snapshot_index in np.flatnonzero(updated):
        record, values = snapshot[snapshot_index], data[snapshot_index]
        fields: tuple[tuple[CoordinateKey, int], ...] = (
            ("x", 0),
            ("y", 1),
            ("size", 2),
            ("brightness", 3),
        )
        for key, index in fields:
            record[key] = float(values[index])
        record["rgb"] = values[4:7].tolist()
        record["count"] = int(values[7])
        maximum = max(maximum, record["size"])
    destination.extend(row for row, keep in zip(incoming, append) if keep)
    destination.sort(key=Y_KEY)
    return maximum
