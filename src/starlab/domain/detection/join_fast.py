"""Native snapshot join without per-record NumPy temporary arrays."""

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from operator import itemgetter
from typing import overload

import numpy as np

from ..array_types import BoolArray, CandidateRecord, CoordinateKey, DoubleArray, FloatArray
from ..jit import njit
from .precision import round_to_float32

Y_KEY = itemgetter("y")


@njit(cache=True, nogil=True)
def merge_snapshot(snapshot: DoubleArray, incoming: DoubleArray) -> tuple[BoolArray, BoolArray]:
    """Merge N×8 raw rows in place; return incoming-append and old-update masks."""
    append_mask: BoolArray = np.ones(len(incoming), np.bool_)
    updated_mask: BoolArray = np.zeros(len(snapshot), np.bool_)
    positions: FloatArray = snapshot[:, :2].astype(np.float32)
    squared_sizes: FloatArray = np.empty(len(snapshot), np.float32)
    for snapshot_index in range(len(snapshot)):
        squared_sizes[snapshot_index] = round_to_float32(
            snapshot[snapshot_index, 2] * snapshot[snapshot_index, 2]
        )
    for incoming_index in range(len(incoming)):
        incoming_row: DoubleArray = incoming[incoming_index]
        x, y = np.float32(incoming_row[0]), np.float32(incoming_row[1])
        incoming_squared_size: float = round_to_float32(incoming_row[2] * incoming_row[2])
        for snapshot_index in range(len(snapshot)):
            dy: np.float32 = np.float32(positions[snapshot_index, 1] - y)
            maximum_squared_size: float | np.float32 = max(
                squared_sizes[snapshot_index], incoming_squared_size
            )
            if np.float32(dy * dy) >= maximum_squared_size:
                continue
            dx: np.float32 = np.float32(positions[snapshot_index, 0] - x)
            if np.float32(np.float32(dx * dx) + np.float32(dy * dy)) >= maximum_squared_size:
                continue
            append_mask[incoming_index] = False
            updated_mask[snapshot_index] = True
            existing_row: DoubleArray = snapshot[snapshot_index]
            observation_count: float = min(max(existing_row[7], 1) + max(incoming_row[7], 1), 255.0)
            weight: float = round_to_float32(max(incoming_row[7], 1) / observation_count)
            remaining: float = round_to_float32(1 - weight)
            for channel in range(7):
                existing_row[channel] = round_to_float32(
                    round_to_float32(existing_row[channel] * remaining)
                    + round_to_float32(incoming_row[channel] * weight)
                )
            existing_row[7] = observation_count
            positions[snapshot_index, 0], positions[snapshot_index, 1] = (
                existing_row[0],
                existing_row[1],
            )
            squared_sizes[snapshot_index] = round_to_float32(existing_row[2] * existing_row[2])
    return append_mask, updated_mask


def packed(rows: Sequence[CandidateRecord]) -> DoubleArray:
    """Pack records into N×8 float64 [x,y,size,brightness,r,g,b,count] rows."""
    return np.array(
        [
            (
                candidate_record["x"],
                candidate_record["y"],
                candidate_record["size"],
                candidate_record["brightness"],
                *candidate_record["rgb"],
                candidate_record["count"],
            )
            for candidate_record in rows
        ],
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
        return max((candidate_record["size"] for candidate_record in incoming), default=0.0)
    if not incoming:
        return maximum
    sorted_input: bool = maximum is not None
    previous_maximum: float = (
        maximum
        if maximum is not None
        else max(candidate_record["size"] for candidate_record in destination)
    )
    maximum = max(previous_maximum, max(candidate_record["size"] for candidate_record in incoming))
    minimum_y: float = min(candidate_record["y"] for candidate_record in incoming) - maximum - 1
    maximum_y: float = max(candidate_record["y"] for candidate_record in incoming) + maximum + 1
    minimum_x: float = min(candidate_record["x"] for candidate_record in incoming) - maximum - 1
    maximum_x: float = max(candidate_record["x"] for candidate_record in incoming) + maximum + 1
    candidates: list[CandidateRecord] = (
        destination[
            bisect_left(destination, minimum_y, key=Y_KEY) : bisect_right(
                destination, maximum_y, key=Y_KEY
            )
        ]
        if sorted_input
        else destination
    )
    snapshot: list[CandidateRecord] = [
        candidate_record
        for candidate_record in candidates
        if minimum_y <= candidate_record["y"] <= maximum_y
        and minimum_x <= candidate_record["x"] <= maximum_x
    ]
    snapshot_data: DoubleArray = packed(snapshot)
    append_mask, updated_mask = merge_snapshot(snapshot_data, packed(incoming))
    for snapshot_index in np.flatnonzero(updated_mask):
        record, values = snapshot[snapshot_index], snapshot_data[snapshot_index]
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
    destination.extend(row for row, keep in zip(incoming, append_mask) if keep)
    destination.sort(key=Y_KEY)
    return maximum
