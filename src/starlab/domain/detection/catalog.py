"""Merge automatic source catalogs against the preceding destination snapshot.

New records are matched only against the destination snapshot that preceded
this tile. Every overlapping old record is updated, as in the native loop.
Vectorized distance tests avoid scanning a panorama-wide Python list per seed.
"""

import numpy as np
from numpy.typing import NDArray

from ..array_types import CandidateRecord, CoordinateKey, FloatArray
from .precision import round_to_float32


def blend_record(
    existing_record: CandidateRecord, incoming_record: CandidateRecord, count: int, weight: float
) -> None:
    """Apply a resolved observation count/weight with the exact field-rounding order."""
    remaining: float = round_to_float32(1 - weight)
    fields: tuple[CoordinateKey, ...] = ("x", "y", "size", "brightness")
    for key in fields:
        existing_record[key] = round_to_float32(
            round_to_float32(existing_record[key] * remaining)
            + round_to_float32(incoming_record[key] * weight)
        )
    existing_record["rgb"] = [
        round_to_float32(
            round_to_float32(existing_channel * remaining)
            + round_to_float32(incoming_channel * weight)
        )
        for existing_channel, incoming_channel in zip(
            existing_record["rgb"], incoming_record["rgb"]
        )
    ]
    existing_record["count"] = count


def combine_records(existing_record: CandidateRecord, incoming_record: CandidateRecord) -> None:
    """Blend a whole incoming record using its capped native observation count."""
    count: int = min(max(existing_record["count"], 1) + max(incoming_record["count"], 1), 255)
    blend_record(
        existing_record,
        incoming_record,
        count,
        round_to_float32(max(incoming_record["count"], 1) / count),
    )


def join_catalogs(destination: list[CandidateRecord], incoming: list[CandidateRecord]) -> None:
    """Join against the preceding destination snapshot, updating every overlapping old row."""
    incoming.sort(key=lambda incoming_record: incoming_record["y"])
    if not destination:
        destination.extend(incoming)
        return
    if not incoming:
        return
    # The Y window is only a conservative acceleration of the native overlap
    # test. Keep indices fixed while rows move through weighted combination.
    maximum_size: float = max(
        max(candidate_record["size"] for candidate_record in destination),
        max(candidate_record["size"] for candidate_record in incoming),
    )
    minimum_y: float = (
        min(candidate_record["y"] for candidate_record in incoming) - maximum_size - 1
    )
    maximum_y: float = (
        max(candidate_record["y"] for candidate_record in incoming) + maximum_size + 1
    )
    snapshot: list[CandidateRecord] = [
        candidate_record
        for candidate_record in destination
        if minimum_y <= candidate_record["y"] <= maximum_y
    ]
    positions: FloatArray = np.array(
        [(candidate_record["x"], candidate_record["y"]) for candidate_record in snapshot],
        np.float32,
    ).reshape(-1, 2)
    squared_sizes: FloatArray = np.array(
        [
            round_to_float32(candidate_record["size"] * candidate_record["size"])
            for candidate_record in snapshot
        ],
        np.float32,
    )
    for incoming_record in incoming:
        delta: FloatArray = positions - np.array(
            [incoming_record["x"], incoming_record["y"]], np.float32
        )
        squared_offsets: FloatArray = delta * delta
        matches: NDArray[np.intp] = np.flatnonzero(
            squared_offsets[:, 0] + squared_offsets[:, 1]
            < np.maximum(
                squared_sizes, round_to_float32(incoming_record["size"] * incoming_record["size"])
            )
        )
        if not len(matches):
            destination.append(incoming_record)
        for index in matches:
            existing_record: CandidateRecord = snapshot[index]
            combine_records(existing_record, incoming_record)
            positions[index] = (existing_record["x"], existing_record["y"])
            squared_sizes[index] = round_to_float32(
                existing_record["size"] * existing_record["size"]
            )
    destination.sort(key=lambda incoming_record: incoming_record["y"])
