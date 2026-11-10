"""Recovered automatic-record catalog join (3EF30 / 3EA70).

New records are matched only against the destination snapshot that preceded
this tile. Every overlapping old record is updated, as in the native loop.
Vectorized distance tests avoid scanning a panorama-wide Python list per seed.
"""

import numpy as np

from ..array_types import CandidateRecord, CoordinateKey
from .rays import f


def blend_record(old: CandidateRecord, new: CandidateRecord, count: int, weight: float) -> None:
    """Apply a resolved observation count/weight with the exact field-rounding order."""
    remaining = f(1 - weight)
    fields: tuple[CoordinateKey, ...] = ("x", "y", "size", "brightness")
    for key in fields:
        old[key] = f(f(old[key] * remaining) + f(new[key] * weight))
    old["rgb"] = [f(f(a * remaining) + f(b * weight)) for a, b in zip(old["rgb"], new["rgb"])]
    old["count"] = count


def combine_records(old: CandidateRecord, new: CandidateRecord) -> None:
    """Blend a whole incoming record using its capped native observation count."""
    count = min(max(old["count"], 1) + max(new["count"], 1), 255)
    blend_record(old, new, count, f(max(new["count"], 1) / count))


def join_catalogs(destination: list[CandidateRecord], incoming: list[CandidateRecord]) -> None:
    """Join against the preceding destination snapshot, updating every overlapping old row."""
    incoming.sort(key=lambda row: row["y"])
    if not destination:
        destination.extend(incoming)
        return
    if not incoming:
        return
    # The Y window is only a conservative acceleration of the native overlap
    # test. Keep indices fixed while rows move through weighted combination.
    maximum = max(max(r["size"] for r in destination), max(r["size"] for r in incoming))
    lower = min(r["y"] for r in incoming) - maximum - 1
    upper = max(r["y"] for r in incoming) + maximum + 1
    snapshot = [r for r in destination if lower <= r["y"] <= upper]
    xy = np.array([(r["x"], r["y"]) for r in snapshot], np.float32).reshape(-1, 2)
    sizes = np.array([f(r["size"] * r["size"]) for r in snapshot], np.float32)
    for row in incoming:
        delta = xy - np.array([row["x"], row["y"]], np.float32)
        squared = delta * delta
        matches = np.flatnonzero(
            squared[:, 0] + squared[:, 1] < np.maximum(sizes, f(row["size"] * row["size"]))
        )
        if not len(matches):
            destination.append(row)
        for index in matches:
            old = snapshot[index]
            combine_records(old, row)
            xy[index] = (old["x"], old["y"])
            sizes[index] = f(old["size"] * old["size"])
    destination.sort(key=lambda row: row["y"])
