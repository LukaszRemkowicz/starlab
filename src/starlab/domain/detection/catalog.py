"""Recovered automatic-record catalog join (3EF30 / 3EA70).

New records are matched only against the destination snapshot that preceded
this tile. Every overlapping old record is updated, as in the native loop.
Vectorized distance tests avoid scanning a panorama-wide Python list per seed.
"""

import numpy as np

from .rays import f


def combine_records(old, new):
    count = min(max(old["count"], 1) + max(new["count"], 1), 255)
    weight = f(max(new["count"], 1) / count)
    remaining = f(1 - weight)
    for key in ("x", "y", "size", "brightness"):
        old[key] = f(f(old[key] * remaining) + f(new[key] * weight))
    old["rgb"] = [f(f(a * remaining) + f(b * weight)) for a, b in zip(old["rgb"], new["rgb"])]
    old["count"] = count


def join_catalogs(destination, incoming):
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
