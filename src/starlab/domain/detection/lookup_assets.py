"""Integrity-checked, immutable numerical assets used by analysis proxies."""

import hashlib
import io
import json
from pathlib import Path
from threading import Lock

import numpy as np
from numpy.typing import NDArray

from ..array_types import ByteArray

# Pin the approved archive identity; recalculating the expected digest would accept changed data.
ANALYSIS_ARCHIVE_SHA256 = "9bd9d8c4798fd340c8356422eff55f2bbfb1e0bf6546c0f56868caed9e8e9a61"
COLOR_ARCHIVE_SHA256 = "6384f6ae50403911b82cceb9781f1fa099880426a44b0b120798704ff9a3dfa0"
COLOR_TABLE_SHA256 = "9cf814a133376c04853756c212d9524c701e0236ed6e79a8123dfe53a21b5a9e"


def load_analysis_tables(path: Path | None = None) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Load fixed READ/WRITE curves, rejecting changed archives before use.

    READ has 256 float32 analysis levels; WRITE has 4096 uint8 encoding levels.
    The stored READ values are compatibility constants, not a freshly evaluated
    power function. Returned arrays own their data and prohibit ordinary writes.
    No photograph or calculated catalogue is read by this operation.
    """
    path = path or Path(__file__).with_name("analysis_luts.npz")
    archive: bytes = path.read_bytes()
    if hashlib.sha256(archive).hexdigest() != ANALYSIS_ARCHIVE_SHA256:
        raise ValueError("Analysis lookup archive does not match its recorded identity.")
    with np.load(io.BytesIO(archive), allow_pickle=False) as tables:
        read: NDArray[np.float32] = tables["read"].copy()
        write: NDArray[np.uint8] = tables["write"].copy()
    read.flags.writeable = write.flags.writeable = False
    return read, write


def load_color_cube(path: Path | None = None) -> NDArray[np.uint8]:
    """Reconstruct the universal RGB8 transform after checking its data identity.

    The archive stores uint8 residuals and three channel-axis baselines. Addition
    intentionally wraps modulo256. The result has shape(256,256,256,3), remains
    read-only, and consumes48MiB regardless of photograph size. JSON describes
    the transform; it is not interchangeable with the compressed numeric data.
    """
    path = path or Path(__file__).with_name("full_lcms_lut.npz")
    metadata: dict[str, str | list[int]] = json.loads(
        path.with_suffix(".json").read_text(encoding="utf-8")
    )
    if metadata.get("shape") != [256, 256, 256, 3] or metadata.get("dtype") != "uint8":
        raise ValueError("Color-transform metadata has unsupported dimensions or dtype.")
    archive: bytes = path.read_bytes()
    if hashlib.sha256(archive).hexdigest() != COLOR_ARCHIVE_SHA256:
        raise ValueError("Color-transform archive does not match its recorded identity.")
    with np.load(io.BytesIO(archive), allow_pickle=False) as stored:
        cube: NDArray[np.uint8] = stored["delta"]
        baselines: tuple[ByteArray, ...] = tuple(
            stored[channel] for channel in ("red", "green", "blue")
        )
    if (
        cube.shape != (256, 256, 256, 3)
        or cube.dtype != np.uint8
        or any(values.shape != (256,) or values.dtype != np.uint8 for values in baselines)
    ):
        raise ValueError("Color-transform arrays have unsupported dimensions or dtype.")
    np.add(cube[..., 0], baselines[0][:, None, None], out=cube[..., 0])
    np.add(cube[..., 1], baselines[1][None, :, None], out=cube[..., 1])
    np.add(cube[..., 2], baselines[2][None, None, :], out=cube[..., 2])
    if hashlib.sha256(memoryview(cube)).hexdigest() != COLOR_TABLE_SHA256:
        raise ValueError("Universal color cube does not match its recorded identity.")
    cube.flags.writeable = False
    return cube


class ColorTransform:
    """Own one immutable universal cube and serialize its initial reconstruction.

    This object contains fixed transform data, never photograph results. Failed
    loads publish no state and can be retried; successful users share one cube.
    """

    def __init__(self) -> None:
        """Prepare empty transform storage and its first-use synchronization lock."""
        self._table: NDArray[np.uint8] | None = None
        self._lock = Lock()

    def table(self) -> NDArray[np.uint8]:
        """Return the validated RGB8 cube, reconstructing it once under the lock."""
        with self._lock:
            table: ByteArray | None = self._table
            if table is None:
                table = load_color_cube()
                self._table = table
            return table
