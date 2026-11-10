"""Integrity-checked, immutable numerical assets used by analysis proxies."""

import hashlib
import io
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ANALYSIS_ARCHIVE_SHA256 = "9bd9d8c4798fd340c8356422eff55f2bbfb1e0bf6546c0f56868caed9e8e9a61"


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
        read = tables["read"].copy()
        write = tables["write"].copy()
    read.flags.writeable = write.flags.writeable = False
    return read, write
