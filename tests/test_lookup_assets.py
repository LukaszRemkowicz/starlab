"""Packaged numerical tables are immutable universal data, not image results."""

import numpy as np
import pytest


def test_analysis_tables_have_exact_shapes_and_cannot_be_modified():
    from starlab.domain.detection.lookup_assets import load_analysis_tables

    read, write = load_analysis_tables()
    assert read.shape == (256,) and read.dtype == np.float32
    assert write.shape == (4096,) and write.dtype == np.uint8
    assert read[0] == 0 and read[-1] == 1
    expected_write = np.rint((np.arange(4096) / 4095) ** (1 / 2.4) * 255).astype(np.uint8)
    np.testing.assert_array_equal(write, expected_write)
    for table in (read, write):
        with pytest.raises(ValueError, match="read-only"):
            table[1] = 0


def test_changed_analysis_archive_fails_before_arrays_are_used(tmp_path):
    from starlab.domain.detection.lookup_assets import load_analysis_tables

    path = tmp_path / "changed.npz"
    np.savez(path, read=np.zeros(1), write=np.zeros(1))
    with pytest.raises(ValueError, match="Analysis lookup archive"):
        load_analysis_tables(path)


def test_color_transform_archive_integrity_is_checked_before_decompression(tmp_path):
    import json
    from pathlib import Path

    from starlab.domain.detection.lookup_assets import load_color_cube

    original = Path("src/starlab/domain/detection/full_lcms_lut.json")
    path = tmp_path / "cube.npz"
    path.write_bytes(b"not a numpy archive")
    path.with_suffix(".json").write_text(original.read_text())
    with pytest.raises(ValueError, match="Color-transform archive"):
        load_color_cube(path)
    metadata = json.loads(original.read_text())
    metadata["shape"] = [256, 256, 256, 4]
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Color-transform metadata"):
        load_color_cube(path)
