"""Tile buffers: zero-filled, independent mappings; labeling picks a wide enough dtype."""

import numpy as np
from scipy import ndimage

from starlab.domain.detection.buffers import anonymous_array
from starlab.domain.detection.candidate import Image
from starlab.domain.detection.peaks import _component_centers


def test_anonymous_array_is_zeroed_and_independent() -> None:
    first = anonymous_array((5, 7, 3), np.float32)
    first[:] = 3
    second = anonymous_array((5, 7, 3), np.float32)
    assert first.shape == second.shape == (5, 7, 3) and not second.any()
    assert not np.shares_memory(first, second)
    assert anonymous_array((0, 4), np.int32).shape == (0, 4)


def test_component_centers_match_default_labels_for_both_label_widths() -> None:
    for step in (40, 2):  # 100 components fit 16-bit labels; 40000 need 32-bit labels
        selected = np.zeros((400, 400), bool)
        selected[::step, ::step] = True
        image = Image(np.zeros((400, 400, 3), np.float32))
        labels, _ = ndimage.label(selected)
        expected = [
            ((r[0].start + r[0].stop - 1) // 2, (r[1].start + r[1].stop - 1) // 2)
            for r in ndimage.find_objects(labels)
        ]
        assert _component_centers(selected, image, (0, 0, 400, 400)) == expected
        image.clear_cache()
