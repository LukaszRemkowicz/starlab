from threading import Event

import numpy as np
import pytest

from starlab.domain.imaging import Cancelled
from starlab.domain.sky_mask import FOREGROUND, SKY, UNKNOWN, SkyMask, build_sky_mask


def test_mask_samples_native_pixels_and_preserves_unknown_horizon():
    mask = SkyMask(2, np.array([[SKY, UNKNOWN], [FOREGROUND, FOREGROUND]], np.uint8), (4, 4))
    assert mask.allowed_at(0, 0) and mask.allowed_at(3, 0)
    assert not mask.allowed_at(1, 3)
    np.testing.assert_array_equal(
        mask.keep(np.array([[0, 0], [3, 0], [0, 3], [np.nan, 1]])), [True, True, False, False]
    )
    np.testing.assert_array_equal(
        mask.region((1, 1, 4, 4), sampling=2), [[True, True], [False, False]]
    )
    assert not mask.classes.flags.writeable


@pytest.mark.parametrize("label", [0.5, 1.5])
def test_mask_rejects_fractional_class_labels(label):
    with pytest.raises(ValueError):
        SkyMask(1, np.array([[label]]), (1, 1))


def test_mask_rejects_fractional_cell_stride():
    with pytest.raises(ValueError):
        SkyMask(1.5, np.ones((2, 2), np.uint8), (3, 3))


def test_mask_does_not_allow_points_outside_the_original_image():
    mask = SkyMask(2, np.ones((2, 3), np.uint8), (3, 5))
    points = [(-0.01, 0), (0, -0.01), (5, 0), (0, 3), (4.99, 2.99), (np.nan, 1)]
    np.testing.assert_array_equal(mask.keep(points), [False, False, False, False, True, False])


def test_mask_builder_uses_encoded_block_means_and_bounds_strip_memory():
    rows = []
    image = np.arange(12 * 20 * 3, dtype=np.uint16).reshape(12, 20, 3)

    class Network:
        tile = 4
        overlap = 1

        def sky_logits(self, tile):
            assert tile.dtype == np.float32 and tile.shape == (4, 4, 3)
            return np.full((4, 4), 2, np.float32)

    def read(top, bottom):
        rows.append((top, bottom))
        return image[top:bottom]

    mask = build_sky_mask(image.shape[:2], read, Network())
    assert mask.cell_px == 2 and mask.classes.shape == (6, 10)
    assert np.all(mask.classes == SKY)
    assert max(bottom - top for top, bottom in rows) <= 512


def test_model_classification_and_edge_padding_are_explicit():
    class Network:
        tile = 8
        overlap = 2

        def sky_logits(self, tile):
            output = np.full(tile.shape[:2], 10, np.float32)
            output[2:4] = 0
            output[4:] = -10
            return output

    mask = build_sky_mask(
        (11, 13), lambda top, bottom: np.ones((bottom - top, 13, 1), np.uint16), Network()
    )
    assert mask.classes.shape == (6, 7)
    assert set(np.unique(mask.classes)) == {FOREGROUND, SKY, UNKNOWN}
    assert mask.allowed_at(0, 4)  # UNKNOWN remains usable near the horizon.
    assert not mask.allowed_at(0, 10)


def test_mask_builder_cancels_between_tiles():
    cancel = Event()

    class Network:
        tile = 4
        overlap = 1

        def sky_logits(self, tile):
            cancel.set()
            return np.zeros(tile.shape[:2], np.float32)

    with pytest.raises(Cancelled):
        build_sky_mask(
            (20, 20),
            lambda top, bottom: np.ones((bottom - top, 20, 3), np.uint16),
            Network(),
            cancel,
        )


def test_batched_overlap_averaging_preserves_classes_and_partial_batch():
    class Single:
        tile = 4
        overlap = 1

        def sky_logits(self, tile):
            return (tile[..., 0] - 0.5) * 3

    class Batched(Single):
        batch_size = 3

        def sky_logits(self, tile):
            raise AssertionError("batch-capable segmenter must receive batches")

        def sky_logits_batch(self, tiles):
            return np.stack([Single.sky_logits(self, tile) for tile in tiles])

    image = np.arange(23 * 25 * 3, dtype=np.uint16).reshape(23, 25, 3)

    def read(top, bottom):
        return image[top:bottom]

    expected = build_sky_mask(image.shape[:2], read, Single())
    actual = build_sky_mask(image.shape[:2], read, Batched())
    np.testing.assert_array_equal(actual.classes, expected.classes)


def test_logit_classification_preserves_sigmoid_decisions_at_all_boundary_neighbors():
    from scipy.special import expit

    from starlab.domain.sky_mask import _classify_logits

    center = np.float32(np.log(1.5)).view(np.uint32)
    neighbors = (np.int64(center) + np.arange(-64, 65)).astype(np.uint32).view(np.float32)
    rng = np.random.default_rng(44)
    random = rng.integers(0, 0x7F800000, 10000, dtype=np.uint32).view(np.float32)
    logits = np.concatenate((neighbors, -neighbors, random, -random, [0, np.inf, -np.inf])).astype(
        np.float32
    )
    probability = expit(logits.astype(np.float64))
    expected = np.full(logits.shape, UNKNOWN, np.uint8)
    expected[probability >= 0.6] = SKY
    expected[probability <= 0.4] = FOREGROUND
    np.testing.assert_array_equal(_classify_logits(logits), expected)


def test_mask_builder_rejects_nonfinite_model_logits():
    class Network:
        tile = 4
        overlap = 0

        def sky_logits(self, tile):
            return np.full((4, 4), np.nan, np.float32)

    with pytest.raises(ValueError, match="invalid logits"):
        build_sky_mask((8, 8), lambda top, bottom: np.zeros((bottom - top, 8, 3)), Network())
