import numpy as np

from starlab.domain import imaging


def test_display_transfer_is_exact_without_repeating_gamma_per_pixel(monkeypatch):
    rng = np.random.default_rng(71)
    values = rng.uniform(-0.01, 1.1, (512, 1024, 3)).astype(np.float32)
    expected = np.rint(np.clip(imaging.linear_to_srgb(values), 0, 1) * 255).astype(np.uint8)
    imaging.display_pixels(np.zeros((2, 2, 3), np.float32), "linear-srgb")
    original = imaging.linear_to_srgb
    sizes = []

    def observe(data):
        sizes.append(data.size)
        return original(data)

    monkeypatch.setattr(imaging, "linear_to_srgb", observe)
    actual = imaging.display_pixels(values, "linear-srgb")
    np.testing.assert_array_equal(actual, expected)
    assert max(sizes, default=0) < values.size // 4


def test_transfer_preserves_all_uint16_rgb_codes_and_rounding_boundaries():
    linear = imaging.srgb_to_linear(np.arange(65536, dtype=np.float32) / 65535)
    edges = imaging.srgb_to_linear((np.arange(255, dtype=np.float32) + 0.5) / 255)
    bits = edges.view(np.uint32)
    around = np.concatenate(
        [
            (bits.astype(np.int64) + offset).astype(np.uint32).view(np.float32)
            for offset in range(-4, 5)
        ]
    )
    values = np.concatenate((linear, around))
    expected = np.rint(np.clip(imaging.linear_to_srgb(values), 0, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(imaging.display_pixels(values, "linear-srgb"), expected)
