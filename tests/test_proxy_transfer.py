import numpy as np
import pytest

from starlab.domain.detection import proxy
from starlab.domain.imaging import linear_to_srgb, srgb_to_linear


def reference_encode16(values):
    result = np.rint(np.clip(linear_to_srgb(values), 0, 1) * 65535).astype(np.uint16)
    return np.repeat(result, 3, axis=2) if result.shape[2] == 1 else result


def test_encoded_proxy_preserves_uint16_samples_and_adjacent_rounding_values():
    ramp = np.arange(65536, dtype=np.float32) / 65535
    edges = srgb_to_linear((np.arange(65535, dtype=np.float32) + 0.5) / 65535)
    around = np.concatenate(
        [
            (edges.view(np.uint32).astype(np.int64) + offset).astype(np.uint32).view(np.float32)
            for offset in range(-8, 9)
        ]
    )
    values = np.concatenate((ramp, srgb_to_linear(ramp), around)).reshape(-1, 1, 1)
    np.testing.assert_array_equal(proxy.encode16(values), reference_encode16(values))


def test_encoded_proxy_preserves_random_float_bits_gray_and_strided_tiles():
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 0x7F800000, size=(128, 257, 3), dtype=np.uint32)
    values = bits.view(np.float32)
    values[:, ::3] = rng.uniform(-0.1, 1.2, values[:, ::3].shape)
    for tile in (values, values[::2, 1::2], values[..., :1]):
        with np.errstate(over="ignore"):
            np.testing.assert_array_equal(proxy.encode16(tile), reference_encode16(tile))


def test_encoded_proxy_does_not_repeat_gamma_for_image_pixels(monkeypatch):
    proxy.encode16(np.zeros((2, 2, 3), np.float32))
    sizes = []

    def observe(values):
        sizes.append(values.size)
        return linear_to_srgb(values)

    monkeypatch.setattr(proxy, "linear_to_srgb", observe)
    values = np.random.default_rng(8).random((256, 512, 3), dtype=np.float32)
    np.testing.assert_array_equal(proxy.encode16(values), reference_encode16(values))
    assert max(sizes, default=0) < values.size // 4


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("masked", [False, True])
def test_coarse_proxy_preserves_encoded_box_rounding_and_mask_divisors(channels, masked):
    from starlab.domain.sky_mask import FOREGROUND, SKY, SkyMask

    pixels = np.random.default_rng(3).random((19, 23, channels), dtype=np.float32)
    classes = np.full(pixels.shape[:2], SKY, np.uint8)
    classes[::3, ::2] = FOREGROUND
    mask = SkyMask(1, classes, pixels.shape[:2]) if masked else None
    height, width = pixels.shape[:2]
    dh, dw = height // 4, width // 4
    bh, ys = proxy.axis_mapping(height, dh)
    bw, xs = proxy.axis_mapping(width, dw)
    expected = np.empty((dh, dw, 3), np.uint8)
    for row, sy in enumerate(ys):
        for col, sx in enumerate(xs):
            block = reference_encode16(pixels[sy : sy + bh, sx : sx + bw])
            valid = classes[sy : sy + bh, sx : sx + bw] != FOREGROUND
            total = block.sum(axis=(0, 1), dtype=np.float32)
            if masked:
                total = block[valid].sum(axis=0, dtype=np.float32)
                average = total / np.float32(valid.sum()) if valid.any() else total * 0
            else:
                average = total * np.float32(1 / (bh * bw))
            expected[row, col] = (np.rint(average).astype(np.uint16) >> 8).astype(np.uint8)
    np.testing.assert_array_equal(
        proxy.coarse_codes(pixels, sky_mask=mask), proxy.convert_codes(expected)
    )


def test_full_proxy_keeps_three_channel_cmm_conversion():
    pixels = np.random.default_rng(13).random((31, 37, 3), dtype=np.float32)
    expected = proxy.convert_codes((reference_encode16(pixels) >> 8).astype(np.uint8))
    np.testing.assert_array_equal(proxy.full_codes(pixels), expected)


def test_full_proxy_preserves_16bit_rounding_before_high_byte_selection():
    edges = srgb_to_linear((np.arange(255, dtype=np.float32) * 256 + 255.5) / 65535)
    values = np.concatenate(
        [
            (edges.view(np.uint32).astype(np.int64) + offset).astype(np.uint32).view(np.float32)
            for offset in range(-16, 17)
        ]
    ).reshape(-1, 1, 1)
    expected = proxy.convert_codes((reference_encode16(values) >> 8).astype(np.uint8))
    np.testing.assert_array_equal(proxy.full_codes(values), expected)


def test_parallel_coarse_chunks_preserve_masked_output_across_chunk_edges(monkeypatch):
    from starlab.domain.sky_mask import FOREGROUND, SKY, SkyMask

    pixels = np.random.default_rng(31).random((1031, 37, 3), dtype=np.float32)
    classes = np.full(pixels.shape[:2], SKY, np.uint8)
    classes[511:515, :19] = FOREGROUND
    mask = SkyMask(1, classes, pixels.shape[:2])
    monkeypatch.setattr(proxy, "COARSE_WORKERS", 1)
    expected = proxy.coarse_codes(pixels, sky_mask=mask)
    monkeypatch.setattr(proxy, "COARSE_WORKERS", 4)
    np.testing.assert_array_equal(proxy.coarse_codes(pixels, sky_mask=mask), expected)


@pytest.mark.parametrize("denominator", [0.01, 0.4, 0.98, 1.0])
def test_fused_full_proxy_preserves_cmm_decode_normalization_and_readonly_tiles(denominator):
    pixels = np.random.default_rng(54).random((73, 151, 3), dtype=np.float32)
    pixels.flags.writeable = False
    for tile in (pixels, pixels[1::2, ::3], pixels[..., :1]):
        expected = proxy.decode(proxy.full_codes(tile), denominator)
        np.testing.assert_array_equal(proxy.full_proxy(tile, denominator), expected)


def test_universal_cmm_cube_matches_independent_runtime_for_color_planes():
    cube = proxy._full_lcms()
    rgb = np.empty((256, 256, 3), np.uint8)
    rgb[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
    rgb[..., 2] = np.arange(256, dtype=np.uint8)[None, :]
    for red in (0, 1, 31, 63, 127, 128, 191, 254, 255):
        rgb[..., 0] = red
        np.testing.assert_array_equal(cube[red], proxy.convert_codes(rgb))
    rgb = np.random.default_rng(27).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        cube[rgb[..., 0], rgb[..., 1], rgb[..., 2]], proxy.convert_codes(rgb)
    )
