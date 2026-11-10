"""Controlled completeness/contamination checks for the optional detector."""

from threading import Event

import numpy as np
import pytest

from starlab.domain.imaging import Cancelled, ImageFrame


def gaussian(shape, x, y, sigma, amplitude):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))


def detect(values, **kwargs):
    from starlab.domain.detection.multiscale import analyze

    return analyze(ImageFrame(values[..., None], color_space="linear-gray"), **kwargs)


def assert_centers(rows, expected, tolerance=0.3):
    actual = np.array([(r["x"] - 0.5, r["y"] - 0.5) for r in rows])
    assert len(rows) == len(expected), actual
    for center in expected:
        assert np.linalg.norm(actual - center, axis=1).min() < tolerance, actual


def test_large_and_saturated_stars_across_tile_seams_and_faint_image_edges():
    shape = (280, 300)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    values = 0.04 + xx * 0.0001 + yy * 0.00005
    centers = [(3.5, 8.2), (293.4, 271.1), (127.7, 129.2), (231.3, 62.5)]
    for (x, y), sigma, amplitude in zip(
        centers, (1.4, 1.5, 9.0, 5.0), (0.008, 0.01, 0.65, 3.0), strict=True
    ):
        values += gaussian(shape, x, y, sigma, amplitude)
    rows, report = detect(np.minimum(values, 1), tile_size=128)
    assert_centers(rows, centers, tolerance=0.4)
    assert max(r["fwhm"] for r in rows) > 17
    assert sum(r["saturated"] for r in rows) == 1
    assert report["count"] == 4


def test_saturated_disk_is_one_connected_source_instead_of_a_ring_of_peaks():
    yy, xx = np.mgrid[:192, :192]
    values = np.full((192, 192), 0.02)
    values[(xx - 95) ** 2 + (yy - 96) ** 2 <= 13**2] = 1
    rows, _ = detect(values, tile_size=64)
    assert_centers(rows, [(95, 96)])
    assert rows[0]["saturated"]


@pytest.mark.parametrize("sigma, amplitude", [(13.6, 0.6), (13.6, 3.0)])
def test_broad_psf_above_thirty_pixels_fwhm_has_subpixel_center(sigma, amplitude):
    values = 0.02 + gaussian((320, 320), 159.7, 160.2, sigma, amplitude)
    rows, _ = detect(np.minimum(values, 1), tile_size=160)
    assert_centers(rows, [(159.7, 160.2)])
    assert rows[0]["fwhm"] > 30


def test_rejects_hot_pixels_noise_and_nebula_ridges_but_keeps_star_on_gradient():
    shape = (256, 256)
    yy, xx = np.mgrid[:256, :256]
    rng = np.random.default_rng(52)
    values = 0.1 + xx * 0.0002 + yy * 0.0001 + rng.normal(0, 0.0003, shape)
    values += 0.05 * np.exp(-((yy - 174) ** 2) / (2 * 5**2))
    values[36, 31] = 1
    values += gaussian(shape, 67.3, 66.6, 1.5, 0.007)
    rows, _ = detect(values, tile_size=128)
    assert_centers(rows, [(67.3, 66.6)])


def test_global_scale_suppression_preserves_eight_pixel_pair_and_stable_tiling():
    shape = (240, 240)
    centers = [(60.2, 64.3), (68.3, 64.1), (127.7, 129.2)]
    values = np.full(shape, 0.02)
    for (x, y), sigma in zip(centers, (1.4, 1.4, 6.0), strict=True):
        values += gaussian(shape, x, y, sigma, 0.25)
    first, _ = detect(values, tile_size=64)
    second, _ = detect(values, tile_size=240)
    assert_centers(first, centers)
    assert_centers(second, centers)
    np.testing.assert_allclose(
        [(r["x"], r["y"], r["size"]) for r in first],
        [(r["x"], r["y"], r["size"]) for r in second],
        atol=1e-4,
    )


def test_foreground_mask_excludes_bright_source_and_its_threshold_influence():
    class Mask:
        def region(self, bounds):
            left, top, right, bottom = bounds
            return np.broadcast_to(
                np.arange(top, bottom)[:, None] < 96, (bottom - top, right - left)
            )

        def allowed_at(self, x, y):
            return y < 96

    shape = (192, 192)
    values = np.full(shape, 0.03) + gaussian(shape, 40.4, 40.2, 1.5, 0.02)
    values += gaussian(shape, 100.3, 140.6, 6, 0.8)
    rows, _ = detect(values, sky_mask=Mask(), tile_size=96)
    assert_centers(rows, [(40.4, 40.2)])
    values[96:] = 100
    changed, _ = detect(values, sky_mask=Mask(), tile_size=96)
    assert_centers(changed, [(40.4, 40.2)])
    assert rows[0]["normalized_brightness"] == changed[0]["normalized_brightness"]


def test_blank_image_and_cancel_before_allocation():
    rows, _ = detect(np.zeros((40, 40)))
    assert rows == []
    token = Event()
    token.set()
    with pytest.raises(Cancelled):
        detect(np.zeros((40, 40)), cancel=token)


@pytest.mark.parametrize(
    "options",
    [
        {"tile_size": 0},
        {"scales": ()},
        {"scales": (np.nan,)},
        {"threshold_sigma": 0},
        {"min_distance": -1},
    ],
)
def test_invalid_options_rejected(options):
    with pytest.raises(ValueError):
        detect(np.zeros((40, 40)), **options)
