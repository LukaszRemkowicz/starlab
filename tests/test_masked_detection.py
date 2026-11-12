import numpy as np

from starlab.domain.imaging import ImageFrame
from starlab.domain.sky_mask import SkyMask
from starlab.domain.stars import detect_stars


def test_bright_foreground_cannot_change_sky_normalization_or_stars():
    y, x = np.mgrid[:96, :128]
    sky = 0.0001 + 0.02 * np.exp(-((x - 40.3) ** 2 + (y - 20.7) ** 2) / 5)
    foreground = 0.98 * np.exp(-((x - 70) ** 2 + (y - 78) ** 2) / 12)
    mask = SkyMask(1, np.where(y < 48, 1, 0).astype(np.uint8), (96, 128))
    reports = []
    detections = []
    for image in (sky, sky + foreground):
        report = {}
        frame = ImageFrame(np.repeat(image[..., None], 3, axis=2).astype(np.float32))
        detections.append(detect_stars(frame, sky_mask=mask, diagnostics=report))
        reports.append(report)
    assert detections[0] and detections[0] == detections[1]
    assert all(s.y < 48 for s in detections[1])
    assert reports[0]["normalization_luminance"] == reports[1]["normalization_luminance"]


def test_fully_foreground_image_produces_no_catalog():
    frame = ImageFrame(np.ones((32, 48, 3), np.float32))
    mask = SkyMask(2, np.zeros((16, 24), np.uint8), (32, 48))
    assert detect_stars(frame, sky_mask=mask) == ()


def test_all_sky_mask_preserves_actual_native_grid():
    from pathlib import Path

    from starlab.infrastructure.tiff_io import read_tiff

    frame = read_tiff(Path("tests/fixtures/detection/grid-1536.tif"))
    mask = SkyMask(
        4,
        np.ones(((frame.height + 3) // 4, (frame.width + 3) // 4), np.uint8),
        (frame.height, frame.width),
    )
    assert detect_stars(frame, sky_mask=mask) == detect_stars(frame)
