import numpy as np
import pytest

from starlab.domain.detection.pipeline import analyze
from starlab.domain.imaging import ImageFrame


@pytest.mark.parametrize("width", [1024, 1023, 1027])
def test_large_stars_land_on_their_centers_for_any_image_width(width):
    height = 600
    rows, columns = np.mgrid[:height, :width].astype(np.float32)
    image = np.full((height, width), 0.01, np.float32)
    truth = [(100.0, 300.0), (540.0, 300.0), (980.0, 300.0)]
    for x, y in truth:
        image += np.minimum(1.0, 3.0 * np.exp(-((columns - x) ** 2 + (rows - y) ** 2) / 392))
    found, _ = analyze(ImageFrame(np.clip(image, 0, 1)[..., None]))
    centers = sorted((row["x"] - 0.5, row["y"] - 0.5) for row in found)
    assert len(centers) == len(truth)
    for (x, y), (true_x, true_y) in zip(centers, truth):
        assert abs(x - true_x) < 1 and abs(y - true_y) < 1
