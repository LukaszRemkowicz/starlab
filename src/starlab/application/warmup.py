"""Compile the native numerical code ahead of the first real photo."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ..domain.effects.settings import KINDS, EffectSettings
from ..domain.imaging import ImageFrame
from ..domain.stars import DetectionSettings, Star
from .use_cases import ImageServices, render_preview


def _synthetic_sky() -> ImageFrame:
    """Return a small linear star field that exercises detection, photometry and effects."""
    rows, columns = np.mgrid[:256, :256]
    pixels: np.ndarray = np.full((256, 256, 3), 0.02, np.float32)
    rng: np.random.Generator = np.random.default_rng(0)
    for x, y, peak in zip(
        rng.uniform(16, 240, 14), rng.uniform(16, 240, 14), rng.uniform(0.2, 1, 14)
    ):
        star: np.ndarray = peak * np.exp(-((columns - x) ** 2 + (rows - y) ** 2) / 4.5)
        pixels += (star[..., None] * np.array([1.0, 0.9, 0.8])).astype(np.float32)
    return ImageFrame(np.minimum(pixels, 1), name="warm-up")


def warm_up(services: ImageServices) -> int:
    """Run the production open, detect, preview and export path once; return the star count.

    Numba compiles each kernel on first use and stores the machine code next to the
    package, so later launches start fast. Only temporary files are written here.
    """
    settings: EffectSettings = EffectSettings()
    settings = replace(
        settings,
        **{kind: replace(getattr(settings, kind), enabled=True) for kind in KINDS},
    )
    settings = replace(settings, soft=replace(settings.soft, soft_filter=50))
    with TemporaryDirectory(prefix="star-effects-lab-") as directory:
        source: Path = Path(directory) / "warm-up.tif"
        synthetic: ImageFrame = _synthetic_sky()
        services.write(source, synthetic, synthetic.pixels)
        frame, sky_mask = services.open_image(source)
        stars: tuple[Star, ...] = ()
        for method in ("multiscale", "native"):
            stars = services.analyze(
                frame,
                source,
                DetectionSettings(
                    threshold_sigma=5.0, method=method, sky_only=sky_mask is not None
                ),
                sky_mask=sky_mask,
            )
        for full_resolution in (False, True):
            render_preview(
                frame, stars, settings, full_resolution=full_resolution, sky_mask=sky_mask
            )
        services.export(
            Path(directory) / "result.tif", frame, stars, settings, False, source, sky_mask=sky_mask
        )
    return len(stars)
