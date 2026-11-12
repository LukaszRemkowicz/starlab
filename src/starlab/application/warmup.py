"""Compile the native numerical code ahead of the first real photo."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import gaussian_filter

from ..domain.effects.settings import KINDS, EffectSettings
from ..domain.imaging import ImageFrame, linear_to_srgb
from ..domain.sky_mask import SkyMask
from ..domain.stars import DetectionSettings, Star
from .use_cases import ImageServices, render_preview

type SampleWriter = Callable[[Path, np.ndarray, bool], None]


def _synthetic_sky(height: int, width: int, count: int) -> ImageFrame:
    """Return a linear star field with a large saturated star and a star on the border.

    At least 256 stars per detection tile take the parallel candidate path; images up to
    1024 px are one tile, larger ones are split, and each case compiles its own code.
    """
    rng: np.random.Generator = np.random.default_rng(0)
    points: np.ndarray = np.zeros((height, width), np.float32)
    rows: np.ndarray = rng.integers(4, height - 4, count)
    columns: np.ndarray = rng.integers(4, width - 4, count)
    points[rows, columns] = rng.uniform(4, 25, count)
    points[height // 2, width // 2] = 3000  # large, saturated star: the coarse detector path
    points[height // 3, 1] = 20  # star on the image border
    light: np.ndarray = gaussian_filter(points, 1.5) + gaussian_filter(points, 12) * 0.05
    rows_grid, columns_grid = np.mgrid[:height, :width]
    background: np.ndarray = 0.02 + 0.00002 * columns_grid + 0.00001 * rows_grid
    pixels: np.ndarray = (background + light)[..., None] * np.array([1.0, 0.9, 0.8])
    return ImageFrame(np.minimum(pixels, 1).astype(np.float32), name="warm-up")


def _encoded_variants(frame: ImageFrame) -> dict[str, tuple[np.ndarray, bool]]:
    """Encoded sample layouts that each need their own compiled code: depth, compression, gray."""
    encoded: np.ndarray = np.clip(linear_to_srgb(frame.pixels), 0, 1)
    rgb16: np.ndarray = np.rint(encoded * 65535).astype(np.uint16)
    rgb8: np.ndarray = np.rint(encoded * 255).astype(np.uint8)
    return {
        "rgb16-compressed": (rgb16, True),
        "rgb8": (rgb8, False),
        "rgb8-compressed": (rgb8, True),
        "gray16": (rgb16[..., 1:2], False),
        "gray16-compressed": (rgb16[..., 1:2], True),
    }


def _open(services: ImageServices, source: Path) -> tuple[ImageFrame, SkyMask | None]:
    """Open one file like the app does: strided thumbnail, then image and mask."""
    if services.read_preview is not None:
        services.read_preview(source, max_edge=64)
    return services.open_image(source)


def _detect(
    services: ImageServices,
    source: Path,
    opened: tuple[ImageFrame, SkyMask | None],
    methods: tuple[str, ...] = ("native",),
) -> tuple[Star, ...]:
    """Run the given detectors on an opened file."""
    frame, sky_mask = opened
    stars: tuple[Star, ...] = ()
    for method in methods:
        stars = services.analyze(
            frame,
            source,
            DetectionSettings(threshold_sigma=5.0, method=method, sky_only=sky_mask is not None),
            sky_mask=sky_mask,
        )
    return stars


def warm_up(services: ImageServices, write_samples: SampleWriter | None = None) -> int:
    """Run the production open, detect, preview and export paths; return the star count.

    Numba compiles each kernel on first use for each input layout and stores the
    machine code next to the package, so later launches start fast. Only temporary
    files are written here, and they are removed afterwards.
    """
    settings: EffectSettings = EffectSettings()
    settings = replace(
        settings,
        **{kind: replace(getattr(settings, kind), enabled=True) for kind in KINDS},
    )
    settings = replace(settings, soft=replace(settings.soft, soft_filter=50))
    synthetic: ImageFrame = _synthetic_sky(512, 512, 400)
    with TemporaryDirectory(prefix="star-effects-lab-") as directory:
        source: Path = Path(directory) / "warm-up.tif"
        services.write(source, synthetic, synthetic.pixels)
        tiled: Path = Path(directory) / "tiled.tif"
        large: ImageFrame = _synthetic_sky(1100, 1300, 1600)
        services.write(tiled, large, large.pixels)
        files: list[Path] = [source, tiled]
        if write_samples is not None:
            for name, (samples, compress) in _encoded_variants(synthetic).items():
                variant: Path = Path(directory) / f"{name}.tif"
                write_samples(variant, samples, compress)
                files.append(variant)
        # Opening first: it is what a user waits for, so its kernels are cached soonest.
        opened = {path: _open(services, path) for path in files}
        for path in files[1:]:
            _detect(services, path, opened[path])
        frame, sky_mask = opened[source]
        stars = _detect(services, source, opened[source], ("multiscale", "native"))
        for full_resolution in (False, True):
            render_preview(
                frame, stars, settings, full_resolution=full_resolution, sky_mask=sky_mask
            )
        services.export(
            Path(directory) / "result.tif", frame, stars, settings, False, source, sky_mask=sky_mask
        )
    return len(stars)
