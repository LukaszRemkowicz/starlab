from dataclasses import dataclass
from math import isfinite
from threading import Event

import numpy as np

from .detection.pipeline import analyze
from .imaging import ImageFrame, check_cancel


@dataclass(frozen=True)
class Star:
    id: int
    x: float
    y: float
    brightness: float
    color: tuple[float, float, float]
    radius: float = 2.0
    visible: bool = True
    source_profile: tuple[float, ...] = ()
    soft_weight: float = 0.5
    source_peak: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not all(
            isfinite(v) for v in (self.x, self.y, self.brightness, self.radius, *self.color)
        ):
            raise ValueError("Star values must be finite.")
        if self.brightness < 0 or self.radius <= 0 or len(self.color) != 3:
            raise ValueError("Invalid star brightness, radius or color.")
        if not isfinite(self.soft_weight) or not 0 <= self.soft_weight <= 1:
            raise ValueError("Invalid soft filter weight.")
        if len(self.source_peak) != 3 or not all(isfinite(v) and v >= 0 for v in self.source_peak):
            raise ValueError("Invalid measured source peak.")
        if self.source_profile and (
            len(self.source_profile) != 10
            or not all(isfinite(v) for v in self.source_profile)
            or self.source_profile[0] <= 0
        ):
            raise ValueError("Invalid measured source profile.")


@dataclass(frozen=True)
class DetectionSettings:
    threshold_sigma: float = 5.0
    min_distance: int = 5
    max_stars: int | None = None
    sky_only: bool = False
    method: str = "native"

    def __post_init__(self) -> None:
        if type(self.sky_only) is not bool or self.method not in ("native", "multiscale"):
            raise ValueError("Invalid sky mask or detection method.")
        if (
            type(self.threshold_sigma) not in (int, float)
            or not isfinite(self.threshold_sigma)
            or not 1 <= self.threshold_sigma <= 30
        ):
            raise ValueError("Detection threshold must be between 1 and 30 sigma.")
        if (
            type(self.min_distance) is not int
            or not 2 <= self.min_distance <= 30
            or (
                self.max_stars is not None
                and (type(self.max_stars) is not int or not 1 <= self.max_stars <= 10000)
            )
        ):
            raise ValueError("Invalid detector limits.")


def filter_stars(stars, sky_mask):
    if sky_mask is None or not stars:
        return stars
    keep = sky_mask.keep(np.array([(star.x, star.y) for star in stars]))
    return tuple(star for star, allowed in zip(stars, keep) if allowed)


def detect_stars(
    frame: ImageFrame,
    settings: DetectionSettings = DetectionSettings(),
    cancel: Event | None = None,
    *,
    diagnostics: dict | None = None,
    sky_mask=None,
) -> tuple[Star, ...]:
    """Recovered analysis with fast seeds; not a bit-exact original full scan.

    Coordinates are pixel indices (native catalog coordinates minus0.5).
    Color is normalized catalog RGB, brightness its luminance, radius its size.
    No automatic top-N limit is applied. The legacy threshold_sigma name now
    scales the proxy cutoff:5 selects the native0.2; it is not a noise estimate.
    """
    check_cancel(cancel)
    if settings.sky_only and sky_mask is None:
        raise ValueError("Sky-only detection requires a sky mask.")
    if settings.method == "multiscale":
        from .detection.multiscale import analyze as analyze_multiscale

        rows, report = analyze_multiscale(
            frame,
            settings.threshold_sigma,
            max(1, settings.min_distance // 2),
            cancel,
            sky_mask=sky_mask,
        )
    else:
        rows, report = analyze(
            frame,
            threshold=min(0.98, 0.2 * settings.threshold_sigma / 5),
            neighbor_radius=max(1, settings.min_distance // 2),
            cancel=cancel,
            sky_mask=sky_mask,
        )
    if diagnostics is not None:
        diagnostics.update(report)
    if settings.max_stars is not None:
        rows = sorted(rows, key=lambda row: row["normalized_brightness"], reverse=True)[
            : settings.max_stars
        ]
    return tuple(
        Star(
            i + 1,
            row["x"] - 0.5,
            row["y"] - 0.5,
            row["normalized_brightness"],
            tuple(row["normalized_rgb"]),
            max(row["size"], 0.25),
        )
        for i, row in enumerate(rows)
    )
