"""Validated effect controls in the original slider units and permitted ranges."""

from dataclasses import dataclass, field, fields, replace
from math import isfinite
from typing import Literal

type EffectKind = Literal["primary", "secondary", "soft", "ring"]

KINDS: tuple[EffectKind, ...] = ("primary", "secondary", "soft", "ring")

# CPU mappings are recovered; the UI retains the prototype's permitted ranges.
RANGES = {
    "quantity": (0, 100),
    "intensity": (0, 200),
    "length": (1, 100),
    "angle": (-180, 180),
    "points": (2, 12),
    "sharpness": (0, 100),
    "spread": (0, 100),
    "saturation": (-100, 100),
    "hue": (-180, 180),
    "rainbow": (0, 100),
    "frequency": (0, 100),
    "phase": (0, 100),
    "size": (1, 100),
    "soft_filter": (0, 100),
    "falloff": (0, 100),
    "inner_width": (1, 100),
    "inner_falloff": (0, 100),
    "outer_width": (1, 100),
    "outer_falloff": (0, 100),
}
PARAMETERS = {
    "primary": (
        "quantity",
        "intensity",
        "length",
        "angle",
        "points",
        "sharpness",
        "spread",
        "saturation",
        "hue",
        "rainbow",
        "frequency",
        "phase",
    ),
    "secondary": (
        "quantity",
        "intensity",
        "length",
        "angle",
        "sharpness",
        "spread",
        "saturation",
        "hue",
        "rainbow",
        "frequency",
        "phase",
    ),
    "soft": ("quantity", "intensity", "size", "soft_filter", "saturation", "hue", "falloff"),
    "ring": (
        "quantity",
        "intensity",
        "size",
        "saturation",
        "hue",
        "inner_width",
        "inner_falloff",
        "outer_width",
        "outer_falloff",
    ),
}


@dataclass(frozen=True)
class EffectParameters:
    """Immutable slider values; inactive groups retain their complete control state."""

    enabled: bool = True
    quantity: float = 80
    intensity: float = 45
    length: float = 25
    angle: float = 0
    points: int = 4
    sharpness: float = 60
    spread: float = 25
    saturation: float = 0
    hue: float = 0
    rainbow: float = 0
    frequency: float = 50
    phase: float = 50
    size: float = 20
    soft_filter: float = 0
    falloff: float = 60
    inner_width: float = 40
    inner_falloff: float = 50
    outer_width: float = 25
    outer_falloff: float = 50

    def __post_init__(self) -> None:
        """Reject nonfinite, out-of-range, or nonintegral point-count inputs."""
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean.")
        for key, (low, high) in RANGES.items():
            value = getattr(self, key)
            if type(value) not in (int, float) or not isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be finite, between {low} and {high}.")
        if type(self.points) is not int:
            raise ValueError("Number of points must be an integer.")


@dataclass(frozen=True)
class EffectSettings:
    """Keep independent parameter values for all four emission groups."""

    primary: EffectParameters = field(default_factory=EffectParameters)
    secondary: EffectParameters = field(
        default_factory=lambda: EffectParameters(enabled=False, intensity=12, length=15, spread=75)
    )
    soft: EffectParameters = field(default_factory=lambda: EffectParameters(intensity=20, size=12))
    ring: EffectParameters = field(
        default_factory=lambda: EffectParameters(enabled=False, intensity=15)
    )

    def __post_init__(self) -> None:
        """Require validated parameter records for every effect group."""
        if any(not isinstance(getattr(self, f.name), EffectParameters) for f in fields(self)):
            raise ValueError("All effect groups must contain EffectParameters.")

    @classmethod
    def disabled(cls) -> "EffectSettings":
        """Return the default control values with every group switched off."""
        base = cls()
        return cls(**{k: replace(getattr(base, k), enabled=False) for k in KINDS})
