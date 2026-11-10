"""Versioned preset serialization with strict control-key and range validation."""

from dataclasses import asdict, fields
from typing import TypedDict, cast

from .effects.settings import KINDS, EffectKind, EffectParameters, EffectSettings


class ParameterData(TypedDict):
    """JSON-compatible control fields accepted by the validated parameter record."""

    enabled: bool
    quantity: float
    intensity: float
    length: float
    angle: float
    points: int
    sharpness: float
    spread: float
    saturation: float
    hue: float
    rainbow: float
    frequency: float
    phase: float
    size: float
    soft_filter: float
    falloff: float
    inner_width: float
    inner_falloff: float
    outer_width: float
    outer_falloff: float


class PresetData(TypedDict):
    """Schema marker and complete independent values for every effect group."""

    schema_version: int
    settings: dict[EffectKind, ParameterData]


def preset_data(settings: EffectSettings) -> PresetData:
    """Serialize validated controls without modifying their slider values."""
    return {
        "schema_version": 1,
        "settings": cast(dict[EffectKind, ParameterData], asdict(settings)),
    }


def parse_preset(data: object) -> EffectSettings:
    """Read one supported schema, accepting legacy presets without soft_filter."""
    try:
        if not isinstance(data, dict) or set(data) != {"schema_version", "settings"}:
            raise ValueError("Invalid preset structure.")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported preset version.")
        groups: object = data["settings"]
        if not isinstance(groups, dict) or set(groups) != set(KINDS):
            raise ValueError("Preset must contain all four groups.")
        groups = {
            kind: {"soft_filter": 0, **parameters} if isinstance(parameters, dict) else parameters
            for kind, parameters in groups.items()
        }
        keys: set[str] = {parameter_field.name for parameter_field in fields(EffectParameters)}
        if any(
            not isinstance(parameters, dict) or set(parameters) != keys
            for parameters in groups.values()
        ):
            raise ValueError("Preset contains missing or unknown parameters.")
        return EffectSettings(
            **{
                kind: EffectParameters(**cast(ParameterData, parameters))
                for kind, parameters in groups.items()
            }
        )
    except (TypeError, KeyError) as exc:
        raise ValueError("Invalid preset values.") from exc
