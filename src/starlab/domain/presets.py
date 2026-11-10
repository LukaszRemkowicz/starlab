from dataclasses import asdict, fields

from .effects.settings import KINDS, EffectParameters, EffectSettings


def preset_data(settings: EffectSettings) -> dict:
    return {"schema_version": 1, "settings": asdict(settings)}


def parse_preset(data: object) -> EffectSettings:
    try:
        if not isinstance(data, dict) or set(data) != {"schema_version", "settings"}:
            raise ValueError("Invalid preset structure.")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported preset version.")
        groups = data["settings"]
        if not isinstance(groups, dict) or set(groups) != set(KINDS):
            raise ValueError("Preset must contain all four groups.")
        groups = {
            k: {"soft_filter": 0, **v} if isinstance(v, dict) else v for k, v in groups.items()
        }
        keys = {f.name for f in fields(EffectParameters)}
        if any(not isinstance(v, dict) or set(v) != keys for v in groups.values()):
            raise ValueError("Preset contains missing or unknown parameters.")
        return EffectSettings(**{k: EffectParameters(**v) for k, v in groups.items()})
    except (TypeError, KeyError) as exc:
        raise ValueError("Invalid preset values.") from exc
