import json
import os
import tempfile
from pathlib import Path

from ..domain.effects.settings import EffectSettings
from ..domain.presets import parse_preset, preset_data


def load_preset(path: Path | str) -> EffectSettings:
    path = Path(path)
    if path.stat().st_size > 100_000:
        raise ValueError("Preset is too large.")
    try:
        return parse_preset(json.loads(path.read_text(encoding="utf-8")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Preset must be a valid UTF-8 JSON file.") from exc


def save_preset(path: Path | str, settings: EffectSettings) -> None:
    path = Path(path)
    text = json.dumps(preset_data(settings), indent=2, allow_nan=False)
    fd, temp = tempfile.mkstemp(prefix=".preset-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
