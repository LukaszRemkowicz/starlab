"""Bounded UTF-8 preset reading and atomic JSON settings persistence."""

import json
import os
import tempfile
from pathlib import Path

from ..domain.effects.settings import EffectSettings
from ..domain.presets import parse_preset, preset_data


def load_preset(path: Path | str) -> EffectSettings:
    """Load a size-bounded settings file through the domain schema validator."""
    path = Path(path)
    if path.stat().st_size > 100_000:
        raise ValueError("Preset is too large.")
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
        return parse_preset(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Preset must be a valid UTF-8 JSON file.") from exc


def save_preset(path: Path | str, settings: EffectSettings) -> None:
    """Serialize validated settings, replacing the destination atomically."""
    path = Path(path)
    text: str = json.dumps(preset_data(settings), indent=2, allow_nan=False)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".preset-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
        os.replace(temporary_path, path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
