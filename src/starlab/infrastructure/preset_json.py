"""Bounded UTF-8 preset reading and atomic JSON settings persistence."""

import json
import os
from pathlib import Path

from ..domain.effects.settings import EffectSettings
from ..domain.presets import parse_preset, preset_data


def load_preset(path: Path | str) -> EffectSettings:
    """Load a size-bounded settings file through the domain schema validator."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("Preset must be a regular file.")
    with path.open("rb") as stream:
        raw: bytes = stream.read(100_001)  # bounded even if the size changes after a check
    if len(raw) > 100_000:
        raise ValueError("Preset is too large.")
    try:
        data: object = json.loads(raw.decode("utf-8"))
        return parse_preset(data)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Preset must be a valid UTF-8 JSON file.") from exc


def save_preset(path: Path | str, settings: EffectSettings) -> None:
    """Serialize validated settings, replacing the destination atomically."""
    path = Path(path)
    text: str = json.dumps(preset_data(settings), indent=2, allow_nan=False)
    # Exclusive creation honours the umask, unlike mkstemp's private 0600 mode.
    temporary_path: Path = path.parent / f".preset-{os.urandom(8).hex()}.json"
    try:
        with temporary_path.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
