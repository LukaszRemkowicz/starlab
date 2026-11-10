"""Disposable analysis cache; the image and settings remain the source of truth."""

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from ..domain.stars import DetectionSettings, Star


@lru_cache(maxsize=1)
def engine_signature():
    digest = hashlib.sha256(b"star-effects-detector-cache-v1")
    directory = Path(__file__).parents[1] / "domain"
    paths = [
        directory / "stars.py",
        directory / "imaging.py",
        directory / "sky_mask.py",
        directory / "photometry.py",
        Path(__file__).with_name("sky_mask.py"),
        Path(__file__).with_name("models") / "sky-mask.json",
        Path(__file__).with_name("models") / "sky-mask.onnx",
        Path(__file__).with_name("tiff_io.py"),
        *sorted((directory / "detection").glob("*.py")),
        directory / "detection/analysis_luts.npz",
        directory / "detection/full_lcms_lut.npz",
        directory / "detection/full_lcms_lut.json",
    ]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def identity(source, settings):
    source = Path(source).resolve()
    stat = source.stat()
    return dict(
        source=str(source),
        size=stat.st_size,
        modified=stat.st_mtime_ns,
        engine=engine_signature(),
        settings=asdict(settings),
    )


def cache_path(source, settings=None):
    key = hashlib.sha256(
        (
            str(Path(source).resolve())
            + json.dumps(asdict(settings or DetectionSettings()), sort_keys=True)
        ).encode()
    ).hexdigest()
    return Path(tempfile.gettempdir()) / "star-effects-lab" / "catalogs" / f"{key}.json"


def load_catalog(source, settings):
    try:
        stored = json.loads(cache_path(source, settings).read_text(encoding="utf-8"))
        if stored["identity"] != identity(source, settings):
            return None
        return tuple(
            Star(
                **{
                    **row,
                    "color": tuple(row["color"]),
                    "source_profile": tuple(row.get("source_profile", ())),
                    "source_peak": tuple(row.get("source_peak", (0.0, 0.0, 0.0))),
                }
            )
            for row in stored["stars"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_catalog(source, settings, stars):
    temporary = None
    try:
        target = cache_path(source, settings)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = dict(identity=identity(source, settings), stars=[asdict(star) for star in stars])
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, separators=(",", ":"))
        os.replace(temporary, target)
    except OSError:
        logging.getLogger(__name__).debug("Analysis cache could not be written", exc_info=True)
        # Cache failures never discard a successful analysis.
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
