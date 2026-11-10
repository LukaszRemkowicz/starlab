"""Disposable analysis cache; the image and settings remain the source of truth."""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from ..domain.array_types import RGB
from ..domain.stars import DetectionSettings, Star


class _DetectionData(TypedDict):
    """JSON detector settings included in the source-identity fingerprint."""

    threshold_sigma: float
    min_distance: int
    max_stars: int | None
    sky_only: bool
    method: str


class _CatalogIdentity(TypedDict):
    """Source metadata, detector configuration and implementation fingerprint."""

    source: str
    size: int
    modified: int
    engine: str
    settings: _DetectionData


class _StarFields(TypedDict):
    """Scalar star fields shared by serialized rows and constructor arguments."""

    id: int
    x: float
    y: float
    brightness: float
    radius: NotRequired[float]
    visible: NotRequired[bool]
    soft_weight: NotRequired[float]


class _StoredStar(_StarFields):
    """A cached row before JSON arrays are normalized to immutable star tuples."""

    color: Sequence[float]
    source_profile: NotRequired[Sequence[float]]
    source_peak: NotRequired[Sequence[float]]


class _StarArguments(_StarFields):
    """Normalized keyword arguments whose values are still validated by Star."""

    color: RGB
    source_profile: tuple[float, ...]
    source_peak: RGB


class _CatalogDocument(TypedDict):
    """Disposable JSON body with its fingerprint and serialized source catalog."""

    identity: _CatalogIdentity
    stars: list[_StoredStar]


@lru_cache(maxsize=1)
def engine_signature() -> str:
    """Hash detector code and assets once per process to invalidate incompatible catalogs."""
    digest = hashlib.sha256(b"star-effects-detector-cache-v1")
    directory = Path(__file__).parents[1] / "domain"
    paths = [
        directory / "stars.py",
        directory / "imaging.py",
        directory / "jit.py",
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


def identity(source: str | Path, settings: DetectionSettings) -> _CatalogIdentity:
    """Describe the resolved input file and all detector inputs affecting its catalog."""
    source = Path(source).resolve()
    stat = source.stat()
    return dict(
        source=str(source),
        size=stat.st_size,
        modified=stat.st_mtime_ns,
        engine=engine_signature(),
        settings=cast(_DetectionData, asdict(settings)),
    )


def cache_path(source: str | Path, settings: DetectionSettings | None = None) -> Path:
    """Choose a stable temporary-cache filename for the source path and detector settings."""
    key = hashlib.sha256(
        (
            str(Path(source).resolve())
            + json.dumps(asdict(settings or DetectionSettings()), sort_keys=True)
        ).encode()
    ).hexdigest()
    return Path(tempfile.gettempdir()) / "star-effects-lab" / "catalogs" / f"{key}.json"


def load_catalog(source: str | Path, settings: DetectionSettings) -> tuple[Star, ...] | None:
    """Return matching validated stars, or None for stale, missing or malformed cache data."""
    try:
        stored = cast(
            _CatalogDocument,
            json.loads(cache_path(source, settings).read_text(encoding="utf-8")),
        )
        if stored["identity"] != identity(source, settings):
            return None
        return tuple(_restore_star(row) for row in stored["stars"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _restore_star(row: _StoredStar) -> Star:
    """Normalize JSON arrays while retaining Star's validation of unknown keys and values."""
    arguments = cast(
        _StarArguments,
        {
            **row,
            "color": tuple(row["color"]),
            "source_profile": tuple(row.get("source_profile", ())),
            "source_peak": tuple(row.get("source_peak", (0.0, 0.0, 0.0))),
        },
    )
    return Star(**arguments)


def save_catalog(source: str | Path, settings: DetectionSettings, stars: tuple[Star, ...]) -> None:
    """Atomically replace a disposable cache without letting write failures lose analysis."""
    temporary: Path | None = None
    try:
        target = cache_path(source, settings)
        target.parent.mkdir(parents=True, exist_ok=True)
        data: _CatalogDocument = {
            "identity": identity(source, settings),
            "stars": [cast(_StoredStar, asdict(star)) for star in stars],
        }
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
