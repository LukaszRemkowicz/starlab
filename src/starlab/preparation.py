"""Background-process entry that compiles the image code once per program version.

Kept free of Qt and heavy imports so a launch whose code is already compiled exits
within a fraction of a second.
"""

import hashlib
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from threading import Thread

PACKAGE = Path(__file__).resolve().parent


def code_signature() -> str:
    """Identify this program build: its sources plus the interpreter, Numba and CPU."""
    digest = hashlib.sha256()
    for source in sorted(PACKAGE.rglob("*.py")):
        digest.update(source.relative_to(PACKAGE).as_posix().encode())
        digest.update(source.read_bytes())
    numba: str = version("numba")
    digest.update(f"{sys.version}|{numba}|{platform.machine()}".encode())
    return digest.hexdigest()


def _compiled_code_present() -> bool:
    """Whether Numba's compiled-code cache exists where this process will look for it."""
    configured: str | None = os.environ.get("NUMBA_CACHE_DIR")
    root: Path = Path(configured) if configured else PACKAGE
    return next(root.rglob("*.nbi"), None) is not None


def prepare(stamp: Path | None) -> None:
    """Compile and warm the image code unless this build is already prepared."""
    signature: str = code_signature()
    if (
        stamp is not None
        and stamp.is_file()
        and stamp.read_text(encoding="utf-8") == signature
        and _compiled_code_present()
    ):
        return
    from .application.warmup import warm_up
    from .bootstrap import create_services
    from .infrastructure.tiff_io import write_samples

    warm_up(create_services(sky_mask=True), write_samples)
    if stamp is not None:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(signature, encoding="utf-8")


def main(arguments: list[str]) -> None:
    """Run preparation, stopping at once if the app's end of stdin closes."""

    def exit_with_parent() -> None:
        """The app holds stdin open; the OS closes it when the app exits or crashes."""
        sys.stdin.read()
        os._exit(0)

    Thread(target=exit_with_parent, daemon=True).start()
    prepare(Path(arguments[0]) if arguments else None)
