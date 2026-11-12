"""Background-process entry that compiles the image code once per program version.

Kept free of Qt and heavy imports so a launch whose code is already compiled exits
within a fraction of a second.
"""

import hashlib
import os
import platform
import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from threading import Thread
from typing import Any

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


def compiled_files() -> list[Path]:
    """StarLab's Numba compiled-code files where this process will look for them."""
    configured: str | None = os.environ.get("NUMBA_CACHE_DIR")
    root: Path = Path(configured) if configured else PACKAGE
    return [
        path
        for pattern in ("*.nbi", "*.nbc")
        for path in root.rglob(pattern)
        if configured is None or "starlab" in path.as_posix()
    ]


def clear_compiled_code() -> None:
    """Remove compiled code from an earlier build.

    Numba only checks each kernel's own source file, so a changed helper in another
    module could otherwise reuse stale machine code.
    """
    for path in compiled_files():
        path.unlink(missing_ok=True)


def prepare(stamp: Path | None) -> None:
    """Compile and warm the image code unless this build is already prepared."""
    signature: str = code_signature()
    if (
        stamp is not None
        and stamp.is_file()
        and stamp.read_text(encoding="utf-8") == signature
        and any(path.suffix == ".nbi" for path in compiled_files())
    ):
        return
    if stamp is not None:
        clear_compiled_code()  # new or changed build: never mix old and new machine code
    from .application.warmup import warm_up
    from .bootstrap import create_services
    from .infrastructure.tiff_io import write_samples

    warm_up(create_services(sky_mask=True), write_samples)
    if stamp is not None:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(signature, encoding="utf-8")


def _wait_for_parent_exit_windows(pid: int) -> None:
    """Block until process `pid` ends; a missing process counts as already ended."""
    winapi: Any = import_module("_winapi")  # Windows only, so untyped here

    try:
        handle = winapi.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    except OSError:
        return
    winapi.WaitForSingleObject(handle, winapi.INFINITE)


def main(arguments: list[str]) -> None:
    """Run preparation, stopping at once when the app that started it goes away."""

    def exit_with_parent() -> None:
        """POSIX: the OS closes the app's stdin pipe on exit or crash."""
        sys.stdin.read()
        os._exit(0)

    def exit_with_parent_windows(pid: int) -> None:
        """Windows: a thread reading the stdin pipe deadlocks `import numpy`, and the
        parent PID is a launcher stub, so wait on the app's own process handle."""
        _wait_for_parent_exit_windows(pid)
        os._exit(0)

    parent: str | None = os.environ.get("STARLAB_PARENT_PID")
    if parent is not None:
        if sys.platform == "win32":
            Thread(target=exit_with_parent_windows, args=(int(parent),), daemon=True).start()
        else:
            Thread(target=exit_with_parent, daemon=True).start()
    prepare(Path(arguments[0]) if arguments else None)
