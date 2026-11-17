"""Build and install the wheel's portable CPU kernels before the first image is opened."""

import hashlib
import os
import shutil
import sys
from importlib.metadata import distribution, version
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from .preparation import PACKAGE, code_signature


def _cache_folder(relative: str) -> str:
    from numba.core.caching import UserProvidedCacheLocator

    # Numba keys each external cache directory by the path of its installed module.
    source = str(PACKAGE / relative / "__init__.py")
    return str(UserProvidedCacheLocator.get_suitable_cache_subpath(source))  # type: ignore[no-untyped-call]


def _signature() -> str:
    return "-".join((code_signature(), version("numpy"), version("llvmlite")))


def build(archive: Path) -> None:
    """Compile production layouts and store the resulting code by package-relative path."""
    from .application.warmup import warm_up
    from .bootstrap import create_services
    from .infrastructure.tiff_io import write_samples

    warm_up(create_services(sky_mask=True), write_samples)
    cache = Path(os.environ["NUMBA_CACHE_DIR"])
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=1) as output:
        output.writestr("signature", _signature())
        for init in sorted(PACKAGE.rglob("__init__.py")):
            relative = init.parent.relative_to(PACKAGE).as_posix()
            for file in sorted((cache / _cache_folder(relative)).glob("*.nb[ci]")):
                output.write(file, (Path(relative) / file.name).as_posix())


def install(data: Path) -> bool:
    """Activate matching wheel code in a writable cache, including on a clean first run."""
    archive = Path(
        str(distribution("star-effects-lab").locate_file("_starlab_kernels/kernels.zip"))
    )
    if (
        not archive.is_file()
        or os.environ.get("NUMBA_CPU_NAME", "generic") != "generic"
        or os.environ.get("NUMBA_CPU_FEATURES")
    ):
        return False
    signature = _signature()
    root = Path(os.environ.get("NUMBA_CACHE_DIR") or data / "kernels")
    identity = hashlib.sha256(f"{PACKAGE}|{signature}".encode()).hexdigest()[:24]
    target = root / identity
    with ZipFile(archive) as bundle:
        if bundle.read("signature").decode() != signature:
            return False
        # Set the target before importing Numba. Its decorators select their cache at import.
        os.environ["NUMBA_CPU_NAME"] = "generic"
        os.environ["NUMBA_CACHE_DIR"] = str(target)
        if target.is_dir():
            return True
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root) as staging:
            staged = Path(staging) / "kernels"
            for name in bundle.namelist():
                member = Path(name)
                if member.suffix not in (".nbi", ".nbc"):
                    continue
                folder = staged / _cache_folder(member.parent.as_posix())
                folder.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, (folder / member.name).open("wb") as dest:
                    shutil.copyfileobj(source, dest)
            try:
                staged.rename(target)
            except OSError:
                if not target.is_dir():
                    raise
    return True


if __name__ == "__main__":
    build(Path(sys.argv[1]))
