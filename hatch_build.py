"""Include platform-specific, precompiled image kernels in regular and editable wheels."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class KernelBuildHook(BuildHookInterface):
    """Attach the numerical kernels for the platform building the wheel."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Compile in a private cache that survives until the wheel is written."""
        self.staging = TemporaryDirectory(prefix="starlab-kernels-")
        stage = Path(self.staging.name)
        archive = stage / "kernels.zip"
        subprocess.run(
            [sys.executable, "-m", "starlab.kernel_bundle", str(archive)],
            cwd=Path(self.root) / "src",
            env={
                **os.environ,
                "NUMBA_CACHE_DIR": str(stage / "cache"),
                "NUMBA_CPU_NAME": "generic",
                "NUMBA_CPU_FEATURES": "",
            },
            check=True,
        )
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        inclusion = {str(archive): "_starlab_kernels/kernels.zip"}
        build_data["force_include"].update(inclusion)
        build_data["force_include_editable"].update(inclusion)

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        """Release staging files after Hatch has included them."""
        self.staging.cleanup()
