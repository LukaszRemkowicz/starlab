"""The installed detector loads machine code on a user's first launch."""

import os
import subprocess
import sys
from pathlib import Path


def test_installed_kernels_work_with_an_empty_user_cache(tmp_path):
    source = Path(__file__).parent / "fixtures/detection/pano-sky-middle.tif"
    # Dispatchers choose their cache during import. Use a fresh interpreter as the GUI does.
    probe = """
import sys
from pathlib import Path
from starlab.kernel_bundle import install
assert install(Path(sys.argv[1]))
from starlab.bootstrap import create_services
from starlab.domain.detection.compiled import scan_block
from starlab.domain.photometry import _measure_batch
source = Path(sys.argv[2])
services = create_services(sky_mask=True)
frame, mask = services.open_image(source)
stars = services.analyze(frame, source, sky_mask=mask)
assert len(stars) > 0
for kernel in (scan_block, _measure_batch):
    assert sum(kernel.stats.cache_hits.values()) > 0
    assert sum(kernel.stats.cache_misses.values()) == 0
"""
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "cache")}
    env.pop("NUMBA_CPU_NAME", None)
    env.pop("NUMBA_CPU_FEATURES", None)
    subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path), str(source)], env=env, check=True, timeout=120
    )
