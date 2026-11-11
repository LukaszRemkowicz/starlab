import numpy as np

from starlab.infrastructure.gpu_preview import GpuPreviewRenderer


def test_close_without_current_context_still_drops_cpu_state(qapp):
    class LostContext:
        def makeCurrent(self, surface):
            return False

    renderer = GpuPreviewRenderer()
    renderer.context = LostContext()
    renderer.group_cache["primary"] = ((), np.ones((4, 4, 3), np.float32))
    renderer.reduction_cache = object()
    renderer.buffer_keys["primary"] = (1, 50.0)
    renderer.close()
    assert renderer.context is None
    assert not renderer.group_cache and not renderer.buffer_keys and not renderer.buffers
    assert renderer.reduction_cache is None
