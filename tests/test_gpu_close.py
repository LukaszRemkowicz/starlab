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


def test_gpu_region_is_clipped_and_validated_like_the_cpu_path():
    import pytest

    from starlab.domain.imaging import ImageFrame
    from starlab.infrastructure.gpu_preview import _view_region

    frame = ImageFrame(np.zeros((40, 60, 3), np.float32))
    assert _view_region(frame, (-5, -3, 80, 50), 1, False, 1600)[0] == (0, 0, 60, 40)
    with pytest.raises(ValueError):
        _view_region(frame, (10, 10, 11, 11), 1, False, 1600)
