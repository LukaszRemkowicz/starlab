"""Opt-in numerical checks using a real desktop OpenGL context and production shaders."""

from dataclasses import replace
from threading import get_ident

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtGui import QGuiApplication, QImage, QOpenGLContext

from starlab.application.use_cases import render_preview
from starlab.bootstrap import create_window
from starlab.domain.effects.render import composite
from starlab.domain.effects.settings import KINDS, EffectSettings
from starlab.domain.imaging import ImageFrame, display_pixels
from starlab.domain.sky_mask import SkyMask
from starlab.domain.stars import Star
from starlab.infrastructure.gpu_preview import GpuPreviewRenderer
from starlab.presentation.qt.worker import JobRunner

pytestmark = pytest.mark.gpu


def _scene(channels):
    y, x = np.mgrid[:96, :160]
    pixels = np.full((96, 160, 3), 0.02, np.float32)
    stars = []
    for number, (sx, sy, sigma, weight, rgb) in enumerate(
        [(42.25, 37.4, 1.5, 0.0, (0.2, 0.4, 0.6)), (108.5, 52.75, 4.0, 1.0, (0.6, 0.3, 0.2))],
        start=1,
    ):
        profile = (sigma * 2.354820045, 0.02, 0.02, 0.02, 0, 0, 0, 0, 0, 0)
        pixels += (
            np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * sigma**2))[..., None] * np.asarray(rgb)
        ).astype(np.float32)
        stars.append(
            Star(
                number,
                sx,
                sy,
                0.8 / number,
                rgb,
                source_profile=profile,
                source_peak=rgb,
                soft_weight=weight,
            )
        )
    if channels == 1:
        pixels = pixels.mean(axis=2, keepdims=True)
    return ImageFrame(pixels), tuple(stars)


def _settings(kind):
    base = EffectSettings.disabled()
    return replace(
        base,
        **{
            kind: replace(
                getattr(base, kind), enabled=True, quantity=100, intensity=55, hue=17, saturation=65
            )
        },
    )


def _all_settings():
    return replace(
        EffectSettings.disabled(),
        **{
            kind: replace(getattr(_settings(kind), kind), soft_filter=65, quantity=70)
            for kind in KINDS
        },
    )


def _sky_mask():
    labels = np.ones((96, 160), np.uint8)
    labels[70:] = 0
    return SkyMask(1, labels, labels.shape)


def _assert_parity(actual, expected, *, secondary=False):
    # Dense secondary rays have a separate <0.1% linear-light error budget.
    # Primary and radial full-resolution profiles use the tighter float32 budget.
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-3 if secondary else 2e-5)
    assert float(np.mean(np.abs(actual - expected))) < 3e-6


@pytest.fixture
def gpu(qtbot):
    assert QGuiApplication.platformName() != "offscreen"
    renderer = GpuPreviewRenderer()
    runner = JobRunner()
    results, failures, worker_threads, cleanup_threads = [], [], [], []
    gui_thread = get_ident()
    runner.completed.connect(lambda kind, value: results.append(value))
    runner.failed.connect(failures.append)

    def render(frame, stars, settings, **options):
        results.clear()
        failures.clear()

        def operation(cancel):
            worker_threads.append(get_ident())
            result = renderer(frame, stars, settings, cancel, **options)
            assert renderer.context.thread() == QThread.currentThread()
            assert QOpenGLContext.currentContext() is None
            return result

        runner.submit("preview", operation)
        qtbot.waitUntil(lambda: not runner.busy, timeout=10000)
        assert not failures, failures
        assert len(results) == 1
        return results[0]

    def cleanup():
        cleanup_threads.append(get_ident())
        renderer.close()
        renderer.close()  # Releasing an already released context is harmless.

    runner.cleanup = cleanup
    try:
        yield render, renderer
    finally:
        assert runner.shutdown()
        assert len(set(worker_threads + cleanup_threads)) == 1
        assert cleanup_threads[0] != gui_thread
        assert renderer.context is None and renderer.target is None
        assert renderer.source_texture is None and not renderer.buffers


@pytest.mark.parametrize("channels", [1, 3], ids=["gray", "rgb"])
@pytest.mark.parametrize("kind", KINDS)
def test_full_resolution_groups_match_cpu(gpu, channels, kind, record_property):
    render, renderer = gpu
    frame, stars = _scene(channels)
    settings = _settings(kind)
    expected = render_preview(frame, stars, settings, full_resolution=True)
    actual = render(frame, stars, settings, full_resolution=True)
    record_property("opengl_renderer", renderer.renderer_name)
    error = float(np.max(np.abs(actual.effects - expected.effects)))
    record_property("maximum_absolute_error", error)
    assert renderer.renderer_name
    assert actual.scale == 1 and not actual.is_region
    assert set(actual.groups) == {kind}
    _assert_parity(actual.effects, expected.effects, secondary=kind == "secondary")


@pytest.mark.parametrize("channels", [1, 3], ids=["gray", "rgb"])
def test_groups_compose_with_both_source_corrections(gpu, channels):
    render, _ = gpu
    frame, stars = _scene(channels)
    settings = _all_settings()
    expected = render_preview(frame, stars, settings, full_resolution=True)
    actual = render(frame, stars, settings, full_resolution=True)
    assert set(actual.groups) == set(KINDS)
    combined = np.zeros_like(actual.effects)
    for group in actual.groups.values():
        combined += (1 - combined) * group
    np.testing.assert_array_equal(actual.effects, combined)
    assert np.max(actual.reduction) > 0 and np.max(actual.enlargement) > 0
    np.testing.assert_allclose(actual.reduction, expected.reduction, rtol=1e-5, atol=3e-7)
    np.testing.assert_allclose(actual.enlargement, expected.enlargement, rtol=1e-5, atol=3e-7)
    _assert_parity(actual.effects, expected.effects, secondary=True)
    _assert_parity(
        composite(
            actual.frame, actual.effects, reduction=actual.reduction, enlargement=actual.enlargement
        ),
        composite(
            expected.frame,
            expected.effects,
            reduction=expected.reduction,
            enlargement=expected.enlargement,
        ),
        secondary=True,
    )


@pytest.mark.parametrize("channels", [1, 3], ids=["gray", "rgb"])
def test_source_corrections_are_independent_of_halo_quantity_and_intensity(gpu, channels):
    render, _ = gpu
    frame, stars = _scene(channels)
    base = EffectSettings.disabled()
    settings = replace(
        base, soft=replace(base.soft, enabled=True, soft_filter=65, quantity=0, intensity=0)
    )
    expected = render_preview(frame, stars, settings, full_resolution=True)
    actual = render(frame, stars, settings, full_resolution=True)
    assert not actual.effects.any() and actual.groups == {}
    assert actual.reduction.any() and actual.enlargement.any()
    np.testing.assert_allclose(actual.reduction, expected.reduction, rtol=1e-5, atol=3e-7)
    np.testing.assert_allclose(actual.enlargement, expected.enlargement, rtol=1e-5, atol=3e-7)
    disabled = render(
        frame,
        stars,
        replace(settings, soft=replace(settings.soft, enabled=False)),
        full_resolution=True,
    )
    assert disabled.reduction is None and disabled.enlargement is None


@pytest.mark.parametrize("channels", [1, 3], ids=["gray", "rgb"])
def test_native_roi_and_sky_mask_match_cpu(gpu, channels):
    render, _ = gpu
    frame, stars = _scene(channels)
    stars += (Star(3, 80, 80, 50, (1, 1, 1)),)
    settings, mask = _all_settings(), _sky_mask()
    region = (16, 8, 144, 88)
    options = dict(region=region, region_sampling=1, sky_mask=mask)
    expected = render_preview(frame, stars, settings, **options)
    actual = render(frame, stars, settings, **options)
    assert actual.is_region and actual.origin == region[:2] and actual.scale == 1
    assert actual.effects.shape == (80, 128, channels)
    _assert_parity(actual.effects, expected.effects, secondary=True)
    for layer in (actual.effects, actual.reduction, actual.enlargement):
        assert np.isfinite(layer).all()
        assert not layer[62:].any()  # Source y>=70 is foreground, including the bright source.
    np.testing.assert_allclose(actual.reduction, expected.reduction, rtol=1e-5, atol=3e-7)
    np.testing.assert_allclose(actual.enlargement, expected.enlargement, rtol=1e-5, atol=3e-7)


@pytest.mark.parametrize("channels", [1, 3], ids=["gray", "rgb"])
@pytest.mark.parametrize("kind", ["soft", "ring"])
def test_reduced_radial_views_obey_bounds_mask_and_repeatability(gpu, channels, kind):
    render, _ = gpu
    frame, stars = _scene(channels)
    settings, mask = _settings(kind), _sky_mask()
    options = dict(region=(16, 8, 144, 88), region_sampling=4, sky_mask=mask)
    actual = render(frame, stars, settings, **options)
    repeated = render(frame, stars, settings, **options)
    assert actual.scale == 4 and actual.origin == (16, 8) and actual.is_region
    assert actual.effects.shape == (20, 32, channels)
    assert np.isfinite(actual.effects).all()
    assert np.min(actual.effects) >= 0 and np.max(actual.effects) <= 1
    assert np.max(actual.effects) > 0
    assert not actual.effects[16:].any()
    np.testing.assert_array_equal(actual.effects, repeated.effects)
    # Reduced GPU radial views integrate the pixel footprint; CPU stride samples
    # use a different numerical contract, so no false point-sample equality here.


def _view_pixels(preview, mode, mask):
    original = preview.frame.pixels
    result = composite(
        preview.frame, preview.effects, reduction=preview.reduction, enlargement=preview.enlargement
    )
    if mode == "original":
        return original
    if mode == "effects":
        return preview.effects
    if mode == "mask":
        palette = np.array([[0.5, 0.03, 0.01], [0.01, 0.13, 0.5], [0.6, 0.4, 0.01]], np.float32)
        return original * 0.65 + palette[mask.classes] * 0.35
    if mode == "compare":
        result = result.copy()
        middle = original.shape[1] // 2
        result[:, :middle] = original[:, :middle]
        result[:, middle : middle + 1] = 0.5
    return result


def _pixmap_pixels(pixmap):
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    data = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return data[:, : image.width() * 3].reshape(image.height(), image.width(), 3).copy()


@pytest.mark.parametrize("channels", [1, 3], ids=["gray", "rgb"])
def test_production_gpu_window_displays_every_mode(qtbot, channels):
    window = create_window(backend="gpu")
    qtbot.addWidget(window)
    controller = window.controller
    renderer = controller.services.preview
    assert isinstance(renderer, GpuPreviewRenderer)
    assert controller.services.sky_mask is not None and controller.services.read_mask is not None
    frame, stars = _scene(channels)
    settings, mask = _all_settings(), _sky_mask()
    controller.session.open(frame)
    controller.session.set_settings(settings, remember=False)
    controller.session.set_catalog(stars, sky_mask=mask, remember=False)
    window.set_loaded(True)
    controller.refresh()
    qtbot.waitUntil(
        lambda: controller.preview is not None and not controller.runner.busy, timeout=10000
    )
    assert not window.error_label.text()
    assert renderer.context.thread() != window.thread()
    expected = render_preview(frame, stars, settings, full_resolution=True, sky_mask=mask)
    try:
        for mode in ("original", "mask", "effects", "compare", "result"):
            window._view(mode)
            pixels = display_pixels(_view_pixels(expected, mode, mask), frame.color_space)
            if pixels.shape[2] == 1:
                pixels = np.repeat(pixels, 3, axis=2)
            np.testing.assert_allclose(
                _pixmap_pixels(window.image_view.item.pixmap()), pixels, atol=1, rtol=0
            )
    finally:
        assert controller.shutdown()
    assert renderer.context is None and renderer.source_texture is None


@pytest.mark.parametrize("channels", [1, 3], ids=["gray", "rgb"])
def test_native_soft_filter_changes_halo_hierarchy_and_invalidates_group_cache(gpu, channels):
    render, _ = gpu
    frame, stars = _scene(channels)
    base = EffectSettings.disabled()
    settings = replace(
        base, soft=replace(base.soft, enabled=True, quantity=100, intensity=40, size=35)
    )
    previews = [
        render(
            frame,
            stars,
            replace(settings, soft=replace(settings.soft, soft_filter=value)),
            full_resolution=True,
        )
        for value in (0, 50, 100, 0, 100)
    ]
    small_halos = [preview.effects[37, 42, 0] for preview in previews]
    large_halos = [preview.effects[53, 109, 0] for preview in previews]
    assert small_halos[0] > small_halos[1] > small_halos[2]
    assert small_halos[2] == 0
    assert 0 < large_halos[0] < large_halos[1] < large_halos[2]
    np.testing.assert_array_equal(previews[0].effects, previews[3].effects)
    np.testing.assert_array_equal(previews[2].effects, previews[4].effects)
    expected = render_preview(
        frame,
        stars,
        replace(settings, soft=replace(settings.soft, soft_filter=100)),
        full_resolution=True,
    )
    _assert_parity(previews[2].effects, expected.effects)
