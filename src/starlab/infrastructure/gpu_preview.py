"""Fixed-catalog OpenGL preview stages; CPU export remains the numerical reference."""

import weakref
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat, QVector2D
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLFunctions_3_3_Core,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from shiboken6 import VoidPtr

from ..application.use_cases import Preview
from ..domain.effects.profiles import (
    directions,
    gain,
    image_diagonal,
    radial_extent,
    ray_coefficients,
    sine,
)
from ..domain.effects.settings import KINDS, EffectParameters, EffectSettings
from ..domain.imaging import ImageFrame, Pixels, check_cancel
from ..domain.sky_mask import SkyMask
from ..domain.stars import Star

Bounds = tuple[int, int, int, int]


def _view_region(
    frame: ImageFrame, region: Bounds | None, sampling: int, full_resolution: bool, max_edge: int
) -> tuple[Bounds, int, bool]:
    """Choose integer sampling with at least two samples per full-image axis."""
    if region is not None:
        return region, sampling, True
    stride = (
        1
        if full_resolution
        else max(1, (max(frame.width, frame.height) + max_edge - 1) // max_edge)
    )
    return (0, 0, frame.width, frame.height), min(stride, min(frame.width, frame.height) - 1), False


class GpuPreviewRenderer:
    """Create the surface on the GUI thread; render/close on one persistent worker."""

    def __init__(self):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.surface = QOffscreenSurface()
        self.surface.setFormat(fmt)
        self.surface.create()
        self.context = None
        self.program = self.vao = self.gl = None
        self.catalog = self.catalog_mask = None
        self.crop_key = None
        self.buffers = {}
        self.buffer_keys = {}
        self.buffer_counts = {}
        self.target = None
        self.group_cache = {}
        self.renderer_name = ""
        self.uniform_locations = {}
        self.uniform_values = {}
        self.source_texture = None
        self.texture_key = None
        self.reduction_cache = None

    def _attributes(self, buffer: QOpenGLBuffer, rays: int) -> None:
        """Bind the 21-float source layout and per-ray instance divisors."""
        buffer.bind()
        for location, offset, size in (
            (0, 0, 2),
            (1, 8, 3),
            (2, 20, 1),
            (3, 28, 1),
            (4, 32, 1),
            (5, 36, 3),
            (6, 48, 3),
            (7, 60, 3),
            (8, 72, 3),
        ):
            self.program.enableAttributeArray(location)
            self.program.setAttributeBuffer(location, 0x1406, offset, size, 84)
            self.gl.glVertexAttribDivisor(location, rays)

    def _initialize(self) -> None:
        """Activate the worker-owned context and build its shader program once."""
        if self.context is not None:
            if not self.context.makeCurrent(self.surface):
                raise RuntimeError("Cannot activate the preview OpenGL context.")
            return
        self.context = QOpenGLContext()
        self.context.setFormat(self.surface.format())
        if not self.context.create() or not self.context.makeCurrent(self.surface):
            self.context = None
            raise RuntimeError("OpenGL 3.3 is required for the live preview.")
        try:
            self._build_program()
        except Exception:
            self.close()
            raise

    def _build_program(self) -> None:
        """Compile packaged GLSL and create the context's vertex-array binding."""
        self.gl = QOpenGLFunctions_3_3_Core()
        if not self.gl.initializeOpenGLFunctions():
            raise RuntimeError("Cannot initialize the preview GPU functions.")
        self.renderer_name = self.gl.glGetString(0x1F01)
        self.program = QOpenGLShaderProgram()
        folder = Path(__file__).with_name("shaders")
        for kind, name in (
            (QOpenGLShader.Vertex, "preview.vert"),
            (QOpenGLShader.Fragment, "preview.frag"),
        ):
            if not self.program.addShaderFromSourceCode(
                kind, (folder / name).read_text(encoding="utf-8")
            ):
                raise RuntimeError(self.program.log())
        if not self.program.link():
            raise RuntimeError(self.program.log())
        self.uniform_locations.clear()
        self.uniform_values.clear()
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

    def _set_crop(self, frame: ImageFrame, region: Bounds, sampling: int) -> None:
        """Copy only a changed sampled region, retaining source identity in its key."""
        try:
            source_key = weakref.ref(frame)
        except TypeError:
            source_key = frame
        key = source_key, region, sampling
        if self.crop_key == key:
            return
        left, top, right, bottom = region
        self.crop = ImageFrame(
            frame.pixels[top:bottom:sampling, left:right:sampling],
            frame.name,
            frame.color_space,
            frame.note,
        )
        self.crop_key = key

    def _set_catalog(self, stars: tuple[Star, ...], sky_mask: SkyMask | None) -> None:
        """Repack a changed catalog and invalidate buffers derived from its identity."""
        if self.catalog is stars and self.catalog_mask is sky_mask:
            return
        self.catalog, self.catalog_mask = stars, sky_mask
        self.records = np.asarray(
            [
                (
                    s.x,
                    s.y,
                    *s.color,
                    s.brightness,
                    float(s.visible),
                    s.soft_weight,
                    *(s.source_profile or (0,) * 10),
                    *s.source_peak,
                )
                for s in stars
            ],
            dtype=np.float32,
        ).reshape(-1, 21)
        if sky_mask is not None and stars:
            self.records = self.records[sky_mask.keep(self.records[:, :2])]
        self.maximum = float(self.records[:, 5].max()) if len(self.records) else 0
        self.buffer_keys.clear()
        self.group_cache.clear()
        self.reduction_cache = None

    def _bind_target(self) -> None:
        """Allocate a size-matched float framebuffer and reset Screen blending."""
        height, width, _ = self.crop.pixels.shape
        self.vao.bind()
        if self.target is None or (self.target.width(), self.target.height()) != (width, height):
            self.target = None
            fmt = QOpenGLFramebufferObjectFormat()
            fmt.setInternalTextureFormat(0x8814)
            self.target = QOpenGLFramebufferObject(width, height, fmt)
            if not self.target.isValid():
                raise RuntimeError("Cannot allocate the preview framebuffer.")
        self.target.bind()
        self.gl.glViewport(0, 0, width, height)
        self.gl.glDisable(0x0B71)
        self.gl.glEnable(0x0BE2)
        self.gl.glBlendEquation(0x8006)
        self.gl.glBlendFunc(1, 0x0301)  # ONE, ONE_MINUS_SRC_COLOR
        self.gl.glClearColor(0, 0, 0, 0)
        self.program.bind()

    def _uniform(self, name: str, value: QVector2D | float | int) -> None:
        """Upload a changed uniform, caching its location and normalized value."""
        comparable = (value.x(), value.y()) if isinstance(value, QVector2D) else value
        if self.uniform_values.get(name) == comparable:
            return
        if name not in self.uniform_locations:
            self.uniform_locations[name] = self.program.uniformLocation(name.encode())
        location = self.uniform_locations[name]
        if isinstance(value, QVector2D):
            self.gl.glUniform2f(location, value.x(), value.y())
        elif type(value) is int:
            self.gl.glUniform1i(location, value)
        else:
            self.gl.glUniform1f(location, value)
        self.uniform_values[name] = comparable

    def _view_uniforms(self, region: Bounds, sampling: int, diagonal: int) -> None:
        """Map framebuffer samples to original full-image coordinates."""
        height, width, channels = self.crop.pixels.shape
        self._uniform("outputSize", QVector2D(width, height))
        self._uniform("origin", QVector2D(*region[:2]))
        self._uniform("sampling", float(sampling))
        self._uniform("scale", float(800 / max(diagonal, 1414)))
        self._uniform("gray", int(channels == 1))

    def _upload_buffer(self, name: str, key: object, records: NDArray[np.float32]) -> None:
        """Replace one instance buffer and record its source-selection key."""
        buffer = self.buffers.setdefault(name, QOpenGLBuffer())
        if not buffer.isCreated():
            buffer.create()
        buffer.bind()
        buffer.allocate(records.tobytes(), records.nbytes)
        self.buffer_keys[name], self.buffer_counts[name] = key, len(records)

    def _effect_buffer(self, kind: str, quantity: float, threshold: float) -> int:
        """Select visible sources once per quantity/catalog change."""
        key = id(self.catalog), quantity
        if self.buffer_keys.get(kind) != key:
            selected = (self.records[:, 5] > threshold) & (self.records[:, 6] > 0)
            self._upload_buffer(kind, key, self.records[selected])
        return self.buffer_counts[kind]

    def _effect_uniforms(
        self, kind: str, p: EffectParameters, diagonal: int, threshold: float, rays: int
    ) -> None:
        """Configure one effect profile without changing its source catalog."""
        coefficient, widening = ray_coefficients(p, 800 / max(diagonal, 1414))
        values = dict(
            kind=KINDS.index(kind),
            rayCount=rays,
            cutoff=radial_extent(kind, p, diagonal),
            coefficient=coefficient,
            widening=widening,
            rayLength=p.length,
            intensity=gain(kind, p),
            threshold=threshold,
            hue=p.hue,
            saturation=p.saturation,
            rainbow=p.rainbow,
            frequency=p.frequency,
            phase=p.phase,
            softSize=1 + (max(diagonal, 1414) // 3) * (p.size / 100) ** 3,
            softPower=0.6 + 0.124 * p.falloff,
            ringPeak=(1 + 0.01 * (p.size + 25) ** 2) * max(diagonal, 1414) / 3768,
            innerWidth=p.inner_width,
            outerWidth=p.outer_width,
            innerPower=0.08 * (p.inner_falloff + 10),
            outerPower=0.08 * (p.outer_falloff + 10),
        )
        for name, value in values.items():
            self._uniform(
                name, value if type(value) is int and name in ("kind", "rayCount") else float(value)
            )
        for index, (angle, weight) in enumerate(directions(kind, p)):
            self._uniform(f"rayDirections[{index}]", QVector2D(sine(angle), -sine(angle + 900)))
            self._uniform(f"rayWeights[{index}]", float(weight))

    def _draw(self, count: int, rays: int) -> None:
        """Draw source quads and surface errors at the GL operation that failed."""
        self.gl.glDrawArraysInstanced(0x0004, 0, 6, count * rays)
        error = self.gl.glGetError()
        if error:
            raise RuntimeError(f"GPU drawing error: 0x{error:04x}")

    def _read_target(self, region: Bounds, sampling: int, sky_mask: SkyMask | None) -> Pixels:
        """Read an immutable top-down linear buffer and apply foreground protection."""
        height, width, channels = self.crop.pixels.shape
        output = np.empty((height, width, 4), np.float32)
        self.gl.glReadPixels(0, 0, width, height, 0x1908, 0x1406, VoidPtr(output.ctypes.data))
        error = self.gl.glGetError()
        if error:
            raise RuntimeError(f"GPU readback error: 0x{error:04x}")
        pixels = np.ascontiguousarray(output[::-1, :, :channels])
        if sky_mask is not None:
            pixels *= sky_mask.region(region, sampling)[..., None]
        pixels.flags.writeable = False
        return pixels

    def _effect_group(
        self,
        kind: str,
        p: EffectParameters,
        region: Bounds,
        sampling: int,
        diagonal: int,
        sky_mask: SkyMask | None,
    ) -> Pixels | None:
        """Reuse or draw one effect layer, independently of source correction."""
        key = (
            replace(p, soft_filter=0),
            region,
            sampling,
            diagonal,
            self.crop.pixels.shape[2],
            sky_mask,
        )
        cached = self.group_cache.get(kind)
        if cached is not None and cached[0] == key:
            return cached[1]
        threshold = (1 - p.quantity / 100) ** 2 * self.maximum
        count = self._effect_buffer(kind, p.quantity, threshold)
        if not count:
            return None
        rays = p.points if kind == "primary" else 84 if kind == "secondary" else 1
        self._attributes(self.buffers[kind], rays)
        self._effect_uniforms(kind, p, diagonal, threshold, rays)
        self.gl.glClear(0x4000)
        self._draw(count, rays)
        group = self._read_target(region, sampling, sky_mask)
        self.group_cache[kind] = key, group
        return group

    def _bind_source_texture(self) -> None:
        """Upload original sampled pixels only when the viewport changes."""
        if self.texture_key != self.crop_key:
            height, width, channels = self.crop.pixels.shape
            if self.source_texture is not None:
                self.source_texture.destroy()
            texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            texture.setSize(width, height)
            texture.setMipLevels(1)
            texture.setMinMagFilters(QOpenGLTexture.Filter.Nearest, QOpenGLTexture.Filter.Nearest)
            texture.setFormat(QOpenGLTexture.TextureFormat.RGB32F)
            texture.allocateStorage(
                QOpenGLTexture.PixelFormat.RGB, QOpenGLTexture.PixelType.Float32
            )
            pixels = self.crop.pixels if channels == 3 else np.repeat(self.crop.pixels, 3, axis=2)
            pixels = np.ascontiguousarray(pixels)
            texture.setData(
                QOpenGLTexture.PixelFormat.RGB,
                QOpenGLTexture.PixelType.Float32,
                VoidPtr(pixels.ctypes.data),
            )
            self.source_texture, self.texture_key = texture, self.crop_key
        self.source_texture.bind(0)

    def _correction_buffer(self, mode: int) -> str:
        """Select measured small/large sources, without halo quantity filtering."""
        name = "reduction" if mode == 4 else "enlargement"
        key = id(self.catalog)
        if self.buffer_keys.get(name) != key:
            eligible = (self.records[:, 6] > 0) & (self.records[:, 8] > 0)
            eligible &= (
                self.records[:, 7] < 0.5
                if mode == 4
                else ((self.records[:, 7] > 0.5) & (self.records[:, 18:21].max(axis=1) > 0))
            )
            self._upload_buffer(name, key, self.records[eligible])
        return name

    def _source_filter(
        self,
        p: EffectParameters,
        region: Bounds,
        sampling: int,
        sky_mask: SkyMask | None,
        cancel: Event | None,
    ) -> tuple[Pixels | None, Pixels | None]:
        """Return bounded corrections for the source pixels before Screen composition."""
        if not p.enabled or not p.soft_filter or not self.catalog:
            return None, None
        key = p.soft_filter, self.crop_key, sky_mask
        if self.reduction_cache is not None and self.reduction_cache[0] == key:
            return self.reduction_cache[1:]
        self._bind_source_texture()
        self._uniform("originalImage", 0)
        self._uniform("rayCount", 1)
        self._uniform("softFilter", float(p.soft_filter / 100))
        passes = []
        self.gl.glBlendEquation(0x8008)
        try:
            for mode in (4, 5):
                check_cancel(cancel)
                self._uniform("kind", mode)
                name = self._correction_buffer(mode)
                self._attributes(self.buffers[name], 1)
                self.gl.glClear(0x4000)
                self._draw(self.buffer_counts[name], 1)
                passes.append(self._read_target(region, sampling, sky_mask))
        finally:
            self.gl.glBlendEquation(0x8006)
        reduction, enlargement = passes
        self.reduction_cache = key, reduction, enlargement
        return reduction, enlargement

    def __call__(
        self,
        frame: ImageFrame,
        stars: tuple[Star, ...],
        settings: EffectSettings,
        cancel: Event | None = None,
        *,
        full_resolution: bool = False,
        interactive: bool = False,
        region: Bounds | None = None,
        region_sampling: int = 1,
        max_edge: int = 1600,
        sky_mask: SkyMask | None = None,
    ) -> Preview:
        """Render one view and release active GL bindings on success or failure."""
        check_cancel(cancel)
        region, sampling, is_region = _view_region(
            frame, region, region_sampling, full_resolution, max_edge
        )
        self._set_crop(frame, region, sampling)
        self._set_catalog(stars, sky_mask)
        self._initialize()
        try:
            self._bind_target()
            diagonal = image_diagonal(frame.width, frame.height)
            self._view_uniforms(region, sampling, diagonal)
            groups = {}
            for kind in KINDS:
                check_cancel(cancel)
                p = getattr(settings, kind)
                if p.enabled and p.intensity and p.quantity and stars:
                    group = self._effect_group(kind, p, region, sampling, diagonal, sky_mask)
                    if group is not None:
                        groups[kind] = group
            effects = np.zeros_like(self.crop.pixels)
            for group in groups.values():
                effects += (1 - effects) * group
            reduction, enlargement = self._source_filter(
                settings.soft, region, sampling, sky_mask, cancel
            )
            check_cancel(cancel)
            return Preview(
                self.crop,
                effects,
                sampling,
                origin=region[:2],
                is_region=is_region,
                groups=groups,
                reduction=reduction,
                enlargement=enlargement,
            )
        finally:
            if self.target is not None:
                self.target.release()
            self.program.release()
            self.context.doneCurrent()

    def close(self) -> None:
        """Release GL objects on their worker thread; repeated close calls are safe."""
        if self.context is None:
            return
        if not self.context.makeCurrent(self.surface):
            self.context = None
            return
        self.target = None
        if self.source_texture is not None:
            self.source_texture.destroy()
            self.source_texture = None
        for buffer in self.buffers.values():
            buffer.destroy()
        self.buffers.clear()
        self.buffer_keys.clear()
        self.group_cache.clear()
        self.reduction_cache = None
        self.catalog = self.catalog_mask = self.texture_key = None
        if self.vao is not None:
            self.vao.destroy()
        if self.program is not None:
            self.program.removeAllShaders()
        self.program = self.vao = self.gl = None
        self.context.doneCurrent()
        self.context = None
