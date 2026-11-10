"""Persistent GPU preview of the recovered profiles, with a fixed source catalog.

The CPU export remains the numerical reference. Reduced views integrate the
transverse ray footprint instead of selecting isolated full-resolution pixels.
"""

import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
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
    sine,
)
from ..domain.effects.settings import KINDS
from ..domain.imaging import ImageFrame, check_cancel


class GpuPreviewRenderer:
    """Create on the GUI thread; render/close on the single persistent job thread."""

    def __init__(self):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.surface = QOffscreenSurface()
        self.surface.setFormat(fmt)
        self.surface.create()
        self.context = None
        self.program = self.vao = self.gl = None
        self.catalog = None
        self.catalog_mask = None
        self.crop_key = None
        self.buffers = {}
        self.target = None
        self.group_cache = {}
        self.renderer_name = ""
        self.uniform_locations = {}
        self.uniform_values = {}
        self.source_texture = None
        self.texture_key = None
        self.reduction_cache = None

    def _attributes(self, buffer, rays):
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

    def _initialize(self):
        if self.context is not None:
            if not self.context.makeCurrent(self.surface):
                raise RuntimeError("Cannot activate the preview OpenGL context.")
            return
        self.context = QOpenGLContext()
        self.context.setFormat(self.surface.format())
        if not self.context.create() or not self.context.makeCurrent(self.surface):
            raise RuntimeError("OpenGL 3.3 is required for the live preview.")
        self.gl = QOpenGLFunctions_3_3_Core()
        if not self.gl.initializeOpenGLFunctions():
            raise RuntimeError("Cannot initialize the preview GPU functions.")
        self.renderer_name = self.gl.glGetString(0x1F01)
        self.program = QOpenGLShaderProgram()
        folder = Path(__file__).with_name("shaders")
        vertex = (folder / "preview.vert").read_text(encoding="utf-8")
        fragment = (folder / "preview.frag").read_text(encoding="utf-8")
        for kind, code in ((QOpenGLShader.Vertex, vertex), (QOpenGLShader.Fragment, fragment)):
            if not self.program.addShaderFromSourceCode(kind, code):
                raise RuntimeError(self.program.log())
        if not self.program.link():
            raise RuntimeError(self.program.log())
        self.uniform_locations.clear()
        self.uniform_values.clear()
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

    def __call__(
        self,
        frame,
        stars,
        settings,
        cancel=None,
        *,
        full_resolution=False,
        interactive=False,
        region=None,
        region_sampling=1,
        max_edge=1600,
        sky_mask=None,
    ):
        check_cancel(cancel)
        if region is None:
            region = (0, 0, frame.width, frame.height)
            sampling = (
                1
                if full_resolution
                else max(1, (max(frame.width, frame.height) + max_edge - 1) // max_edge)
            )
            is_region = False
            sampling = min(sampling, min(frame.width, frame.height) - 1)
        else:
            sampling, is_region = region_sampling, True
        left, top, right, bottom = region
        try:
            source_key = weakref.ref(frame)
        except TypeError:
            source_key = frame
        crop_key = source_key, region, sampling
        if self.crop_key != crop_key:
            self.crop = ImageFrame(
                frame.pixels[top:bottom:sampling, left:right:sampling],
                frame.name,
                frame.color_space,
                frame.note,
            )
            self.crop_key = crop_key
        height, width, channels = self.crop.pixels.shape
        if self.catalog is not stars or self.catalog_mask is not sky_mask:
            self.catalog = stars
            self.catalog_mask = sky_mask
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
            self.buffer_keys = {}
            self.group_cache.clear()
            self.reduction_cache = None
        self._initialize()
        gl, program = self.gl, self.program
        self.vao.bind()
        if self.target is None or (self.target.width(), self.target.height()) != (width, height):
            self.target = None
            fmt = QOpenGLFramebufferObjectFormat()
            fmt.setInternalTextureFormat(0x8814)  # RGBA32F: preserve linear HDR calculations.
            self.target = QOpenGLFramebufferObject(width, height, fmt)
            if not self.target.isValid():
                raise RuntimeError("Cannot allocate the preview framebuffer.")
        self.target.bind()
        gl.glViewport(0, 0, width, height)
        gl.glDisable(0x0B71)  # depth
        gl.glEnable(0x0BE2)  # blend
        gl.glBlendEquation(0x8006)  # FUNC_ADD
        gl.glBlendFunc(1, 0x0301)  # ONE, ONE_MINUS_SRC_COLOR: native Screen.
        gl.glClearColor(0, 0, 0, 0)
        program.bind()

        def uniform(name, value):
            comparable = (value.x(), value.y()) if isinstance(value, QVector2D) else value
            if self.uniform_values.get(name) == comparable:
                return
            if name not in self.uniform_locations:
                self.uniform_locations[name] = program.uniformLocation(name.encode())
            location = self.uniform_locations[name]
            if isinstance(value, QVector2D):
                gl.glUniform2f(location, value.x(), value.y())
            elif type(value) is int:
                gl.glUniform1i(location, value)
            else:
                gl.glUniform1f(location, value)
            self.uniform_values[name] = comparable

        diagonal = image_diagonal(frame.width, frame.height)
        scale = 800 / max(diagonal, 1414)
        uniform("outputSize", QVector2D(width, height))
        uniform("origin", QVector2D(left, top))
        uniform("sampling", float(sampling))
        uniform("scale", float(scale))
        uniform("gray", int(channels == 1))
        groups = {}
        for kind_index, kind in enumerate(KINDS):
            check_cancel(cancel)
            p = getattr(settings, kind)
            if not p.enabled or not p.intensity or not p.quantity or not stars:
                continue
            group_key = replace(p, soft_filter=0), region, sampling, diagonal, channels, sky_mask
            cached = self.group_cache.get(kind)
            if cached is not None and cached[0] == group_key:
                groups[kind] = cached[1]
                continue
            gl.glClear(0x4000)
            threshold = (1 - p.quantity / 100) ** 2 * self.maximum
            key = id(stars), p.quantity
            if self.buffer_keys.get(kind) != key:
                records = self.records[(self.records[:, 5] > threshold) & (self.records[:, 6] > 0)]
                buffer = self.buffers.setdefault(kind, QOpenGLBuffer())
                if not buffer.isCreated():
                    buffer.create()
                buffer.bind()
                buffer.allocate(records.tobytes(), records.nbytes)
                self.buffer_keys[kind] = key
                self.buffer_counts = getattr(self, "buffer_counts", {})
                self.buffer_counts[kind] = len(records)
            count = self.buffer_counts[kind]
            if not count:
                continue
            rays = p.points if kind == "primary" else 84 if kind == "secondary" else 1
            self._attributes(self.buffers[kind], rays)
            sharp = (
                (p.sharpness / 90) ** 1.4
                if p.sharpness <= 90
                else (1 + 0.251416 * ((p.sharpness - 90) / 10) ** 1.4 / scale)
            )
            coefficient = (4 * sharp + 0.08) * scale
            values = dict(
                kind=kind_index,
                rayCount=rays,
                cutoff=radial_extent(kind, p, diagonal),
                coefficient=coefficient,
                widening=0.15 * (p.spread / 100) ** 2 * coefficient,
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
                uniform(
                    name,
                    value if type(value) is int and name in ("kind", "rayCount") else float(value),
                )
            for index, (angle, weight) in enumerate(directions(kind, p)):
                uniform(f"rayDirections[{index}]", QVector2D(sine(angle), -sine(angle + 900)))
                uniform(f"rayWeights[{index}]", float(weight))
            gl.glDrawArraysInstanced(0x0004, 0, 6, count * rays)
            error = gl.glGetError()
            if error:
                raise RuntimeError(f"GPU drawing {kind}: {error:#x}")
            output = np.empty((height, width, 4), np.float32)
            gl.glReadPixels(0, 0, width, height, 0x1908, 0x1406, VoidPtr(output.ctypes.data))
            error = gl.glGetError()
            if error:
                raise RuntimeError(f"GPU preview error: 0x{error:04x}")
            group = np.ascontiguousarray(output[::-1, :, :channels])
            if sky_mask is not None:
                group *= sky_mask.region(region, sampling)[..., None]
            group.flags.writeable = False
            groups[kind] = group
            self.group_cache[kind] = group_key, group
        effects = np.zeros_like(self.crop.pixels)
        for group in groups.values():
            effects += (1 - effects) * group
        reduction = None
        enlargement = None
        p = settings.soft
        if p.enabled and p.soft_filter and stars:
            key = p.soft_filter, crop_key, sky_mask
            if self.reduction_cache is not None and self.reduction_cache[0] == key:
                reduction, enlargement = self.reduction_cache[1:]
            else:
                if self.texture_key != crop_key:
                    if self.source_texture is not None:
                        self.source_texture.destroy()
                    self.source_texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
                    self.source_texture.setSize(width, height)
                    self.source_texture.setMipLevels(1)
                    self.source_texture.setMinMagFilters(
                        QOpenGLTexture.Filter.Nearest, QOpenGLTexture.Filter.Nearest
                    )
                    self.source_texture.setFormat(QOpenGLTexture.TextureFormat.RGB32F)
                    self.source_texture.allocateStorage(
                        QOpenGLTexture.PixelFormat.RGB, QOpenGLTexture.PixelType.Float32
                    )
                    pixels = (
                        self.crop.pixels
                        if channels == 3
                        else np.repeat(self.crop.pixels, 3, axis=2)
                    )
                    pixels = np.ascontiguousarray(pixels)
                    self.source_texture.setData(
                        QOpenGLTexture.PixelFormat.RGB,
                        QOpenGLTexture.PixelType.Float32,
                        VoidPtr(pixels.ctypes.data),
                    )
                    self.texture_key = crop_key
                self.source_texture.bind(0)
                uniform("originalImage", 0)
                uniform("rayCount", 1)
                uniform("softFilter", float(p.soft_filter / 100))
                passes = []
                for mode in (4, 5):
                    check_cancel(cancel)
                    uniform("kind", mode)
                    buffer_name = "reduction" if mode == 4 else "enlargement"
                    if self.buffer_keys.get(buffer_name) != id(stars):
                        eligible = (self.records[:, 6] > 0) & (self.records[:, 8] > 0)
                        eligible &= (
                            self.records[:, 7] < 0.5
                            if mode == 4
                            else (
                                (self.records[:, 7] > 0.5)
                                & (self.records[:, 18:21].max(axis=1) > 0)
                            )
                        )
                        records = self.records[eligible]
                        buffer = self.buffers.setdefault(buffer_name, QOpenGLBuffer())
                        if not buffer.isCreated():
                            buffer.create()
                        buffer.bind()
                        buffer.allocate(records.tobytes(), records.nbytes)
                        self.buffer_keys[buffer_name] = id(stars)
                        self.buffer_counts = getattr(self, "buffer_counts", {})
                        self.buffer_counts[buffer_name] = len(records)
                    self._attributes(self.buffers[buffer_name], 1)
                    gl.glClear(0x4000)
                    gl.glBlendEquation(0x8008)  # MAX bounds overlapping source corrections.
                    gl.glDrawArraysInstanced(0x0004, 0, 6, self.buffer_counts[buffer_name])
                    output = np.empty((height, width, 4), np.float32)
                    gl.glReadPixels(
                        0, 0, width, height, 0x1908, 0x1406, VoidPtr(output.ctypes.data)
                    )
                    if gl.glGetError():
                        raise RuntimeError("GPU source prefilter failed.")
                    correction = np.ascontiguousarray(output[::-1, :, :channels])
                    if sky_mask is not None:
                        correction *= sky_mask.region(region, sampling)[..., None]
                    correction.flags.writeable = False
                    passes.append(correction)
                reduction, enlargement = passes
                self.reduction_cache = key, reduction, enlargement
                gl.glBlendEquation(0x8006)
        self.target.release()
        program.release()
        self.context.doneCurrent()
        check_cancel(cancel)
        return Preview(
            self.crop,
            effects,
            sampling,
            origin=(left, top),
            is_region=is_region,
            groups=groups,
            reduction=reduction,
            enlargement=enlargement,
        )

    def close(self):
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
        if self.vao is not None:
            self.vao.destroy()
        if self.program is not None:
            self.program.removeAllShaders()
        self.program = None
        self.vao = None
        self.gl = None
        self.context.doneCurrent()
        self.context = None
