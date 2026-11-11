"""Coordinate immutable previews, original-pixel analysis and protected image export."""

from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import TypedDict

from ..domain.array_types import Bounds, DoubleArray, FloatArray
from ..domain.effects.profiles import image_diagonal, radial_extent
from ..domain.effects.render import render_effects
from ..domain.effects.settings import KINDS, EffectKind, EffectParameters, EffectSettings
from ..domain.effects.soft_filter import (
    FilterOptions,
    enlargement_records,
    reduction_records,
    soft_filter_sources,
    star_enlargement,
    star_reduction,
)
from ..domain.imaging import ImageFrame, Pixels, check_cancel, preview_region_bounds
from ..domain.photometry import calibrate_sources
from ..domain.sky_mask import SkyMask
from ..domain.stars import DetectionSettings, Star, detect_stars, filter_stars
from .ports import (
    ImageReader,
    ImageWriter,
    MaskReader,
    PreviewReader,
    PreviewRenderer,
    SkyMaskProvider,
)

# Rows rendered, corrected and encoded at once during export.
EXPORT_BAND_ROWS = 256


@dataclass(frozen=True)
class Preview:
    """Immutable layer snapshot with full-source pixel scale and optional ROI origin."""

    frame: ImageFrame
    effects: Pixels
    scale: int
    draft_count: tuple[int, int] | None = None
    origin: tuple[int, int] = (0, 0)
    is_region: bool = False
    groups: dict[EffectKind, Pixels] | None = None
    reduction: Pixels | None = None
    enlargement: Pixels | None = None


class _MaskOption(TypedDict, total=False):
    """Pass a mask only when supplied, preserving adapter call compatibility."""

    sky_mask: SkyMask


def preview_source_filter(
    pixels: Pixels,
    stars: tuple[Star, ...],
    settings: EffectSettings,
    cancel: Event | None = None,
    *,
    origin: tuple[int, int] = (0, 0),
    sampling: int = 1,
    sky_mask: SkyMask | None = None,
) -> tuple[Pixels, Pixels] | tuple[None, None]:
    """Build independent reduction/enlargement layers on the preview's source grid."""
    if not settings.soft.enabled or not settings.soft.soft_filter:
        return None, None
    sources: tuple[Star, ...] = soft_filter_sources(filter_stars(stars, sky_mask), settings.soft)
    options: FilterOptions = FilterOptions(
        origin=origin, sampling=sampling, sky_mask=sky_mask, cancel=cancel
    )
    reduction: FloatArray = star_reduction(
        pixels,
        reduction_records(sources),
        settings.soft.soft_filter,
        **options,
    )
    enlargement: FloatArray = star_enlargement(
        pixels,
        enlargement_records(sources, settings.soft.hue, settings.soft.saturation),
        settings.soft.soft_filter,
        size=settings.soft.size,
        falloff=settings.soft.falloff,
        **options,
    )
    return reduction, enlargement


@dataclass(frozen=True)
class _RegionView:
    """Read-only source view; sampling never copies the entire zoomed-out ROI."""

    pixels: Pixels

    @property
    def width(self) -> int:
        """Return the unsampled ROI width without copying its source pixels."""
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        """Return the unsampled ROI height without copying its source pixels."""
        return int(self.pixels.shape[0])


def _region_sources(
    stars: tuple[Star, ...],
    settings: EffectSettings,
    region: tuple[int, int, int, int],
    diagonal: int,
) -> tuple[Star, ...]:
    """Translate contributing sources while retaining the global brightness maximum."""
    left, top, right, bottom = region
    groups: Iterator[tuple[EffectKind, EffectParameters]] = (
        (kind, getattr(settings, kind)) for kind in KINDS
    )
    active: list[tuple[EffectKind, EffectParameters]] = [
        (kind, parameters)
        for kind, parameters in groups
        if parameters.enabled and parameters.intensity and parameters.quantity
    ]
    cutoff: float = (
        max((radial_extent(kind, parameters, diagonal) for kind, parameters in active), default=0)
        + 1
    )
    maximum: Star | None = max(stars, key=lambda star: star.brightness, default=None)
    threshold: float = (
        min(
            ((1 - parameters.quantity / 100) ** 2 * maximum.brightness for _, parameters in active),
            default=float("inf"),
        )
        if maximum is not None
        else float("inf")
    )
    relevant: tuple[Star, ...] = tuple(
        star
        for star in stars
        if star.visible
        and star.brightness > threshold
        and left - cutoff <= star.x <= right + cutoff
        and top - cutoff <= star.y <= bottom + cutoff
    )
    translated: tuple[Star, ...] = tuple(
        replace(star, x=star.x - left, y=star.y - top) for star in relevant
    )
    if maximum is not None and maximum not in relevant:
        translated += (replace(maximum, x=maximum.x - left, y=maximum.y - top, visible=False),)
    return translated


def _render_region(
    frame: ImageFrame,
    stars: tuple[Star, ...],
    settings: EffectSettings,
    region: Bounds,
    sampling: int,
    cancel: Event | None,
    sky_mask: SkyMask | None = None,
) -> Preview:
    """Render a source-aligned crop while retaining full-image effect scaling."""
    check_cancel(cancel)
    left, top, right, bottom = preview_region_bounds(frame, region, sampling)
    diagonal: int = image_diagonal(frame.width, frame.height)
    translated: tuple[Star, ...] = _region_sources(
        stars, settings, (left, top, right, bottom), diagonal
    )
    view: _RegionView = _RegionView(frame.pixels[top:bottom, left:right])
    preview_frame: ImageFrame = ImageFrame(
        view.pixels[::sampling, ::sampling], frame.name, frame.color_space, frame.note
    )
    effects: FloatArray = render_effects(
        view, translated, settings, cancel, sampling=sampling, diagonal=diagonal
    )
    if sky_mask is not None:
        effects *= sky_mask.region((left, top, right, bottom), sampling)[..., None]
    reduction, enlargement = preview_source_filter(
        preview_frame.pixels,
        stars,
        settings,
        cancel,
        origin=(left, top),
        sampling=sampling,
        sky_mask=sky_mask,
    )
    return Preview(
        preview_frame,
        effects,
        sampling,
        origin=(left, top),
        is_region=True,
        reduction=reduction,
        enlargement=enlargement,
    )


def render_preview(
    frame: ImageFrame,
    stars: tuple[Star, ...],
    settings: EffectSettings,
    cancel: Event | None = None,
    *,
    full_resolution: bool = False,
    interactive: bool = False,
    region: tuple[int, int, int, int] | None = None,
    region_sampling: int = 1,
    sky_mask: SkyMask | None = None,
) -> Preview:
    """Render an overview or source-aligned ROI without changing catalogue membership."""
    if region is not None:
        return _render_region(
            frame,
            filter_stars(stars, sky_mask),
            settings,
            region,
            region_sampling,
            cancel,
            sky_mask,
        )
    # Integer subsampling keeps the coordinate mapping exact and bounds preview memory.
    limit: int = 1600
    scale: int = max(1, (max(frame.width, frame.height) + limit - 1) // limit)
    scale = 1 if full_resolution else min(scale, min(frame.width, frame.height) - 1)
    preview_frame: ImageFrame = (
        frame
        if scale == 1
        else ImageFrame(frame.pixels[::scale, ::scale], frame.name, frame.color_space, frame.note)
    )
    mask_option: _MaskOption = {} if sky_mask is None else {"sky_mask": sky_mask}
    effects: FloatArray = render_effects(
        frame, stars, settings, cancel, sampling=scale, **mask_option
    )
    reduction, enlargement = preview_source_filter(
        preview_frame.pixels, stars, settings, cancel, sampling=scale, sky_mask=sky_mask
    )
    return Preview(preview_frame, effects, scale, reduction=reduction, enlargement=enlargement)


def _apply_export_source_filter(
    output: Pixels,
    effects: Pixels,
    source: Pixels,
    top: int,
    stars: tuple[Star, ...],
    settings: EffectSettings,
    cancel: Event | None,
    sky_mask: SkyMask | None,
) -> None:
    """Apply original-pixel corrections in bounded strips; ``source`` starts at row ``top``."""
    sources: tuple[Star, ...] = soft_filter_sources(filter_stars(stars, sky_mask), settings.soft)
    records: DoubleArray = reduction_records(sources)
    large_records: DoubleArray = enlargement_records(
        sources, settings.soft.hue, settings.soft.saturation
    )
    for start in range(0, source.shape[0], 128):
        check_cancel(cancel)
        reduction: FloatArray = star_reduction(
            source[start : start + 128],
            records,
            settings.soft.soft_filter,
            origin=(0, top + start),
            sky_mask=sky_mask,
            cancel=cancel,
        )
        enlargement: FloatArray = star_enlargement(
            source[start : start + 128],
            large_records,
            settings.soft.soft_filter,
            size=settings.soft.size,
            falloff=settings.soft.falloff,
            origin=(0, top + start),
            sky_mask=sky_mask,
            cancel=cancel,
        )
        output[start : start + 128] += (enlargement - reduction) * (
            1 - effects[start : start + 128]
        )


def _export_bands(
    frame: ImageFrame,
    stars: tuple[Star, ...],
    settings: EffectSettings,
    effects_only: bool,
    cancel: Event | None,
    sky_mask: SkyMask | None,
) -> Iterator[Pixels]:
    """Yield the final linear pixels one full-width band at a time."""
    diagonal: int = image_diagonal(frame.width, frame.height)
    visible: tuple[Star, ...] = filter_stars(stars, sky_mask)
    for top in range(0, frame.height, EXPORT_BAND_ROWS):
        check_cancel(cancel)
        bottom: int = min(top + EXPORT_BAND_ROWS, frame.height)
        source: Pixels = frame.pixels[top:bottom]
        translated: tuple[Star, ...] = _region_sources(
            visible, settings, (0, top, frame.width, bottom), diagonal
        )
        effects: FloatArray = render_effects(
            _RegionView(source), translated, settings, cancel, diagonal=diagonal, export=True
        )
        if sky_mask is not None:
            effects *= sky_mask.region((0, top, frame.width, bottom), 1)[..., None]
        if effects_only:
            yield effects
            continue
        output: FloatArray = source + (1 - source) * effects
        if settings.soft.enabled and settings.soft.soft_filter:
            _apply_export_source_filter(
                output, effects, source, top, stars, settings, cancel, sky_mask
            )
        yield output


@dataclass
class ImageServices:
    """Inject external adapters while keeping analysis and composition in the application."""

    read: ImageReader
    write: ImageWriter
    read_preview: PreviewReader | None = None
    preview: PreviewRenderer | None = None
    sky_mask: SkyMaskProvider | None = None
    read_mask: MaskReader | None = None
    # Detection waits for background compilation so an early analysis does not compile it twice;
    # opening never waits, since it compiles quickly on its own.
    ready: Callable[[Event | None], None] | None = None

    def open_image(
        self, source: Path, cancel: Event | None = None
    ) -> tuple[ImageFrame, SkyMask | None]:
        """Read the image and optional encoded-source mask concurrently under one token."""
        if self.read_mask is None:
            return self.read(source, cancel), None
        token: Event = cancel if cancel is not None else Event()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="image-open") as pool:
            image: Future[ImageFrame] = pool.submit(self.read, source, token)
            mask: Future[SkyMask] = pool.submit(self.read_mask, source, token)
            try:
                return image.result(), mask.result()
            except BaseException:
                token.set()
                raise

    def analyze(
        self,
        frame: ImageFrame,
        source: Path | None,
        settings: DetectionSettings = DetectionSettings(),
        cancel: Event | None = None,
        *,
        sky_mask: SkyMask | None = None,
    ) -> tuple[Star, ...]:
        """Detect and measure sources on the original pixels of the open image."""
        check_cancel(cancel)
        if self.ready is not None:
            self.ready(cancel)
        if sky_mask is None and settings.sky_only and self.sky_mask is not None:
            sky_mask = self.sky_mask(frame, source, cancel)
        if settings.sky_only and sky_mask is None:
            raise ValueError("Sky mask is unavailable. Disable Sky only to analyze without it.")
        stars: tuple[Star, ...] = (
            detect_stars(frame, settings, cancel, sky_mask=sky_mask)
            if sky_mask is not None
            else detect_stars(frame, settings, cancel)
        )
        stars = calibrate_sources(frame, stars, sky_mask=sky_mask, cancel=cancel)
        check_cancel(cancel)
        return stars

    def export(
        self,
        path: Path,
        frame: ImageFrame,
        stars: tuple[Star, ...],
        settings: EffectSettings,
        effects_only: bool,
        source: Path | None,
        cancel: Event | None = None,
        sky_mask: SkyMask | None = None,
    ) -> None:
        """Stream full-resolution row bands to the protected writer; no full-size layer exists."""
        bands: Iterator[Pixels] = _export_bands(
            frame, stars, settings, effects_only, cancel, sky_mask
        )
        self.write(path, frame, bands, source_path=source, cancel=cancel)
