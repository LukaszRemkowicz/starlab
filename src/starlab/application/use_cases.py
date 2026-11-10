from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event

from ..domain.effects.profiles import image_diagonal, radial_extent
from ..domain.effects.render import composite, render_effects
from ..domain.effects.settings import EffectSettings
from ..domain.effects.soft_filter import (
    enlargement_records,
    reduction_records,
    star_enlargement,
    star_reduction,
)
from ..domain.imaging import ImageFrame, Pixels, check_cancel
from ..domain.photometry import calibrate_sources
from ..domain.sky_mask import SkyMask
from ..domain.stars import DetectionSettings, Star, detect_stars, filter_stars
from .ports import ImageReader, ImageWriter


@dataclass(frozen=True)
class Preview:
    frame: ImageFrame
    effects: Pixels
    scale: int
    draft_count: tuple[int, int] | None = None
    origin: tuple[int, int] = (0, 0)
    is_region: bool = False
    groups: dict[str, Pixels] | None = None
    reduction: Pixels | None = None
    enlargement: Pixels | None = None


def preview_source_filter(
    pixels, stars, settings, cancel=None, *, origin=(0, 0), sampling=1, sky_mask=None
):
    if not settings.soft.enabled or not settings.soft.soft_filter:
        return None, None
    sources = filter_stars(stars, sky_mask)
    options = dict(origin=origin, sampling=sampling, sky_mask=sky_mask, cancel=cancel)
    reduction = star_reduction(
        pixels,
        reduction_records(sources),
        settings.soft.soft_filter,
        **options,
    )
    enlargement = star_enlargement(
        pixels, enlargement_records(sources), settings.soft.soft_filter, **options
    )
    return reduction, enlargement


@dataclass(frozen=True)
class _RegionView:
    """Read-only source view; sampling never copies the entire zoomed-out ROI."""

    pixels: Pixels

    @property
    def width(self):
        return self.pixels.shape[1]

    @property
    def height(self):
        return self.pixels.shape[0]


def _preview_region_bounds(
    frame: ImageFrame, region: tuple[int, int, int, int], sampling: int
) -> tuple[int, int, int, int]:
    """Clip a requested ROI and require two sampled pixels along each image axis."""
    left, top, right, bottom = region
    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(frame.width, int(right)), min(frame.height, int(bottom))
    if (
        type(sampling) is not int
        or sampling < 1
        or (right - left + sampling - 1) // sampling < 2
        or (bottom - top + sampling - 1) // sampling < 2
    ):
        raise ValueError("Preview region must contain at least two sampled pixels on each axis.")
    return left, top, right, bottom


def _region_sources(
    stars: tuple[Star, ...],
    settings: EffectSettings,
    region: tuple[int, int, int, int],
    diagonal: int,
) -> tuple[Star, ...]:
    """Translate contributing sources while retaining the global brightness maximum."""
    left, top, right, bottom = region
    groups = ((kind, getattr(settings, kind)) for kind in ("primary", "secondary", "soft", "ring"))
    active = [(kind, p) for kind, p in groups if p.enabled and p.intensity and p.quantity]
    cutoff = max((radial_extent(kind, p, diagonal) for kind, p in active), default=0) + 1
    maximum = max(stars, key=lambda star: star.brightness, default=None)
    threshold = (
        min(
            ((1 - p.quantity / 100) ** 2 * maximum.brightness for _, p in active),
            default=float("inf"),
        )
        if maximum is not None
        else float("inf")
    )
    relevant = tuple(
        star
        for star in stars
        if star.visible
        and star.brightness > threshold
        and left - cutoff <= star.x <= right + cutoff
        and top - cutoff <= star.y <= bottom + cutoff
    )
    translated = tuple(replace(star, x=star.x - left, y=star.y - top) for star in relevant)
    if maximum is not None and maximum not in relevant:
        translated += (replace(maximum, x=maximum.x - left, y=maximum.y - top, visible=False),)
    return translated


def _render_region(frame, stars, settings, region, sampling, cancel, sky_mask=None):
    check_cancel(cancel)
    left, top, right, bottom = _preview_region_bounds(frame, region, sampling)
    diagonal = image_diagonal(frame.width, frame.height)
    translated = _region_sources(stars, settings, (left, top, right, bottom), diagonal)
    view = _RegionView(frame.pixels[top:bottom, left:right])
    preview_frame = ImageFrame(
        view.pixels[::sampling, ::sampling], frame.name, frame.color_space, frame.note
    )
    effects = render_effects(
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
    sky_mask=None,
) -> Preview:
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
    limit = 1600
    scale = max(1, (max(frame.width, frame.height) + limit - 1) // limit)
    scale = 1 if full_resolution else min(scale, min(frame.width, frame.height) - 1)
    preview_frame = (
        frame
        if scale == 1
        else ImageFrame(frame.pixels[::scale, ::scale], frame.name, frame.color_space, frame.note)
    )
    mask_option = {} if sky_mask is None else {"sky_mask": sky_mask}
    effects = render_effects(frame, stars, settings, cancel, sampling=scale, **mask_option)
    reduction, enlargement = preview_source_filter(
        preview_frame.pixels, stars, settings, cancel, sampling=scale, sky_mask=sky_mask
    )
    return Preview(preview_frame, effects, scale, reduction=reduction, enlargement=enlargement)


def _apply_export_source_filter(
    output: Pixels,
    effects: Pixels,
    frame: ImageFrame,
    stars: tuple[Star, ...],
    settings: EffectSettings,
    cancel: Event | None,
    sky_mask: SkyMask | None,
) -> None:
    """Apply original-pixel corrections in bounded strips before effect composition."""
    records = reduction_records(filter_stars(stars, sky_mask))
    large_records = enlargement_records(filter_stars(stars, sky_mask))
    for top in range(0, frame.height, 128):
        check_cancel(cancel)
        reduction = star_reduction(
            frame.pixels[top : top + 128],
            records,
            settings.soft.soft_filter,
            origin=(0, top),
            sky_mask=sky_mask,
            cancel=cancel,
        )
        enlargement = star_enlargement(
            frame.pixels[top : top + 128],
            large_records,
            settings.soft.soft_filter,
            origin=(0, top),
            sky_mask=sky_mask,
            cancel=cancel,
        )
        output[top : top + 128] += (enlargement - reduction) * (1 - effects[top : top + 128])


@dataclass
class ImageServices:
    read: ImageReader
    write: ImageWriter
    load_catalog: Callable[[Path, DetectionSettings], tuple[Star, ...] | None] | None = None
    save_catalog: Callable[[Path, DetectionSettings, tuple[Star, ...]], None] | None = None
    read_preview: Callable | None = None
    preview: Callable | None = None
    sky_mask: Callable | None = None
    read_mask: Callable | None = None

    def open_image(self, source, cancel=None):
        if self.read_mask is None:
            return self.read(source, cancel), None
        token = cancel if cancel is not None else Event()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="image-open") as pool:
            image = pool.submit(self.read, source, token)
            mask = pool.submit(self.read_mask, source, token)
            try:
                return image.result(), mask.result()
            except BaseException:
                token.set()
                raise

    def analyze(
        self,
        frame,
        source,
        settings=DetectionSettings(),
        cancel=None,
        *,
        force=False,
        sky_mask=None,
    ):
        check_cancel(cancel)
        if sky_mask is None and settings.sky_only and self.sky_mask is not None:
            sky_mask = self.sky_mask(frame, source, cancel)
        if settings.sky_only and sky_mask is None:
            raise ValueError("Sky mask is unavailable. Disable Sky only to analyze without it.")
        if source is not None and self.load_catalog is not None and not force:
            cached = self.load_catalog(source, settings)
            if cached is not None:
                check_cancel(cancel)
                return cached
        stars = (
            detect_stars(frame, settings, cancel, sky_mask=sky_mask)
            if sky_mask is not None
            else detect_stars(frame, settings, cancel)
        )
        stars = calibrate_sources(frame, stars, sky_mask=sky_mask, cancel=cancel)
        check_cancel(cancel)
        if source is not None and self.save_catalog is not None:
            self.save_catalog(source, settings, stars)
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
        sky_mask=None,
    ) -> None:
        effects = render_effects(frame, stars, settings, cancel, sky_mask=sky_mask)
        output = effects if effects_only else composite(frame, effects)
        if not effects_only and settings.soft.enabled and settings.soft.soft_filter:
            _apply_export_source_filter(output, effects, frame, stars, settings, cancel, sky_mask)
        check_cancel(cancel)
        self.write(path, frame, output, source_path=source, cancel=cancel)
