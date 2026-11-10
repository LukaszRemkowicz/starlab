"""GUI-thread session orchestration with immutable worker snapshots and revision checks."""

from collections.abc import Callable
from dataclasses import replace
from math import ceil, floor
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, QTimer
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QMessageBox

from ...application.session import Session
from ...application.use_cases import ImageServices, Preview, render_preview
from ...domain.array_types import BoolArray, Bounds, DoubleArray
from ...domain.effects.render import composite
from ...domain.effects.settings import KINDS, EffectSettings
from ...domain.imaging import ImageFrame, Pixels
from ...domain.sky_mask import SkyMask
from ...domain.stars import DetectionSettings, Star
from .image_view import ViewState
from .job_results import AnalysisResult, MaskResult, OpenPreviewResult, OpenResult, PreviewResult

if TYPE_CHECKING:
    from .main_window import MainWindow
from .worker import JobRunner


class _RegionOptions(TypedDict, total=False):
    """Optional ROI arguments shared by the CPU and GPU preview call contracts."""

    sky_mask: SkyMask
    region: Bounds
    region_sampling: int


class Controller(QObject):
    """Keep session mutation on the GUI thread and reject obsolete asynchronous results."""

    count_brightness: DoubleArray
    count_eligible: BoolArray
    count_visible: BoolArray

    def __init__(
        self,
        window: "MainWindow",
        services: ImageServices,
        load_preset: Callable[[str], EffectSettings],
        save_preset: Callable[[str, EffectSettings], None],
    ) -> None:
        """Wire the window to a GUI-owned session and a persistent single-thread runner."""
        super().__init__(window)
        self.window, self.services = window, services
        self.load_preset, self.save_preset = load_preset, save_preset
        self.session = Session()
        self.source: Path | None = None
        self.preview: Preview | None = None
        self.overview: Preview | None = None
        self.detail_region: tuple[Bounds, int] | None = None
        self.progressive_open = False
        self._open_view_state: ViewState | None = None
        self._open_labels: tuple[tuple[str, str], ...] = ()
        self.count_catalog: tuple[Star, ...] | None = None
        self.count_mask: SkyMask | None = None
        self.mask_preview: SkyMask | None = None
        self.runner = JobRunner(self)
        self.dragging = False
        self.exclusive = False
        self.full_resolution = False
        self.saved_message = ""
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(16)
        self.debounce.timeout.connect(self.refresh)
        window.open_requested.connect(self.open_path)
        window.image_view.file_dropped.connect(self.open_path)
        window.image_view.detail_requested.connect(self.request_detail)
        window.export_requested.connect(self.export_path)
        window.apply_requested.connect(self.refresh)
        window.analyze_button.clicked.connect(lambda: self.analyze(force=True))
        window.detection_method.currentIndexChanged.connect(lambda _: self.analyze(force=False))
        window.cancel_button.clicked.connect(self.cancel)
        window.effect_panel.settings_changed.connect(self.change_settings)
        window.effect_panel.edit_started.connect(self.begin_edit)
        window.effect_panel.edit_finished.connect(self.end_edit)
        window.undo_requested.connect(self.undo)
        window.redo_requested.connect(self.redo)
        window.view_changed.connect(self.display)
        window.tool_changed.connect(window.image_view.set_tool)
        window.centers.toggled.connect(self._centers)
        window.image_view.image_clicked.connect(self.click_star)
        window.preset_load_requested.connect(self.read_preset)
        window.preset_save_requested.connect(self.write_preset)
        self.runner.completed.connect(self.completed)
        self.runner.failed.connect(self._failed)
        self.runner.activity.connect(self.activity)
        self.runner.idle.connect(self.idle)
        self.runner.shutdown_finished.connect(window.close)

    def activity(self, kind: str) -> None:
        """Reflect worker activity and lock editing during document-changing operations."""
        self.saved_message = ""
        self.exclusive = kind != "preview"
        w = self.window
        w.progress.show()
        w.cancel_button.show()
        w.set_activity(
            "open_masked" if kind == "open" and self.services.read_mask is not None else kind
        )
        w.set_exclusive(self.exclusive)
        w.effect_panel.setEnabled(not self.exclusive)
        w.analyze_button.setEnabled(not self.exclusive and self.session.frame is not None)
        w.export_button.setEnabled(not self.exclusive and self.session.frame is not None)
        w.open_button.setEnabled(not self.exclusive)
        w.undo_button.setEnabled(not self.exclusive)
        w.redo_button.setEnabled(not self.exclusive)
        w.load_preset_button.setEnabled(not self.exclusive)

    def idle(self) -> None:
        """Restore controls after all active and pending work has finished."""
        self.exclusive = False
        w = self.window
        w.finish_activity()
        w.progress.hide()
        w.cancel_button.hide()
        w.open_button.setEnabled(True)
        w.load_preset_button.setEnabled(True)
        w.set_loaded(self.session.frame is not None)
        hint = "Ready · GPU" if self.services.preview is not None else "Ready"
        if self.dragging:
            hint = "Live preview"
        w.status_text.setText(
            (self.saved_message or hint)
            if not w.error_label.isVisible()
            else "Operation failed — see message above"
        )

    def open_path(self, path: str | Path) -> None:
        """Read an image asynchronously while retaining the current document until commit."""
        if self.exclusive:
            return
        self.debounce.stop()
        self.mask_preview = None
        self.window.show_error("")
        path = Path(path)
        settings = self.session.settings

        def read(cancel: Event) -> OpenResult:
            """Decode a full image and construct its initial effect-free overview."""
            frame, mask = self.services.open_image(path, cancel)
            preview = render_preview(frame, (), settings, cancel)
            return OpenResult(path, frame, preview, mask)

        self.activity("open")
        read_preview = self.services.read_preview
        if read_preview is None:
            self.runner.submit("open", read)
        else:

            def read_thumbnail(cancel: Event) -> OpenPreviewResult:
                """Return a temporary thumbnail with the typed full-read continuation."""
                return OpenPreviewResult(path, read_preview(path, cancel), read)

            self.runner.submit("open_preview", read_thumbnail)

    def analyze(self, *, force: bool = True, detect: bool = True) -> None:
        """Confirm replacement of manual edits and prepare masking and star detection."""
        frame = self.session.frame
        if frame is None or self.exclusive:
            return
        if self.session.edited:
            answer = QMessageBox.question(
                self.window,
                "Replace manual edits?",
                "Reanalysis creates a new star catalog and replaces manual edits. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._restore_detection_controls()
                return
        self.debounce.stop()
        self.window.show_error("")
        revision = self.session.revision
        method = self.window.detection_method.currentData()
        threshold = self.window.threshold.value()
        settings = DetectionSettings(
            threshold_sigma=threshold if method == "multiscale" else threshold * 25,
            method=method,
            sky_only=self.services.sky_mask is not None,
        )
        self.window.star_count.setText("Detecting stars…")
        if settings.sky_only:
            if self.session.sky_mask is not None:
                if detect:
                    self._submit_detection(revision, settings, self.session.sky_mask, force)
                else:
                    self.refresh()
                return
            source = self.source
            mask_service = self.services.sky_mask
            assert mask_service is not None
            self.activity("mask")
            self.runner.submit(
                "mask",
                lambda cancel: MaskResult(
                    revision,
                    mask_service(frame, source, cancel),
                    settings,
                    force,
                    detect,
                ),
            )
            return
        self.mask_preview = None
        self._submit_detection(revision, settings, None, force)

    def _submit_detection(
        self, revision: int, settings: DetectionSettings, sky_mask: SkyMask | None, force: bool
    ) -> None:
        """Submit catalog detection from the captured frame, mask and detector settings."""
        frame, source = self.session.frame, self.source
        assert frame is not None
        self.activity("analyze")
        self.runner.submit(
            "analyze",
            lambda cancel: AnalysisResult(
                revision,
                self.services.analyze(
                    frame, source, settings, cancel, force=force, sky_mask=sky_mask
                ),
                sky_mask,
                settings,
            ),
        )

    def completed(self, kind: str, result: object) -> None:
        """Decode tuple-compatible payloads at the Qt object-signal boundary."""
        self.exclusive = False
        match kind:
            case "open_preview":
                self._complete_open_preview(*cast(OpenPreviewResult, result))
            case "open":
                self._complete_open(*cast(OpenResult, result))
            case "mask":
                self._complete_mask(*cast(MaskResult, result))
            case "analyze":
                self._complete_analyze(*cast(AnalysisResult, result))
            case "preview":
                self._complete_preview(*cast(PreviewResult, result))
            case "export":
                self.saved_message = f"Saved: {result}"
                self.window.status_text.setText(self.saved_message)

    def _complete_open_preview(
        self,
        path: Path,
        thumbnail_info: tuple[ImageFrame, float, int, int],
        read: Callable[[Event], OpenResult],
    ) -> None:
        """Show a temporary thumbnail while preserving the committed document and view."""
        thumbnail, scale, width, height = thumbnail_info
        scale = max(1, round(1 / scale)) if 0 < scale <= 1 else int(scale)
        view = self.window.image_view
        self._open_view_state = view.capture_state()
        self._open_labels = tuple(
            (label.text(), label.toolTip())
            for label in (self.window.filename, self.window.image_info)
        )
        self.progressive_open = True
        view.fitted, view.has_image, view.stars = True, False, ()
        view.set_pixels(thumbnail.pixels, thumbnail.color_space, scale, width, height)
        self.window.filename.setText(path.name)
        self.window.stack.setCurrentIndex(1)
        self.window.hud.show()
        self.window.image_info.show()
        self.window.image_info.setText(f"{width} × {height} · preparing full-resolution original")
        self.runner.submit("open", read)

    def _complete_open(
        self,
        path: Path,
        frame: ImageFrame,
        preview: Preview,
        mask: SkyMask | None,
    ) -> None:
        """Commit a fully decoded document, then prepare its detection and effects."""
        self.source, self.preview, self.overview = path, preview, preview
        self.detail_region = None
        self.session.open(frame)
        if mask is not None:
            self.session.set_catalog((), sky_mask=mask, remember=False)
        self.full_resolution = False
        if not self.progressive_open:
            self.window.image_view.fitted = True
            self.window.image_view.has_image = False
        self.progressive_open = False
        self._open_view_state = None
        self._open_labels = ()
        self.window.image_view.stars = ()
        self._show_frame_info(frame)
        self.window.set_loaded(True)
        self.window.status_text.setText(frame.note)
        self.display()
        if not self.window.image_view.fitted:
            self.request_detail()
        if self.window.auto_detect.isChecked() or self.services.sky_mask is not None:
            self.analyze(force=False, detect=self.window.auto_detect.isChecked())
        else:
            self.refresh()

    def _show_frame_info(self, frame: ImageFrame) -> None:
        """Describe the committed image and its color interpretation."""
        self.window.filename.setText(frame.name)
        self.window.filename.setToolTip(frame.name)
        color_label = frame.color_space
        if "assumed" in frame.note:
            color_label += " (assumed)"
        elif "near-sRGB" in frame.note:
            color_label = "sRGB approximation"
        self.window.image_info.setText(f"{frame.width} × {frame.height} · {color_label}")
        self.window.image_info.setToolTip(frame.note)

    def _restore_open_presentation(self) -> None:
        """Remove an interrupted thumbnail without changing document history or ROI."""
        if self._open_view_state is None:
            return
        self.progressive_open = False
        self.window.image_view.restore_state(self._open_view_state)
        for label, (text, tooltip) in zip(
            (self.window.filename, self.window.image_info),
            self._open_labels,
            strict=True,
        ):
            label.setText(text)
            label.setToolTip(tooltip)
        self._open_view_state = None
        self._open_labels = ()
        self.window.set_loaded(self.session.frame is not None)
        self.display()

    def _complete_mask(
        self,
        revision: int,
        mask: SkyMask,
        settings: DetectionSettings,
        force: bool,
        detect: bool,
    ) -> None:
        """Accept a current mask before starting the optional catalog detection."""
        if revision != self.session.revision:
            return
        self.mask_preview = mask
        if self.session.sky_mask is None and not self.session.stars:
            # The initial mask survives cancellation of the subsequent detection.
            self.session.set_catalog((), sky_mask=mask, detection_settings=settings, remember=False)
            revision = self.session.revision
        if self.window.mode == "mask":
            self.display()
        if detect:
            self._submit_detection(revision, settings, mask, force)
        else:
            self.mask_preview = None
            self.refresh()

    def _complete_analyze(
        self,
        revision: int,
        stars: tuple[Star, ...],
        mask: SkyMask | None,
        settings: DetectionSettings,
    ) -> None:
        """Commit a current catalog together with the mask used to detect it."""
        if revision == self.session.revision:
            self.session.set_catalog(
                stars, reset_edits=True, sky_mask=mask, detection_settings=settings
            )
            self.mask_preview = None
            self._clear_effect_history()
            self.refresh()

    def _complete_preview(self, revision: int, preview: Preview) -> None:
        """Accept current preview work, allowing intermediate slider frames while dragging."""
        preview = self._enabled_preview(preview)
        if preview.is_region and (
            self.detail_region is None or preview.origin != self.detail_region[0][:2]
        ):
            return
        if self.session.accept_preview(revision, preview.effects):
            self.preview = preview
            if not preview.is_region:
                self.overview = preview
            self.display()
        elif self.dragging:
            self.preview = preview
            self.display()
            self.debounce.start(0)

    def request_detail(self) -> None:
        """Request a sampled ROI in full-image coordinates for the current viewport."""
        if self.session.frame is None or self.progressive_open:
            return
        view = self.window.image_view
        if view.fitted:
            self.full_resolution = False
            self.detail_region = None
            if not self.exclusive:
                self.refresh()
            return
        frame = self.session.frame
        sampling = max(1, floor(1 / max(view.transform().m11(), 0.001)))
        rect = view.visible_region()
        left = max(0, floor((rect.left() - 32) / sampling) * sampling)
        top = max(0, floor((rect.top() - 32) / sampling) * sampling)
        right = min(frame.width, ceil((rect.right() + 32) / sampling) * sampling)
        bottom = min(frame.height, ceil((rect.bottom() + 32) / sampling) * sampling)
        if right <= left or bottom <= top:
            return
        self.detail_region = ((left, top, right, bottom), sampling)
        if self.window.mode in ("original", "mask"):
            self._show_original_detail()
        if not self.exclusive and self.window.mode not in ("original", "mask"):
            self.refresh()

    def _show_original_detail(self) -> None:
        """Present native source pixels immediately without waiting for effect rendering."""
        if self.detail_region is None or self.session.frame is None:
            return
        (left, top, right, bottom), sampling = self.detail_region
        frame = self.session.frame
        pixels = frame.pixels[top:bottom:sampling, left:right:sampling]
        if self.window.mode == "mask":
            pixels = self._mask_pixels(pixels, (left, top, right, bottom), sampling)
        self.window.image_view.set_region_pixels(
            pixels,
            frame.color_space,
            sampling,
            (left, top),
        )

    def refresh(self) -> None:
        """Submit a preview from the current immutable session snapshot."""
        self.debounce.stop()
        if self.exclusive:
            return
        if self.dragging and self.runner.busy:
            self.debounce.start(16)
            return
        self.window.effect_panel.set_settings(self.session.settings)
        if self.session.frame is None:
            return
        if self.services.sky_mask is not None and self.session.sky_mask is None:
            self.window.show_error("Foreground mask is not ready. Run Detect stars to prepare it.")
            return
        frame, stars, settings, revision = (
            self.session.frame,
            self.session.stars,
            self.session.settings,
            self.session.revision,
        )
        self.update_star_count(stars, settings)
        self.window.image_view.stars = stars
        interactive = self.dragging
        full_resolution = self.full_resolution and not interactive
        region_options: _RegionOptions = {}
        if self.session.sky_mask is not None:
            region_options["sky_mask"] = self.session.sky_mask
        renderer = self.services.preview
        max_edge = 1600
        if renderer is not None:
            viewport = self.window.image_view.viewport()
            max_edge = min(
                1600,
                max(640, viewport.width() if frame.width >= frame.height else viewport.height()),
            )
        if self.detail_region is not None and not self.window.image_view.fitted:
            region_options["region"] = self.detail_region[0]
            region_options["region_sampling"] = self.detail_region[1]

        def render(cancel: Event) -> PreviewResult:
            """Render a captured snapshot with arguments supported by its selected backend."""
            if renderer is None:
                preview = render_preview(
                    frame,
                    stars,
                    settings,
                    cancel,
                    full_resolution=full_resolution,
                    interactive=interactive,
                    **region_options,
                )
            else:
                preview = renderer(
                    frame,
                    stars,
                    settings,
                    cancel,
                    full_resolution=full_resolution,
                    interactive=interactive,
                    max_edge=max_edge,
                    **region_options,
                )
            return PreviewResult(revision, preview)

        self.runner.submit("preview", render)

    def display(self, *_args: object) -> None:
        """Present both image layers in the selected mode without accepting stale regions."""
        if self.preview is None or self.session.frame is None or self.progressive_open:
            return
        view = self.window.image_view
        overview = self.overview or (self.preview if not self.preview.is_region else None)
        if overview is not None:
            self._show_overview(overview)
        if view.fitted:
            return
        if self.window.mode in ("original", "mask") and self.detail_region is not None:
            self._show_original_detail()
            return
        region_ready = self.detail_region is not None and (
            self.preview.is_region
            and self.preview.origin == self.detail_region[0][:2]
            and self.preview.scale == self.detail_region[1]
        )
        if region_ready:
            view.set_region_pixels(
                self._mode_pixels(self.preview),
                self.session.frame.color_space,
                self.preview.scale,
                self.preview.origin,
            )
        elif _args and self.detail_region is not None:
            view.clear_detail()
            if not self.exclusive:
                self.refresh()

    def _show_overview(self, preview: Preview) -> None:
        """Keep the exposed base layer in the same display mode as zoomed detail."""
        frame = self.session.frame
        assert frame is not None
        self.window.image_view.set_pixels(
            self._mode_pixels(preview),
            frame.color_space,
            preview.scale,
            frame.width,
            frame.height,
            preserve_detail=not self.window.image_view.fitted,
        )

    def _mode_pixels(self, preview: Preview) -> Pixels:
        """Convert either image layer with a comparison divider fixed in image coordinates."""
        original = preview.frame.pixels
        match self.window.mode:
            case "mask":
                left, top = preview.origin
                return self._mask_pixels(
                    original,
                    (
                        left,
                        top,
                        left + original.shape[1] * preview.scale,
                        top + original.shape[0] * preview.scale,
                    ),
                    preview.scale,
                )
            case "original":
                return original
            case "effects":
                return preview.effects
        result = composite(
            preview.frame,
            preview.effects,
            reduction=preview.reduction,
            enlargement=preview.enlargement,
        )
        if self.window.mode == "compare":
            result = result.copy()
            frame = self.session.frame
            assert frame is not None
            middle = frame.width // 2
            columns = preview.origin[0] + np.arange(result.shape[1]) * preview.scale
            result[:, columns < middle] = original[:, columns < middle]
            divider = (middle - preview.origin[0]) // preview.scale
            if 0 <= divider < result.shape[1]:
                result[:, divider : divider + 1] = 0.5
        return result

    def _mask_pixels(self, pixels: Pixels, bounds: Bounds, sampling: int) -> Pixels:
        """Blend sampled mask classes into source pixels using full-image bounds."""
        mask = self.mask_preview if self.mask_preview is not None else self.session.sky_mask
        if mask is None:
            return pixels
        palette = np.array([[0.5, 0.03, 0.01], [0.01, 0.13, 0.5], [0.6, 0.4, 0.01]], np.float32)
        colors = palette[mask.labels_region(bounds, sampling)]
        return pixels * 0.65 + colors * 0.35

    def update_star_count(self, stars: tuple[Star, ...], settings: EffectSettings) -> None:
        """Report enabled effect source counts using the committed catalog and mask."""
        mask = self.session.sky_mask
        if self.count_catalog is not stars or self.count_mask is not mask:
            self.count_catalog = stars
            self.count_mask = mask
            self.count_brightness = np.array([s.brightness for s in stars], dtype=np.float64)
            self.count_eligible = (
                np.ones(len(stars), dtype=bool)
                if mask is None
                else mask.keep(np.array([(s.x, s.y) for s in stars]))
            )
            self.count_visible = (
                np.array([s.visible for s in stars], dtype=bool) & self.count_eligible
            )
        values = self.count_brightness
        maximum = float(values[self.count_eligible].max(initial=0))
        active: BoolArray = np.zeros(len(stars), dtype=bool)
        counts: list[str] = []
        for kind in KINDS:
            params = getattr(settings, kind)
            selected = self.count_visible & (values > (1 - params.quantity / 100) ** 2 * maximum)
            if not params.enabled or not params.intensity or not params.quantity:
                selected[:] = False
            active |= selected
            count = int(np.count_nonzero(selected))
            counts.append(f"{kind}: {count}")
            self.window.effect_panel.source_counts[kind].setText(f"{count} / {len(stars)}")
        self.window.star_count.setText(
            f"{len(stars)} detected · {np.count_nonzero(active)} with effects"
        )
        self.window.star_count.setToolTip(
            "\n".join(counts) + "\nQuantity selects how many detected sources receive each effect."
        )

    def change_settings(self, settings: EffectSettings) -> None:
        """Record settings edits and debounce rendering while retaining slider history."""
        if self.exclusive:
            return
        self.session.set_settings(settings, remember=not self.dragging)
        self._remove_disabled_effects()
        self.window.mark_custom()
        if not self.debounce.isActive():
            self.debounce.start(0 if self.dragging else 16)

    def _remove_disabled_effects(self) -> None:
        """Invalidate both zoom and overview layers before an old effect can reappear."""
        changed = False
        for name in ("preview", "overview"):
            preview = getattr(self, name)
            if preview is None:
                continue
            filtered = self._enabled_preview(preview)
            if filtered is preview:
                continue
            setattr(self, name, filtered)
            changed = True
        if changed:
            self.display()

    def _enabled_preview(self, preview: Preview) -> Preview:
        """Remove newly disabled groups and source corrections from a retained preview."""
        if (preview.reduction is not None or preview.enlargement is not None) and (
            not self.session.settings.soft.enabled or not self.session.settings.soft.soft_filter
        ):
            preview = replace(preview, reduction=None, enlargement=None)
        if preview.groups is None:
            return preview
        groups = {
            kind: pixels
            for kind, pixels in preview.groups.items()
            if (p := getattr(self.session.settings, kind)).enabled and p.intensity and p.quantity
        }
        if groups.keys() == preview.groups.keys():
            return preview
        effects = np.zeros_like(preview.effects)
        for group in groups.values():
            effects += (1 - effects) * group
        return replace(preview, effects=effects, groups=groups)

    def begin_edit(self) -> None:
        """Group slider changes into one undo state and allow interactive preview frames."""
        if not self.exclusive:
            self.dragging = True
            self.session.begin_edit()
            if self.runner.active is not None and self.runner.active.kind == "preview":
                self.runner.cancel()
            self.debounce.start(0)

    def end_edit(self) -> None:
        """Finish a slider edit and request the final noninteractive preview."""
        self.dragging = False
        self.debounce.start(0)

    def undo(self) -> None:
        """Restore the preceding edit state and invalidate catalog-dependent layers."""
        self._restore_history(self.session.undo)

    def redo(self) -> None:
        """Restore the next edit state and invalidate catalog-dependent layers."""
        self._restore_history(self.session.redo)

    def _restore_history(self, restore: Callable[[], bool]) -> None:
        """Restore an edit and invalidate cached effects when its catalog or mask changes."""
        previous_stars, previous_mask = self.session.stars, self.session.sky_mask
        if not self.exclusive and restore():
            if (
                self.session.stars is not previous_stars
                or self.session.sky_mask is not previous_mask
            ):
                self._clear_effect_history()
            self._restore_detection_controls()
            self.window.mark_custom()
            self.refresh()

    def _centers(self, checked: bool) -> None:
        """Toggle foreground catalog markers without recalculating any effect pixels."""
        self.window.image_view.show_centers = checked
        self.window.image_view.viewport().update()

    def click_star(self, x: float, y: float) -> None:
        """Apply the selected catalog tool at full-image pixel coordinates."""
        if self.exclusive or self.session.frame is None:
            return
        tool = self.window.image_view.tool
        if tool == "add":
            self.session.add_star(x, y)
        else:
            closest = min(
                self.session.stars, key=lambda s: (s.x - x) ** 2 + (s.y - y) ** 2, default=None
            )
            tolerance = 18 / max(self.window.image_view.transform().m11(), 0.01)
            if closest is None or np.hypot(closest.x - x, closest.y - y) > tolerance:
                return
            if tool == "hide":
                self.session.edit_star(closest.id, visible=not closest.visible)
            elif tool == "edit":
                self.edit_dialog(closest)
            else:
                return
        self.refresh()

    def edit_dialog(self, star: Star) -> None:
        """Edit one source through a modal dialog and release its widgets afterwards."""
        dialog = QDialog(self.window)
        dialog.setWindowTitle(f"Edit star {star.id}")
        form = QFormLayout(dialog)
        fields: dict[str, QDoubleSpinBox] = {}
        frame = self.session.frame
        assert frame is not None
        for key, maximum, value in (
            ("x", frame.width - 0.001, star.x),
            ("y", frame.height - 0.001, star.y),
            ("brightness", 10, star.brightness),
            ("red", 1, star.color[0]),
            ("green", 1, star.color[1]),
            ("blue", 1, star.color[2]),
        ):
            field = QDoubleSpinBox()
            field.setDecimals(3)
            field.setRange(0, maximum)
            field.setValue(value)
            form.addRow(key.capitalize(), field)
            fields[key] = field
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        try:
            if dialog.exec():
                self.session.edit_star(
                    star.id,
                    x=fields["x"].value(),
                    y=fields["y"].value(),
                    brightness=fields["brightness"].value(),
                    color=(fields["red"].value(), fields["green"].value(), fields["blue"].value()),
                )
        finally:
            dialog.deleteLater()

    def export_path(self, path: str | Path, effects_only: bool) -> None:
        """Export a frozen document snapshot as result pixels or effects only."""
        if self.session.frame is None or self.exclusive:
            return
        if self.services.sky_mask is not None and self.session.sky_mask is None:
            self.window.show_error(
                "Foreground mask is not ready. Run Detect stars before exporting."
            )
            return
        self.debounce.stop()
        path = Path(path)
        frame, stars, settings, source, sky_mask = (
            self.session.frame,
            self.session.stars,
            self.session.settings,
            self.source,
            self.session.sky_mask,
        )
        self.window.show_error("")

        def export(cancel: Event) -> Path:
            """Export the captured session snapshot without accessing mutable GUI state."""
            self.services.export(
                path, frame, stars, settings, effects_only, source, cancel, sky_mask=sky_mask
            )
            return path

        self.activity("export")
        self.runner.submit("export", export)

    def read_preset(self, path: str) -> None:
        """Load and apply validated effect settings from a preset file."""
        if self.exclusive:
            return
        try:
            self.session.set_settings(self.load_preset(path))
            self.window.effect_panel.set_settings(self.session.settings)
            self.window.mark_custom()
            self.refresh()
            self.window.show_error("")
        except (ValueError, OSError) as exc:
            self.window.show_error(str(exc))

    def write_preset(self, path: str) -> None:
        """Persist the current effect settings and report expected file errors."""
        try:
            self.save_preset(path, self.session.settings)
            self.window.show_error("")
        except (ValueError, OSError) as exc:
            self.window.show_error(str(exc))

    def cancel(self) -> None:
        """Invalidate pending results and restore committed presentation and controls."""
        self.debounce.stop()
        self.runner.cancel()
        self._restore_open_presentation()
        self._restore_detection_controls()
        self.window.set_activity("cancel")

    def _failed(self, message: str) -> None:
        """Restore committed state before showing an asynchronous operation error."""
        self._restore_open_presentation()
        self._restore_detection_controls()
        self.window.show_error(message)

    def _restore_detection_controls(self) -> None:
        """Restore detector settings and counts from the committed catalog after interruption."""
        self.mask_preview = None
        if self.session.frame is None:
            self.window.star_count.setText("No image loaded")
        else:
            self.update_star_count(self.session.stars, self.session.settings)
        w = self.window
        settings = self.session.detection_settings or DetectionSettings(
            sky_only=self.session.sky_mask is not None
        )
        with QSignalBlocker(w.detection_method):
            index = w.detection_method.findData(settings.method)
            w.detection_method.setCurrentIndex(index)
            w._detector_threshold_mode(index)
            w.threshold.setValue(
                settings.threshold_sigma
                if settings.method == "multiscale"
                else settings.threshold_sigma / 25
            )
        if w.mode == "mask":
            self.display()

    def _clear_effect_history(self) -> None:
        """Remove effects tied to an obsolete catalog while retaining its source overview."""
        if self.overview is None:
            return
        self.overview = replace(
            self.overview,
            effects=np.zeros_like(self.overview.effects),
            groups={},
            reduction=None,
            enlargement=None,
        )
        self.preview = self.overview
        frame = self.session.frame
        assert frame is not None
        self.window.image_view.set_pixels(
            self.overview.frame.pixels,
            frame.color_space,
            self.overview.scale,
            frame.width,
            frame.height,
        )

    def shutdown(self, timeout_ms: int = 5000) -> bool:
        """Stop preview scheduling and drain worker-owned resources on their owning thread."""
        self.debounce.stop()
        return self.runner.shutdown(timeout_ms)
