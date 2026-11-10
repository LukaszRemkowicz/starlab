from dataclasses import replace
from math import ceil, floor
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, QTimer
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QMessageBox

from ...application.session import Session
from ...application.use_cases import ImageServices, Preview, render_preview
from ...domain.effects.render import composite
from ...domain.effects.settings import KINDS
from ...domain.stars import DetectionSettings
from .worker import JobRunner


class Controller(QObject):
    def __init__(self, window, services: ImageServices, load_preset, save_preset):
        super().__init__(window)
        self.window, self.services = window, services
        self.load_preset, self.save_preset = load_preset, save_preset
        self.session = Session()
        self.source: Path | None = None
        self.preview: Preview | None = None
        self.overview: Preview | None = None
        self.detail_region = None
        self.progressive_open = False
        self.count_catalog = None
        self.count_mask = None
        self.mask_preview = None
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

    def activity(self, kind: str) -> None:
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
        if self.exclusive:
            return
        self.debounce.stop()
        self.mask_preview = None
        self.window.show_error("")
        path = Path(path)
        settings = self.session.settings

        def read(cancel):
            frame, mask = self.services.open_image(path, cancel)
            preview = render_preview(frame, (), settings, cancel)
            return path, frame, preview, mask

        self.activity("open")
        if self.services.read_preview is None:
            self.runner.submit("open", read)
        else:
            self.runner.submit(
                "open_preview",
                lambda cancel: (path, self.services.read_preview(path, cancel), read),
            )

    def analyze(self, *, force=True, detect=True) -> None:
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
            self.activity("mask")
            self.runner.submit(
                "mask",
                lambda cancel: (
                    revision,
                    self.services.sky_mask(frame, source, cancel),
                    settings,
                    force,
                    detect,
                ),
            )
            return
        self.mask_preview = None
        self._submit_detection(revision, settings, None, force)

    def _submit_detection(self, revision, settings, sky_mask, force):
        frame, source = self.session.frame, self.source
        self.activity("analyze")
        self.runner.submit(
            "analyze",
            lambda cancel: (
                revision,
                self.services.analyze(
                    frame, source, settings, cancel, force=force, sky_mask=sky_mask
                ),
                sky_mask,
                settings,
            ),
        )

    def completed(self, kind: str, result) -> None:
        self.exclusive = False
        if kind == "open_preview":
            path, (thumbnail, scale, width, height), read = result
            scale = max(1, round(1 / scale)) if 0 < scale <= 1 else int(scale)
            settings = self.session.settings
            self.session = Session()
            self.session.set_settings(settings, remember=False)
            self.source = path
            self.preview = self.overview = None
            self.detail_region = None
            self.mask_preview = None
            self.progressive_open = True
            self.window.image_view.fitted = True
            self.window.image_view.has_image = False
            self.window.image_view.stars = ()
            self.window.image_view.set_pixels(
                thumbnail.pixels, thumbnail.color_space, scale, width, height
            )
            self.window.filename.setText(path.name)
            self.window.stack.setCurrentIndex(1)
            self.window.hud.show()
            self.window.image_info.show()
            self.window.image_info.setText(
                f"{width} × {height} · preparing full-resolution original"
            )
            self.runner.submit("open", read)
        elif kind == "open":
            self.source, frame, self.preview, mask = result
            self.overview = self.preview
            self.detail_region = None
            self.session.open(frame)
            if mask is not None:
                self.session.set_catalog((), sky_mask=mask, remember=False)
            self.full_resolution = False
            if not self.progressive_open:
                self.window.image_view.fitted = True
                self.window.image_view.has_image = False
            self.progressive_open = False
            self.window.image_view.stars = ()
            self.window.filename.setText(frame.name)
            self.window.filename.setToolTip(frame.name)
            color_label = frame.color_space
            if "assumed" in frame.note:
                color_label += " (assumed)"
            elif "near-sRGB" in frame.note:
                color_label = "sRGB approximation"
            self.window.image_info.setText(f"{frame.width} × {frame.height} · {color_label}")
            self.window.image_info.setToolTip(frame.note)
            self.window.set_loaded(True)
            self.window.status_text.setText(frame.note)
            self.display()
            if not self.window.image_view.fitted:
                self.request_detail()
            if self.window.auto_detect.isChecked() or self.services.sky_mask is not None:
                self.analyze(force=False, detect=self.window.auto_detect.isChecked())
            else:
                self.refresh()
        elif kind == "mask":
            revision, mask, settings, force, detect = result
            if revision == self.session.revision:
                self.mask_preview = mask
                if self.session.sky_mask is None and not self.session.stars:
                    # The initial mask is independent of detection and survives its cancellation.
                    self.session.set_catalog(
                        (), sky_mask=mask, detection_settings=settings, remember=False
                    )
                    revision = self.session.revision
                if self.window.mode == "mask":
                    self.display()
                if detect:
                    self._submit_detection(revision, settings, mask, force)
                else:
                    self.mask_preview = None
                    self.refresh()
        elif kind == "analyze":
            revision, stars, mask, settings = result
            if revision == self.session.revision:
                self.session.set_catalog(
                    stars, reset_edits=True, sky_mask=mask, detection_settings=settings
                )
                self.mask_preview = None
                self._clear_effect_history()
                self.refresh()
        elif kind == "preview":
            revision, preview = result
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
                # Show completed intermediate frames while newer slider values wait.
                self.preview = preview
                self.display()
                self.debounce.start(0)
        elif kind == "export":
            self.saved_message = f"Saved: {result}"
            self.window.status_text.setText(self.saved_message)

    def request_detail(self) -> None:
        if self.session.frame is None:
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

    def _show_original_detail(self):
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
        region_options = {}
        if self.session.sky_mask is not None:
            region_options["sky_mask"] = self.session.sky_mask
        if self.services.preview is not None:
            viewport = self.window.image_view.viewport()
            region_options["max_edge"] = min(
                1600,
                max(640, viewport.width() if frame.width >= frame.height else viewport.height()),
            )
        if self.detail_region is not None and not self.window.image_view.fitted:
            region_options.update(
                region=self.detail_region[0], region_sampling=self.detail_region[1]
            )
        self.runner.submit(
            "preview",
            lambda cancel: (
                revision,
                (self.services.preview or render_preview)(
                    frame,
                    stars,
                    settings,
                    cancel,
                    full_resolution=full_resolution,
                    interactive=interactive,
                    **region_options,
                ),
            ),
        )

    def display(self, *_args) -> None:
        if self.preview is None or self.session.frame is None:
            return
        if self.window.mode in ("original", "mask") and self.detail_region is not None:
            self._show_original_detail()
            return
        region_ready = self.detail_region is not None and (
            self.preview.is_region
            and self.preview.origin == self.detail_region[0][:2]
            and self.preview.scale == self.detail_region[1]
        )
        if _args and self.detail_region is not None and not region_ready:
            self._show_original_detail()
            if not self.exclusive:
                self.refresh()
            return
        preview = (
            self.overview
            if self.window.image_view.fitted and self.preview.is_region
            else self.preview
        )
        original = preview.frame.pixels
        result = composite(
            preview.frame,
            preview.effects,
            reduction=preview.reduction,
            enlargement=preview.enlargement,
        )
        mode = self.window.mode
        if mode == "mask":
            left, top = preview.origin
            pixels = self._mask_pixels(
                original,
                (
                    left,
                    top,
                    left + original.shape[1] * preview.scale,
                    top + original.shape[0] * preview.scale,
                ),
                preview.scale,
            )
        elif mode == "original":
            pixels = original
        elif mode == "effects":
            pixels = preview.effects
        elif mode == "compare":
            pixels = result.copy()
            middle = pixels.shape[1] // 2
            pixels[:, :middle] = original[:, :middle]
            pixels[:, middle : middle + 1] = 0.5
        else:
            pixels = result
        frame = self.session.frame
        if preview.is_region:
            self.window.image_view.set_region_pixels(
                pixels, frame.color_space, preview.scale, preview.origin
            )
        else:
            self.window.image_view.set_pixels(
                pixels, frame.color_space, preview.scale, frame.width, frame.height
            )

    def _mask_pixels(self, pixels, bounds, sampling):
        mask = self.mask_preview if self.mask_preview is not None else self.session.sky_mask
        if mask is None:
            return pixels
        palette = np.array([[0.5, 0.03, 0.01], [0.01, 0.13, 0.5], [0.6, 0.4, 0.01]], np.float32)
        colors = palette[mask.labels_region(bounds, sampling)]
        return pixels * 0.65 + colors * 0.35

    def update_star_count(self, stars, settings):
        mask = self.session.sky_mask
        if self.count_catalog is not stars or self.count_mask is not mask:
            self.count_catalog = stars
            self.count_mask = mask
            self.count_brightness = np.array([s.brightness for s in stars])
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
        active = np.zeros(len(stars), dtype=bool)
        counts = []
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

    def change_settings(self, settings) -> None:
        if self.exclusive:
            return
        self.session.set_settings(settings, remember=not self.dragging)
        self._remove_disabled_effects()
        self.window.mark_custom()
        if not self.debounce.isActive():
            self.debounce.start(0 if self.dragging else 16)

    def _remove_disabled_effects(self):
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
            # The overview may be exposed by panning before a new ROI arrives.
            overview = self.overview
            if overview is not None and self.preview.is_region:
                frame = self.session.frame
                self.window.image_view.set_pixels(
                    composite(
                        overview.frame,
                        overview.effects,
                        reduction=overview.reduction,
                        enlargement=overview.enlargement,
                    ),
                    frame.color_space,
                    overview.scale,
                    frame.width,
                    frame.height,
                    preserve_detail=True,
                )
            self.display()

    def _enabled_preview(self, preview):
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
        if not self.exclusive:
            self.dragging = True
            self.session.begin_edit()
            if self.runner.active is not None and self.runner.active.kind == "preview":
                self.runner.cancel()
            self.debounce.start(0)

    def end_edit(self) -> None:
        self.dragging = False
        self.debounce.start(0)

    def undo(self) -> None:
        previous_stars, previous_mask = self.session.stars, self.session.sky_mask
        if not self.exclusive and self.session.undo():
            if (
                self.session.stars is not previous_stars
                or self.session.sky_mask is not previous_mask
            ):
                self._clear_effect_history()
            self._restore_detection_controls()
            self.window.mark_custom()
            self.refresh()

    def redo(self) -> None:
        previous_stars, previous_mask = self.session.stars, self.session.sky_mask
        if not self.exclusive and self.session.redo():
            if (
                self.session.stars is not previous_stars
                or self.session.sky_mask is not previous_mask
            ):
                self._clear_effect_history()
            self._restore_detection_controls()
            self.window.mark_custom()
            self.refresh()

    def _centers(self, checked: bool) -> None:
        self.window.image_view.show_centers = checked
        self.window.image_view.viewport().update()

    def click_star(self, x: float, y: float) -> None:
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

    def edit_dialog(self, star) -> None:
        dialog = QDialog(self.window)
        dialog.setWindowTitle(f"Edit star {star.id}")
        form = QFormLayout(dialog)
        fields = {}
        frame = self.session.frame
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
        if dialog.exec():
            self.session.edit_star(
                star.id,
                x=fields["x"].value(),
                y=fields["y"].value(),
                brightness=fields["brightness"].value(),
                color=tuple(fields[c].value() for c in ("red", "green", "blue")),
            )

    def export_path(self, path: str | Path, effects_only: bool) -> None:
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

        def export(cancel):
            self.services.export(
                path, frame, stars, settings, effects_only, source, cancel, sky_mask=sky_mask
            )
            return path

        self.activity("export")
        self.runner.submit("export", export)

    def read_preset(self, path: str) -> None:
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
        try:
            self.save_preset(path, self.session.settings)
            self.window.show_error("")
        except (ValueError, OSError) as exc:
            self.window.show_error(str(exc))

    def cancel(self) -> None:
        self.debounce.stop()
        self.runner.cancel()
        self._restore_detection_controls()
        self.window.set_activity("cancel")

    def _failed(self, message):
        self._restore_detection_controls()
        self.window.show_error(message)

    def _restore_detection_controls(self):
        self.mask_preview = None
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

    def _clear_effect_history(self):
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
        self.window.image_view.set_pixels(
            self.overview.frame.pixels,
            frame.color_space,
            self.overview.scale,
            frame.width,
            frame.height,
        )

    def shutdown(self) -> bool:
        self.debounce.stop()
        return self.runner.shutdown()
