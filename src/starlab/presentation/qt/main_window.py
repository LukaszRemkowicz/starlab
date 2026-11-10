from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import (
    QElapsedTimer,
    QSettings,
    QSignalBlocker,
    QStandardPaths,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...domain.effects.settings import EffectSettings
from .effect_panel import EffectPanel
from .image_view import ImageView
from .theme import STYLE, icon


class Canvas(QWidget):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("canvas")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.file_dropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                break


class MainWindow(QMainWindow):
    open_requested = Signal(str)
    export_requested = Signal(str, bool)
    preset_load_requested = Signal(str)
    preset_save_requested = Signal(str)
    view_changed = Signal(str)
    tool_changed = Signal(str)
    undo_requested = Signal()
    redo_requested = Signal()
    apply_requested = Signal()

    def __init__(self, preferences: QSettings | None = None):
        super().__init__()
        self.preferences = (
            preferences
            if preferences is not None
            else QSettings("Star Effects Lab", "Star Effects Lab")
        )
        self.setWindowTitle("Star Effects Lab")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 720)
        self.setStyleSheet(STYLE)
        self.controller = None
        self.mode = "result"
        self._mask_return_mode = "result"
        self._before_mode = None
        self._loaded = False
        self._activity = ""
        self.clock = QElapsedTimer()
        self.activity_timer = QTimer(self)
        self.activity_timer.setInterval(1000)
        self.activity_timer.timeout.connect(self._activity_tick)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setObjectName("error")
        self.error_label.hide()
        root.addWidget(self.error_label)
        splitter = QSplitter()
        splitter.setHandleWidth(1)
        root.addWidget(splitter, 1)
        self.canvas = Canvas()
        self.canvas.file_dropped.connect(self.open_requested)
        canvas_layout = QGridLayout(self.canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        self.stack.setObjectName("canvas")
        self.stack.addWidget(self._empty_state())
        self.image_view = ImageView()
        self.stack.addWidget(self.image_view)
        canvas_layout.addWidget(self.stack, 0, 0)
        self.hud = self._toolbar()
        hud_margin = QWidget()
        hud_margin.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        hud_box = QHBoxLayout(hud_margin)
        hud_box.setContentsMargins(12, 12, 0, 0)
        hud_box.addWidget(self.hud)
        canvas_layout.addWidget(
            hud_margin, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.image_info = QLabel()
        self.image_info.setObjectName("imageInfo")
        self.image_info.setMaximumWidth(600)
        self.image_info.setWordWrap(False)
        self.image_info.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        canvas_layout.addWidget(
            self.image_info, 0, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft
        )
        splitter.addWidget(self.canvas)
        self.inspector = QWidget()
        self.inspector.setObjectName("inspector")
        self.inspector.setMinimumWidth(320)
        self.inspector.setMaximumWidth(420)
        sidebar = QVBoxLayout(self.inspector)
        sidebar.setContentsMargins(0, 0, 0, 0)
        sidebar.setSpacing(0)
        header = QFrame()
        header.setObjectName("panelHeader")
        header_box = QHBoxLayout(header)
        header_box.setContentsMargins(12, 12, 12, 12)
        title = QLabel("Parameters")
        title.setStyleSheet("font-weight: 600;")
        header_box.addWidget(title)
        header_box.addStretch()
        self.filename = QLabel("No image")
        self.filename.setObjectName("muted")
        self.filename.setMaximumWidth(155)
        header_box.addWidget(self.filename)
        sidebar.addWidget(header)
        self.effect_panel = EffectPanel()
        sidebar.addWidget(self.effect_panel, 1)
        self._general_controls()
        splitter.addWidget(self.inspector)
        splitter.setSizes([1110, 330])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        root.addWidget(self._footer())
        self._shortcuts()
        self.set_loaded(False)

    def _empty_state(self):
        empty = QWidget()
        empty.setObjectName("canvas")
        box = QVBoxLayout(empty)
        box.setSpacing(16)
        box.addStretch()
        emblem = QLabel()
        emblem.setObjectName("emptyIcon")
        emblem.setFixedSize(68, 68)
        emblem.setAlignment(Qt.AlignmentFlag.AlignCenter)
        emblem.setPixmap(icon("image", "#4d78cc", 32).pixmap(32, 32))
        box.addWidget(emblem, alignment=Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Star Effects Lab")
        title.setObjectName("emptyTitle")
        box.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        text = QLabel("Open an image to start detecting stars and applying optical effects")
        text.setObjectName("muted")
        box.addWidget(text, alignment=Qt.AlignmentFlag.AlignCenter)
        self.empty_open_button = QPushButton(icon("folder", "#ffffff"), "Load Image")
        self.empty_open_button.setObjectName("accent")
        self.empty_open_button.setMinimumHeight(34)
        self.empty_open_button.clicked.connect(self.choose_open)
        box.addWidget(self.empty_open_button, alignment=Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("or drag and drop a TIFF image here")
        hint.setObjectName("muted")
        box.addWidget(hint, alignment=Qt.AlignmentFlag.AlignCenter)
        box.addStretch()
        return empty

    def _tool_button(self, name, label, callback):
        button = QToolButton()
        button.setIcon(icon(name))
        button.setToolTip(label)
        button.setAccessibleName(label)
        button.clicked.connect(callback)
        return button

    def _toolbar(self):
        hud = QFrame()
        hud.setObjectName("hud")
        hud.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        box = QHBoxLayout(hud)
        box.setContentsMargins(6, 5, 6, 5)
        box.setSpacing(4)
        self.open_button = self._tool_button("folder", "Open TIFF (Ctrl+O)", self.choose_open)
        box.addWidget(self.open_button)
        self.tool_group = QButtonGroup(self)
        for key, name, label in [
            ("pan", "pan", "Pan image"),
            ("edit", "select", "Edit star"),
            ("add", "add", "Add star"),
            ("hide", "hide", "Hide / show star"),
        ]:
            button = self._tool_button(
                name, label, lambda checked=False, k=key: self.tool_changed.emit(k)
            )
            button.setCheckable(True)
            button.setChecked(key == "pan")
            self.tool_group.addButton(button)
            box.addWidget(button)
        for text, callback in [
            ("−", lambda: self.image_view.zoom_by(1 / 1.2)),
            ("+", lambda: self.image_view.zoom_by(1.2)),
            ("Fit", self.image_view.fit_image),
            ("100%", self.image_view.actual_size),
        ]:
            button = QToolButton()
            button.setText(text)
            button.setToolTip(
                {
                    "−": "Zoom out",
                    "+": "Zoom in",
                    "Fit": "Fit image",
                    "100%": "Render full-resolution detail",
                }[text]
            )
            button.clicked.connect(callback)
            box.addWidget(button)
        self.mask_button = self._tool_button(
            "mask", "Show / hide foreground mask (view only)", self._toggle_mask
        )
        self.mask_button.setCheckable(True)
        box.addWidget(self.mask_button)
        self.view_combo = QComboBox()
        for name, mode in [
            ("Result", "result"),
            ("Original", "original"),
            ("Sky mask", "mask"),
            ("Effects only", "effects"),
            ("Compare", "compare"),
        ]:
            self.view_combo.addItem(name, mode)
        self.view_combo.setAccessibleName("Preview mode")
        self.view_combo.setToolTip(
            "Result applies Soft filter to the source stars before effects. Effects only shows added glow."
        )
        self.view_combo.currentIndexChanged.connect(
            lambda index: self._view(self.view_combo.itemData(index))
        )
        box.addWidget(self.view_combo)
        self.undo_button = self._tool_button("undo", "Undo (Ctrl+Z)", self.undo_requested.emit)
        self.redo_button = self._tool_button("redo", "Redo", self.redo_requested.emit)
        box.addWidget(self.undo_button)
        box.addWidget(self.redo_button)
        return hud

    def _general_controls(self):
        box = self.effect_panel.general_box
        self.auto_detect = QCheckBox("Automatic Star Detection")
        self.auto_detect.setChecked(True)
        box.addWidget(self.auto_detect)
        self.detection_method = QComboBox()
        self.detection_method.addItem("Original-style detector", "native")
        self.detection_method.addItem("Multiscale · larger stars", "multiscale")
        self.detection_method.setAccessibleName("Star detector")
        box.addWidget(self.detection_method)
        row = QHBoxLayout()
        self.threshold_label = QLabel("Minimum brightness")
        row.addWidget(self.threshold_label)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.04, 0.98)
        self.threshold.setValue(0.20)
        self.threshold.setSuffix("")
        self.threshold.setDecimals(2)
        self.threshold.setAccessibleName("Minimum normalized star brightness")
        self.threshold.setToolTip(
            "0.20 is the original detector threshold. Lower values include fainter candidates."
        )
        self.threshold.setFixedWidth(64)
        self.threshold.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(self.threshold)
        box.addLayout(row)
        threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider = threshold_slider
        threshold_slider.setRange(4, 98)
        threshold_slider.setValue(20)
        threshold_slider.setAccessibleName("Detection threshold")
        threshold_slider.valueChanged.connect(lambda value: self.threshold.setValue(value / 100))
        self.threshold.valueChanged.connect(
            lambda value: threshold_slider.setValue(round(value * 100))
        )
        box.addWidget(threshold_slider)
        self._threshold_method = "native"
        self._threshold_values = {"native": 0.2, "multiscale": 5.0}
        self.detection_method.currentIndexChanged.connect(self._detector_threshold_mode)
        self.analyze_button = QPushButton("Detect stars")
        box.addWidget(self.analyze_button)
        self.star_count = QLabel("No image loaded")
        self.star_count.setWordWrap(True)
        self.star_count.setObjectName("muted")
        box.addWidget(self.star_count)
        self.centers = QCheckBox("Show star centers")
        box.addWidget(self.centers)
        hint = QLabel(
            "Primary and Secondary have separate tabs under Spikes. Advanced reveals all controls.\n\nForeground is excluded automatically. The mask icon shows blue sky, red foreground and yellow uncertain areas. Multiscale also detects broader stars."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        box.addWidget(hint)
        box.addStretch()

    def _detector_threshold_mode(self, _index):
        method = self.detection_method.currentData()
        self._threshold_values[self._threshold_method] = self.threshold.value()
        self._threshold_method = method
        multi = method == "multiscale"
        self.threshold_label.setText("Detection threshold" if multi else "Minimum brightness")
        with QSignalBlocker(self.threshold), QSignalBlocker(self.threshold_slider):
            self.threshold.setRange(*((1, 30) if multi else (0.04, 0.98)))
            self.threshold.setSuffix(" σ" if multi else "")
            self.threshold.setValue(self._threshold_values[method])
            self.threshold_slider.setRange(*((100, 3000) if multi else (4, 98)))
            self.threshold_slider.setValue(round(self.threshold.value() * 100))
        self.threshold.setToolTip(
            "Local-noise sigma threshold for multiscale sources."
            if multi
            else "0.20 is the original proxy threshold. Lower values include fainter candidates."
        )

    def _footer(self):
        footer = QFrame()
        footer.setObjectName("footer")
        box = QHBoxLayout(footer)
        box.setContentsMargins(12, 8, 12, 8)
        box.setSpacing(8)
        label = QLabel("Preset:")
        label.setObjectName("muted")
        box.addWidget(label)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            ["Default Spikes", "Subtle Glow", "Dramatic Diffraction", "Custom"]
        )
        self.preset_combo.setFixedWidth(164)
        self.preset_combo.currentIndexChanged.connect(self._preset)
        box.addWidget(self.preset_combo)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(lambda: self._preset(0))
        box.addWidget(self.reset_button)
        self.load_preset_button = self._tool_button(
            "folder", "Load preset JSON", self.choose_preset_load
        )
        self.save_preset_button = self._tool_button(
            "save", "Save preset JSON", self.choose_preset_save
        )
        box.addWidget(self.load_preset_button)
        box.addWidget(self.save_preset_button)
        box.addStretch()
        activity = QVBoxLayout()
        activity.setSpacing(3)
        self.status_text = QLabel("Ready")
        self.status_text.setObjectName("muted")
        self.status_text.setMaximumWidth(210)
        self.status_text.setMinimumWidth(75)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(3)
        self.progress.hide()
        activity.addWidget(self.status_text)
        activity.addWidget(self.progress)
        box.addLayout(activity)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.hide()
        box.addWidget(self.cancel_button)
        self.before_button = QPushButton(icon("eye"), "Hold for Before")
        self.before_button.pressed.connect(self._before_pressed)
        self.before_button.released.connect(self._before_released)
        box.addWidget(self.before_button)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip("Recalculate the preview using current settings")
        self.apply_button.clicked.connect(self.apply_requested.emit)
        box.addWidget(self.apply_button)
        self.export_button = QPushButton(icon("download", "#ffffff"), "Export")
        self.export_button.setObjectName("accent")
        self.export_button.clicked.connect(self.choose_export)
        box.addWidget(self.export_button)
        return footer

    def _preset(self, index):
        if index == 3 or (self.controller and self.controller.exclusive):
            return
        settings = EffectSettings()
        if index == 1:
            settings = replace(
                settings,
                primary=replace(settings.primary, intensity=12, length=12),
                soft=replace(settings.soft, intensity=12, size=9),
            )
        elif index == 2:
            settings = replace(
                settings,
                primary=replace(settings.primary, intensity=85, length=40),
                secondary=replace(settings.secondary, enabled=True, intensity=20),
            )
        self.effect_panel.set_settings(settings)
        self.effect_panel.settings_changed.emit(settings)
        with QSignalBlocker(self.preset_combo):
            self.preset_combo.setCurrentIndex(index)

    def mark_custom(self):
        with QSignalBlocker(self.preset_combo):
            self.preset_combo.setCurrentIndex(3)

    def _before_pressed(self):
        if self._before_mode is None:
            self._before_mode = self.mode
            self._view("original")

    def _before_released(self):
        if self._before_mode is not None:
            self._view(self._before_mode)
            self._before_mode = None

    def _toggle_mask(self):
        self._view(self._mask_return_mode if self.mode == "mask" else "mask")

    def _view(self, mode):
        if mode == "mask" and self.mode != "mask" and self._before_mode is None:
            self._mask_return_mode = self.mode
        self.mode = mode
        with QSignalBlocker(self.mask_button):
            self.mask_button.setChecked(mode == "mask")
        with QSignalBlocker(self.view_combo):
            self.view_combo.setCurrentIndex(self.view_combo.findData(mode))
        self.view_changed.emit(mode)

    def _shortcuts(self):
        for title, shortcut, callback in [
            ("Open", QKeySequence.StandardKey.Open, self.choose_open),
            ("Export", QKeySequence.StandardKey.SaveAs, self.choose_export),
            ("Undo", QKeySequence.StandardKey.Undo, self.undo_requested.emit),
            ("Redo", QKeySequence.StandardKey.Redo, self.redo_requested.emit),
        ]:
            action = QAction(title, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.addAction(action)

    def set_activity(self, kind):
        self._activity = {
            "open": "Opening TIFF",
            "open_masked": "Opening TIFF · sky mask",
            "open_preview": "Reading image preview",
            "analyze": "Detecting stars",
            "mask": "Building sky mask",
            "preview": "Rendering preview",
            "export": "Exporting TIFF",
            "cancel": "Cancelling",
        }[kind]
        self.clock.start()
        self.activity_timer.start()
        self._activity_tick()

    def _activity_tick(self):
        self.status_text.setText(f"{self._activity}… {self.clock.elapsed() // 1000}s")

    def finish_activity(self):
        self.activity_timer.stop()

    def set_exclusive(self, exclusive):
        for widget in (
            self.preset_combo,
            self.reset_button,
            self.apply_button,
            self.empty_open_button,
        ):
            widget.setEnabled(not exclusive and (self._loaded or widget is not self.apply_button))

    def choose_open(self):
        if self.controller and self.controller.exclusive:
            return
        directory = self.preferences.value("dialogs/open_image_directory", "", type=str)
        if not directory or not Path(directory).is_dir():
            directory = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.PicturesLocation
            )
            if not directory or not Path(directory).is_dir():
                directory = str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", directory, "TIFF images (*.tif *.tiff)"
        )
        if path:
            self.preferences.setValue(
                "dialogs/open_image_directory", str(Path(path).resolve().parent)
            )
            self.preferences.sync()
            self.open_requested.emit(path)

    def choose_export(self):
        if not self.export_button.isEnabled():
            return
        from PySide6.QtWidgets import QInputDialog

        kind, ok = QInputDialog.getItem(
            self,
            "Export",
            "Image content",
            ["Result", "Effects only (added glow on black)"],
            0,
            False,
        )
        if ok:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export 16-bit TIFF", "star-effects.tif", "TIFF (*.tif)"
            )
            if path:
                self.export_requested.emit(path, kind.startswith("Effects"))

    def choose_preset_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "Preset (*.json)")
        if path:
            self.preset_load_requested.emit(path)

    def choose_preset_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save preset", "effects.json", "Preset (*.json)"
        )
        if path:
            self.preset_save_requested.emit(path)

    def set_loaded(self, loaded):
        self._loaded = loaded
        for widget in (
            self.export_button,
            self.analyze_button,
            self.undo_button,
            self.redo_button,
            self.before_button,
            self.apply_button,
        ):
            widget.setEnabled(loaded)
        self.stack.setCurrentIndex(1 if loaded else 0)
        self.effect_panel.setEnabled(True)
        self.hud.setVisible(loaded)
        self.image_info.setVisible(loaded)
        self.set_exclusive(False)

    def show_error(self, text):
        self.error_label.setText(text)
        self.error_label.setVisible(bool(text))

    def closeEvent(self, event):
        if self.controller is not None and not self.controller.shutdown():
            event.ignore()
            self.status_text.setText("Finishing cancellation…")
        else:
            self.activity_timer.stop()
            event.accept()
