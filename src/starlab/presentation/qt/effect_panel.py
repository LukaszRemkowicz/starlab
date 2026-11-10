from dataclasses import replace

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...domain.effects.settings import KINDS, PARAMETERS, RANGES, EffectSettings

LABELS = {
    "quantity": "Quantity",
    "intensity": "Intensity",
    "length": "Length",
    "angle": "Angle",
    "points": "Number of points",
    "sharpness": "Spike sharpness",
    "spread": "Spike spread",
    "saturation": "Color saturation",
    "hue": "Color hue",
    "rainbow": "Rainbow intensity",
    "frequency": "Rainbow frequency",
    "phase": "Rainbow phase",
    "size": "Size",
    "soft_filter": "Soft filter",
    "falloff": "Edge falloff",
    "inner_width": "Inner edge width",
    "inner_falloff": "Inner edge falloff",
    "outer_width": "Outer edge width",
    "outer_falloff": "Outer edge falloff",
}
TITLES = {
    "primary": "Primary",
    "secondary": "Secondary",
    "soft": "Soft flare",
    "ring": "Ring flare",
}
BASIC = {"quantity", "intensity", "length", "angle", "points", "size", "soft_filter"}


class ParameterControl(QWidget):
    changed = Signal(float)
    edit_started = Signal()
    edit_finished = Signal()

    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self.key = key
        tips = {
            "quantity": "Brightness threshold for this effect. Higher values include fainter detected sources across the image.",
            "saturation": "0 preserves the source color. Positive values add saturation to this effect.",
            "rainbow": "0 adds no rainbow modulation. Increase to color the rays.",
            "soft_filter": "First shrink small stars and enlarge large stars in the original image, then apply effects. Independent of halo Quantity, Intensity and Size; 0 leaves source pixels unchanged.",
        }
        if key in tips:
            self.setToolTip(tips[key])
        self.factor = 10 if key == "angle" else 1
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        row = QHBoxLayout()
        label = QLabel(LABELS[key])
        self.spin = QDoubleSpinBox()
        self.spin.setRange(*RANGES[key])
        self.spin.setDecimals(1 if key == "angle" else 0)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(64)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin.setAccessibleName(LABELS[key])
        if key == "angle":
            self.spin.setSuffix("°")
        label.setBuddy(self.spin)
        row.addWidget(label)
        row.addStretch()
        row.addWidget(self.spin)
        layout.addLayout(row)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setTracking(True)
        self.slider.setRange(*(int(v * self.factor) for v in RANGES[key]))
        self.slider.setAccessibleName(LABELS[key])
        layout.addWidget(self.slider)
        self.spin.valueChanged.connect(self._spin_changed)
        self.slider.valueChanged.connect(self._slider_changed)
        self.slider.sliderPressed.connect(self.edit_started)
        self.slider.sliderReleased.connect(self.edit_finished)

    def set_value(self, value: float) -> None:
        with QSignalBlocker(self.spin), QSignalBlocker(self.slider):
            self.spin.setValue(value)
            self.slider.setValue(round(value * self.factor))

    def _spin_changed(self, value: float) -> None:
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(value * self.factor))
        self.changed.emit(value)

    def _slider_changed(self, value: int) -> None:
        numeric = value / self.factor
        with QSignalBlocker(self.spin):
            self.spin.setValue(numeric)
        self.changed.emit(numeric)


class EffectPanel(QWidget):
    settings_changed = Signal(object)
    edit_started = Signal()
    edit_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = EffectSettings()
        self.controls: dict[str, dict[str, ParameterControl]] = {}
        self.enabled: dict[str, QCheckBox] = {}
        self.source_counts: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.general = QWidget()
        self.general_box = QVBoxLayout(self.general)
        self.general_box.setContentsMargins(14, 16, 14, 14)
        self.general_box.setSpacing(16)
        modes = QHBoxLayout()
        modes.setContentsMargins(14, 6, 14, 6)
        self.basic, self.advanced = QRadioButton("Basic"), QRadioButton("Advanced")
        self.advanced.setChecked(True)
        modes.addWidget(self.basic)
        modes.addWidget(self.advanced)
        modes.addStretch()
        layout.addLayout(modes)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setDrawBase(False)
        layout.addWidget(self.tabs)
        self.tabs.addTab(self.general, "General")
        spikes = QWidget()
        spike_box = QVBoxLayout(spikes)
        spike_box.setContentsMargins(0, 0, 0, 0)
        self.spike_selector = QTabWidget()
        self.spike_selector.setDocumentMode(True)
        self.spike_selector.tabBar().setDrawBase(False)
        spike_box.addWidget(self.spike_selector)
        self.tabs.addTab(spikes, "Spikes")
        for kind in KINDS:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            body = QWidget()
            body.setObjectName("parameterBody")
            box = QVBoxLayout(body)
            box.setContentsMargins(14, 10, 14, 10)
            box.setSpacing(8)
            toggle = QCheckBox(f"Enable {TITLES[kind].lower()}")
            toggle.toggled.connect(lambda checked, k=kind: self._change(k, "enabled", checked))
            header = QHBoxLayout()
            header.addWidget(toggle)
            header.addStretch()
            count = QLabel("0 sources")
            count.setObjectName("muted")
            count.setToolTip(
                "Sources receiving this effect / total detected sources. Quantity changes the brightness cutoff."
                + (
                    " Soft filter processes measured source stars independently of this halo count."
                    if kind == "soft"
                    else ""
                )
            )
            header.addWidget(count)
            self.source_counts[kind] = count
            box.addLayout(header)
            self.enabled[kind] = toggle
            self.controls[kind] = {}
            for key in PARAMETERS[kind]:
                control = ParameterControl(key)
                control.changed.connect(
                    lambda value, k=kind, p=key: self._change(
                        k, p, int(value) if p == "points" else value
                    )
                )
                control.edit_started.connect(self.edit_started)
                control.edit_finished.connect(self.edit_finished)
                box.addWidget(control)
                self.controls[kind][key] = control
            reset = QPushButton("Reset this group")
            reset.clicked.connect(lambda checked=False, k=kind: self._reset(k))
            box.addWidget(reset)
            box.addStretch()
            scroll.setWidget(body)
            if kind in ("primary", "secondary"):
                self.spike_selector.addTab(scroll, TITLES[kind])
            else:
                self.tabs.addTab(scroll, TITLES[kind].title())
        self.basic.toggled.connect(self._mode)
        self.set_settings(self.settings)

    def _change(self, kind: str, key: str, value) -> None:
        group = replace(getattr(self.settings, kind), **{key: value})
        self.settings = replace(self.settings, **{kind: group})
        self.settings_changed.emit(self.settings)

    def _reset(self, kind: str) -> None:
        self.set_settings(replace(self.settings, **{kind: getattr(EffectSettings(), kind)}))
        self.settings_changed.emit(self.settings)

    def set_settings(self, settings: EffectSettings) -> None:
        self.settings = settings
        for kind in KINDS:
            group = getattr(settings, kind)
            with QSignalBlocker(self.enabled[kind]):
                self.enabled[kind].setChecked(group.enabled)
            for key, control in self.controls[kind].items():
                control.set_value(getattr(group, key))

    def _mode(self, basic: bool) -> None:
        for kind in KINDS:
            for key, control in self.controls[kind].items():
                control.setVisible(not basic or key in BASIC)
