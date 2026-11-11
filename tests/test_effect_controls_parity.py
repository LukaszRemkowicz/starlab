"""Keep all reference effect sliders discoverable in the native inspector."""

from dataclasses import replace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QTabWidget

from starlab.domain.effects.settings import EffectSettings
from starlab.presentation.qt.effect_panel import EffectPanel
from starlab.presentation.qt.theme import STYLE


@pytest.mark.parametrize(
    ("kind", "tab", "spike_tab", "names"),
    [
        (
            "primary",
            1,
            0,
            [
                "Quantity",
                "Intensity",
                "Length",
                "Angle",
                "Number of points",
                "Spike sharpness",
                "Spike spread",
                "Color saturation",
                "Color hue",
                "Rainbow intensity",
                "Rainbow frequency",
                "Rainbow phase",
            ],
        ),
        (
            "secondary",
            1,
            1,
            [
                "Quantity",
                "Intensity",
                "Length",
                "Angle",
                "Spike sharpness",
                "Spike spread",
                "Color saturation",
                "Color hue",
                "Rainbow intensity",
                "Rainbow frequency",
                "Rainbow phase",
            ],
        ),
        (
            "soft",
            2,
            None,
            [
                "Quantity",
                "Intensity",
                "Size",
                "Soft filter",
                "Color saturation",
                "Color hue",
                "Edge falloff",
            ],
        ),
        (
            "ring",
            3,
            None,
            [
                "Quantity",
                "Intensity",
                "Size",
                "Color saturation",
                "Color hue",
                "Inner edge width",
                "Inner edge falloff",
                "Outer edge width",
                "Outer edge falloff",
            ],
        ),
    ],
)
def test_advanced_exposes_every_reference_slider(qtbot, kind, tab, spike_tab, names):
    panel = EffectPanel()
    qtbot.addWidget(panel)
    panel.resize(340, 760)
    panel.show()
    qtbot.mouseClick(
        panel.tabs.tabBar(),
        Qt.MouseButton.LeftButton,
        pos=panel.tabs.tabBar().tabRect(tab).center(),
    )
    if spike_tab is not None:
        assert isinstance(panel.spike_selector, QTabWidget)
        assert [panel.spike_selector.tabText(i) for i in range(2)] == ["Primary", "Secondary"]
        qtbot.mouseClick(
            panel.spike_selector.tabBar(),
            Qt.MouseButton.LeftButton,
            pos=panel.spike_selector.tabBar().tabRect(spike_tab).center(),
        )
    assert panel.advanced.isChecked()
    assert panel.basic.isVisible() and panel.advanced.isVisible()
    assert [control.slider.accessibleName() for control in panel.controls[kind].values()] == names
    assert all(control.slider.isVisible() for control in panel.controls[kind].values())
    assert all(control.spin.isVisible() for control in panel.controls[kind].values())


def test_basic_and_advanced_keep_all_group_values_and_restore_sliders(qtbot):
    panel = EffectPanel()
    qtbot.addWidget(panel)
    settings = EffectSettings()
    settings = replace(
        settings,
        **{
            kind: replace(getattr(settings, kind), intensity=71, saturation=23, hue=42)
            for kind in ("primary", "secondary", "soft", "ring")
        },
    )
    panel.set_settings(settings)
    panel.tabs.setCurrentIndex(3)
    panel.show()
    emitted = []
    panel.settings_changed.connect(emitted.append)
    assert panel.basic.isVisible()
    qtbot.mouseClick(panel.basic, Qt.MouseButton.LeftButton)
    assert all(controls["saturation"].isHidden() for controls in panel.controls.values())
    assert all(not controls["intensity"].isHidden() for controls in panel.controls.values())
    qtbot.mouseClick(panel.advanced, Qt.MouseButton.LeftButton)
    assert all(
        not control.isHidden()
        for controls in panel.controls.values()
        for control in controls.values()
    )
    assert panel.settings == settings
    assert emitted == []
    for controls in panel.controls.values():
        assert controls["saturation"].spin.value() == 23
        assert controls["hue"].spin.value() == 42


def test_primary_fits_tall_inspector_and_scrolls_on_small_displays(qtbot):
    panel = EffectPanel()
    qtbot.addWidget(panel)
    panel.setStyleSheet(STYLE)
    panel.resize(340, 760)
    panel.tabs.setCurrentIndex(1)
    panel.show()
    qtbot.wait(1)
    scroll = panel.controls["primary"]["phase"].parentWidget().parentWidget().parentWidget()
    assert isinstance(scroll, QScrollArea)
    assert scroll.verticalScrollBar().maximum() == 0
    panel.resize(340, 440)
    qtbot.wait(1)
    assert scroll.verticalScrollBar().maximum() > 0
    last = panel.controls["primary"]["phase"]
    scroll.ensureWidgetVisible(last)
    qtbot.wait(1)
    assert scroll.viewport().rect().contains(last.mapTo(scroll.viewport(), last.rect().center()))
