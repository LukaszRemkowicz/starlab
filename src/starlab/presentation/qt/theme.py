"""Native Qt styling and small vector icons matching the supplied GUI reference."""

from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

STYLE = """
QWidget { color: #d5dbe5; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 12px; }
QMainWindow, QDialog { background: #181a1f; }
QWidget#canvas, QGraphicsView { background: #0f1013; border: none; }
QWidget#inspector, QScrollArea, QStackedWidget, QTabWidget::pane { background: #21252b; border: none; }
QWidget#parameterBody { background: #21252b; }
QFrame#panelHeader, QFrame#footer { background: #282c34; border: 1px solid #333842; }
QFrame#hud { background: #21252b; border: 1px solid #333842; border-radius: 5px; }
QLabel { background: transparent; }
QLabel#muted { color: #8c9bb1; }
QLabel#emptyTitle { font-size: 15px; font-weight: 600; }
QLabel#emptyIcon { background: #21252b; border: 1px solid #333842; border-radius: 34px; }
QLabel#imageInfo { color: #a0aec2; background: #181a1f; padding: 5px 8px; border-radius: 3px; }
QLabel#error { color: #ffaaa0; background: #39282c; padding: 8px; }
QPushButton, QToolButton, QComboBox { background: #181a1f; border: 1px solid #333842;
    border-radius: 4px; padding: 5px 9px; min-height: 18px; }
QPushButton:hover, QToolButton:hover, QComboBox:hover { background: #303640; border-color: #536175; }
QPushButton:pressed, QToolButton:pressed { background: #344866; }
QPushButton:focus, QToolButton:focus, QComboBox:focus { border-color: #729cf2; }
QPushButton#accent { background: #4d78cc; border-color: #4d78cc; color: white; font-weight: 600; }
QPushButton#accent:hover { background: #5c87df; }
QPushButton#accent:disabled { background: #28354a; border-color: #35445d; color: #8c9bb1; }
QPushButton:disabled, QToolButton:disabled { color: #677182; border-color: #30353d; background: #202329; }
QToolButton { padding: 4px; }
QToolButton:checked { color: #91b6ff; background: #2c3c55; border-color: #4d78cc; }
QComboBox { padding-right: 20px; }
QComboBox::drop-down { width: 18px; border: none; }
QComboBox QAbstractItemView { background: #21252b; color: #d5dbe5; selection-background-color: #3a5684; }
QTabWidget::pane { border-top: 1px solid #333842; }
QTabBar { background: #181a1f; }
QTabBar::tab { background: #181a1f; color: #9babc1; padding: 10px 8px; border: none;
    border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #7fa5f5; background: #21252b; border-bottom-color: #4d78cc; }
QTabBar::tab:hover { color: #d5dbe5; }
QSlider::groove:horizontal { height: 3px; background: #333842; border-radius: 1px; }
QSlider::handle:horizontal { background: #4d78cc; width: 12px; height: 12px; margin: -5px 0;
    border: none; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: #87acf5; }
QSlider::handle:horizontal:disabled { background: #4b576c; }
QSlider { min-height: 16px; }
QDoubleSpinBox { color: #80a5f1; background: #2c313a; border: 1px solid #3a424e;
    border-radius: 3px; padding: 0px 3px; min-height: 18px; selection-background-color: #385c96; }
QDoubleSpinBox:hover { background: #333a45; border-color: #536175; }
QDoubleSpinBox:focus { background: #333a45; border-color: #729cf2; }
QDoubleSpinBox:disabled { color: #6e7c91; background: #252930; border-color: #30353d; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0px; height: 0px; border: none; }
QCheckBox, QRadioButton { spacing: 7px; background: transparent; }
QCheckBox::indicator { width: 12px; height: 12px; background: #181a1f; border: 1px solid #56647a; border-radius: 2px; }
QCheckBox::indicator:checked { background: #4d78cc; border-color: #8eaff2; }
QRadioButton::indicator { width: 11px; height: 11px; background: #181a1f; border: 1px solid #56647a; border-radius: 6px; }
QRadioButton::indicator:checked { background: #4d78cc; border-color: #8eaff2; }
QScrollBar:vertical { width: 5px; background: #21252b; margin: 0; }
QScrollBar::handle:vertical { background: #495364; min-height: 24px; border-radius: 2px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QProgressBar { border: none; background: #333842; border-radius: 2px; max-height: 3px; }
QProgressBar::chunk { background: #4d78cc; }
QSplitter::handle { background: #333842; width: 1px; }
QToolTip { color: #e1e7f0; background: #282c34; border: 1px solid #536175; padding: 4px; }
"""

PATHS = {
    "mask": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="m3 15 5-6 5 5 3-3 5 5M5 19l3-3m2 3 3-3m2 3 3-3"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8" cy="8" r="2"/><path d="m21 15-5-5L5 21"/>',
    "folder": '<path d="M3 7V5h6l2 2h10v3M3 7h6l2 3h11l-3 10H2L3 7Z"/>',
    "eye": '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
    "download": '<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
    "pan": '<path d="M8 13V5a2 2 0 0 1 4 0v7-9a2 2 0 0 1 4 0v9-6a2 2 0 0 1 4 0v9c0 5-3 7-7 7-3 0-5-2-7-5l-3-5c-1-2 2-3 3-1l2 2Z"/>',
    "select": '<path d="m4 3 16 10-7 1-3 7L4 3Z"/>',
    "add": '<circle cx="12" cy="12" r="9"/><path d="M12 7v10M7 12h10"/>',
    "hide": '<path d="m3 3 18 18M9 5c7-2 13 7 13 7l-4 4M6 6l-4 6s7 10 15 6"/>',
    "undo": '<path d="m8 4-5 5 5 5M3 9h11a7 7 0 0 1 0 14"/>',
    "redo": '<path d="m16 4 5 5-5 5M21 9H10a7 7 0 0 0 0 14"/>',
    "save": '<path d="M4 3h13l4 4v14H3V3h1Z M7 3v6h10V3M7 21v-8h10v8"/>',
}

_chevron = (Path(__file__).parent / "assets" / "chevron.svg").as_posix()
STYLE += f'QComboBox::down-arrow {{ image: url("{_chevron}"); width: 12px; height: 12px; }}'


def icon(name: str, color: str = "#a9bad1", size: int = 18) -> QIcon:
    """Rasterize a bundled vector path on the GUI thread at a fixed high-DPI ratio."""
    svg: str = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{PATHS[name]}</svg>'
    pixmap: QPixmap = QPixmap(size * 2, size * 2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter: QPainter = QPainter(pixmap)
    QSvgRenderer(QByteArray(svg.encode())).render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)
