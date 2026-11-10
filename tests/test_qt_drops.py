from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent

from starlab.presentation.qt.image_view import ImageView
from starlab.presentation.qt.main_window import Canvas


@pytest.mark.parametrize("widget_type", [Canvas, ImageView])
@pytest.mark.parametrize("with_tiff", [False, True])
def test_drop_surfaces_select_first_local_tiff(qtbot, tmp_path, widget_type, with_tiff):
    widget = widget_type()
    qtbot.addWidget(widget)
    paths = [QUrl("https://example.com/remote.tif"), QUrl.fromLocalFile(str(tmp_path / "note.txt"))]
    selected = str(tmp_path / "first.TIFF")
    if with_tiff:
        paths.extend(
            [QUrl.fromLocalFile(selected), QUrl.fromLocalFile(str(tmp_path / "second.tif"))]
        )
    mime = QMimeData()
    mime.setUrls(paths)
    events = (
        QDragEnterEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
        QDragMoveEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
        QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    dropped = []
    widget.file_dropped.connect(dropped.append)
    widget.dragEnterEvent(events[0])
    widget.dragMoveEvent(events[1])
    widget.dropEvent(events[2])
    assert [event.isAccepted() for event in events] == [with_tiff] * 3
    assert [Path(path) for path in dropped] == ([Path(selected)] if with_tiff else [])
