"""GUI-thread image layers, full-image coordinates, and direct Qt navigation events."""

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget

from ...domain.imaging import Pixels, display_pixels
from ...domain.stars import Star
from .drop_paths import local_tiff_path


def _pixmap(data: Pixels, color_space: str) -> QPixmap:
    """Create a GUI-thread pixmap with copied image storage independent of NumPy."""
    rgb = display_pixels(data, color_space)
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    rgb = np.ascontiguousarray(rgb)
    image = QImage(
        rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
    ).copy()
    return QPixmap.fromImage(image)


@dataclass(frozen=True)
class ViewState:
    """Implicitly shared pixmaps and navigation state for a temporary thumbnail."""

    overview: tuple[QPixmap, float, QPointF]
    detail: tuple[QPixmap, float, QPointF]
    scene_rect: QRectF
    transform: QTransform
    scroll: tuple[int, int]
    fitted: bool
    has_image: bool
    stars: tuple[Star, ...]


class ImageView(QGraphicsView):
    """Display an overview and native ROI with source-pixel centers aligned at +0.5."""

    image_clicked = Signal(float, float)
    file_dropped = Signal(str)
    detail_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create GUI-owned graphics items and debounce detail requests during navigation."""
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.detail_item = QGraphicsPixmapItem()
        self.detail_item.setZValue(1)
        self.scene().addItem(self.detail_item)
        self.detail_timer = QTimer(self)
        self.detail_timer.setSingleShot(True)
        self.detail_timer.setInterval(16)
        self.detail_timer.timeout.connect(self.detail_requested)
        self.horizontalScrollBar().valueChanged.connect(self._queue_detail)
        self.verticalScrollBar().valueChanged.connect(self._queue_detail)
        self.setBackgroundBrush(QColor("#0f1013"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setAcceptDrops(True)
        self.setAccessibleName(
            "Image preview. Mouse wheel zooms. Select an edit tool to change stars."
        )
        self.stars: tuple[Star, ...] = ()
        self.show_centers = False
        self.tool = "pan"
        self.fitted = True
        self.has_image = False
        self._space_pan = False

    def capture_state(self) -> ViewState:
        """Retain both image layers and navigation before showing an import thumbnail."""
        return ViewState(
            (self.item.pixmap(), self.item.scale(), self.item.pos()),
            (self.detail_item.pixmap(), self.detail_item.scale(), self.detail_item.pos()),
            self.sceneRect(),
            self.transform(),
            (self.horizontalScrollBar().value(), self.verticalScrollBar().value()),
            self.fitted,
            self.has_image,
            self.stars,
        )

    def restore_state(self, state: ViewState) -> None:
        """Restore a committed document without requesting replacement preview work."""
        self.fitted = False
        for item, layer in ((self.item, state.overview), (self.detail_item, state.detail)):
            pixmap, scale, position = layer
            item.setPixmap(pixmap)
            item.setScale(scale)
            item.setPos(position)
        self.scene().setSceneRect(state.scene_rect)
        self.setTransform(state.transform)
        self.horizontalScrollBar().setValue(state.scroll[0])
        self.verticalScrollBar().setValue(state.scroll[1])
        self.fitted, self.has_image, self.stars = state.fitted, state.has_image, state.stars
        self.detail_timer.stop()
        self.viewport().update()

    def clear_detail(self) -> None:
        """Expose the overview while detail for a different display mode is pending."""
        self.detail_item.setPixmap(QPixmap())

    def set_pixels(
        self,
        data: Pixels,
        color_space: str,
        scale: int,
        width: int,
        height: int,
        *,
        preserve_detail: bool = False,
    ) -> None:
        """Update the overview without copying full-resolution source arrays."""
        self.item.setPixmap(_pixmap(data, color_space))
        if not preserve_detail:
            self.detail_item.setPixmap(QPixmap())
        self.item.setScale(scale)
        # A subsampled pixel represents full-image index i*scale, centered at +0.5.
        self.item.setPos((1 - scale) / 2, (1 - scale) / 2)
        size_changed = self.sceneRect() != QRectF(0, 0, width, height) or not self.has_image
        self.scene().setSceneRect(QRectF(0, 0, width, height))
        self.has_image = True
        if self.fitted and size_changed:
            self.fit_image()
        self.viewport().update()

    def fit_image(self) -> None:
        """Fit the full-image rectangle and discard any obsolete detail layer."""
        was_fitted = self.fitted
        self.fitted = True
        self.detail_timer.stop()
        self.detail_item.setPixmap(QPixmap())
        if self.has_image:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            if not was_fitted:
                self.detail_requested.emit()

    def actual_size(self) -> None:
        """Show one display pixel per source pixel and request native detail."""
        self.fitted = False
        self.resetTransform()
        self.detail_requested.emit()

    def _queue_detail(self, *_: object) -> None:
        """Debounce scrollbar changes into one detail request when the image is zoomed."""
        if self.has_image and not self.fitted:
            self.detail_timer.start()

    def visible_region(self) -> QRectF:
        """Return visible bounds clipped to the full-image coordinate rectangle."""
        return self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())

    def set_region_pixels(
        self, data: Pixels, color_space: str, scale: int, origin: tuple[int, int]
    ) -> None:
        """Place sampled ROI pixels using a full-image origin and integer sampling step."""
        self.detail_item.setPixmap(_pixmap(data, color_space))
        self.detail_item.setScale(scale)
        self.detail_item.setPos(origin[0] + (1 - scale) / 2, origin[1] + (1 - scale) / 2)
        self.viewport().update()

    def set_tool(self, tool: str) -> None:
        """Select hand panning or crosshair-based catalog editing."""
        self.tool = tool
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if tool == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.OpenHandCursor if tool == "pan" else Qt.CursorShape.CrossCursor
        )

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom around the pointer when an image is loaded."""
        if self.has_image:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.zoom_by(factor)
            event.accept()

    def zoom_by(self, factor: float) -> None:
        """Apply bounded zoom and request detail after navigation settles."""
        if self.has_image and 0.01 <= self.transform().m11() * factor <= 32:
            self.fitted = False
            self.scale(factor, factor)
            self._queue_detail()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Maintain fitted scale or request detail for the resized viewport."""
        super().resizeEvent(event)
        if self.fitted:
            self.fit_image()
        else:
            self._queue_detail()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Emit source coordinates for editing tools or delegate pan gestures."""
        if (
            self.has_image
            and self.tool != "pan"
            and not self._space_pan
            and event.button() == Qt.MouseButton.LeftButton
        ):
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point):
                self.image_clicked.emit(max(0, point.x() - 0.5), max(0, point.y() - 0.5))
            event.accept()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Temporarily enable hand panning while Space is held."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Restore the selected edit tool when Space is released."""
        if event.key() == Qt.Key.Key_Space:
            self._space_pan = False
            self.set_tool(self.tool)
        else:
            super().keyReleaseEvent(event)

    def drawForeground(self, painter: QPainter, rect: QRectF | QRect) -> None:
        """Draw visible and hidden catalog markers at their full-image pixel centers."""
        rect = QRectF(rect)
        if not self.show_centers:
            return
        pen = QPen(QColor("#87b8ff"), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        radius = 7 / max(self.transform().m11(), 0.01)
        for star in self.stars:
            star_x, star_y = star.x + 0.5, star.y + 0.5
            if not rect.contains(QPointF(star_x, star_y)):
                continue
            if star.visible:
                painter.drawEllipse(QPointF(star_x, star_y), radius, radius)
            else:
                painter.drawLine(
                    QPointF(star_x - radius, star_y - radius),
                    QPointF(star_x + radius, star_y + radius),
                )
                painter.drawLine(
                    QPointF(star_x - radius, star_y + radius),
                    QPointF(star_x + radius, star_y - radius),
                )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drags containing a local TIFF URL."""
        if local_tiff_path(event.mimeData()) is not None:
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Keep accepting only drags with an eligible local TIFF."""
        if local_tiff_path(event.mimeData()) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Request the first eligible local TIFF from a drop."""
        path = local_tiff_path(event.mimeData())
        if path is not None:
            self.file_dropped.emit(path)
            event.acceptProposedAction()
