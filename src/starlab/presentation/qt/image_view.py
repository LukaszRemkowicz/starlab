import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ...domain.imaging import Pixels, display_pixels
from ...domain.stars import Star


class ImageView(QGraphicsView):
    image_clicked = Signal(float, float)
    file_dropped = Signal(str)
    detail_requested = Signal()

    def __init__(self, parent=None):
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

    def set_pixels(
        self,
        data: Pixels,
        color_space: str,
        scale: int,
        width: int,
        height: int,
        *,
        preserve_detail=False,
    ) -> None:
        rgb = display_pixels(data, color_space)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        rgb = np.ascontiguousarray(rgb)
        image = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self.item.setPixmap(QPixmap.fromImage(image))
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
        was_fitted = self.fitted
        self.fitted = True
        self.detail_timer.stop()
        self.detail_item.setPixmap(QPixmap())
        if self.has_image:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            if not was_fitted:
                self.detail_requested.emit()

    def actual_size(self) -> None:
        self.fitted = False
        self.resetTransform()
        self.detail_requested.emit()

    def _queue_detail(self, *_):
        if self.has_image and not self.fitted:
            self.detail_timer.start()

    def visible_region(self):
        return self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())

    def set_region_pixels(self, data, color_space, scale, origin):
        rgb = display_pixels(data, color_space)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        rgb = np.ascontiguousarray(rgb)
        image = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self.detail_item.setPixmap(QPixmap.fromImage(image))
        self.detail_item.setScale(scale)
        self.detail_item.setPos(origin[0] + (1 - scale) / 2, origin[1] + (1 - scale) / 2)
        self.viewport().update()

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if tool == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.OpenHandCursor if tool == "pan" else Qt.CursorShape.CrossCursor
        )

    def wheelEvent(self, event) -> None:
        if self.has_image:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.zoom_by(factor)
            event.accept()

    def zoom_by(self, factor: float) -> None:
        if self.has_image and 0.01 <= self.transform().m11() * factor <= 32:
            self.fitted = False
            self.scale(factor, factor)
            self._queue_detail()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.fitted:
            self.fit_image()
        else:
            self._queue_detail()

    def mousePressEvent(self, event) -> None:
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

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space_pan = False
            self.set_tool(self.tool)
        else:
            super().keyReleaseEvent(event)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
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

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()
