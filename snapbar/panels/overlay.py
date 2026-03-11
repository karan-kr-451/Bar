"""
Full-screen drag-to-select overlay for region screenshots.

Intentionally NOT stealth — the user must see and interact with it.
Spans all connected monitors.
"""

from PyQt6.QtWidgets import QWidget, QLabel, QRubberBand
from PyQt6.QtCore    import Qt, QRect, QSize, QPoint, pyqtSignal
from PyQt6.QtGui     import QColor, QFont, QPainter, QGuiApplication, QCursor

from snapbar.core.constants import PANEL_FLAGS
from snapbar.core.logging_config import get_logger

logger = get_logger("snapbar.overlay")


class SelectionOverlay(QWidget):
    region_selected = pyqtSignal(QRect)   # emitted on mouse-release with selection rect

    def __init__(self):
        super().__init__()
        self.setWindowFlags(PANEL_FLAGS)
        # No WA_TranslucentBackground — causes UpdateLayeredWindowIndirect on Windows.
        # Dark fill is drawn manually in paintEvent instead.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        logger.info("SelectionOverlay initialized")

        # Span all monitors
        full = QRect()
        for scr in QGuiApplication.screens():
            full = full.united(scr.geometry())
        self.setGeometry(full)

        self.origin = QPoint()
        self.band   = QRubberBand(QRubberBand.Shape.Rectangle, self)

        # Hint label centred on primary monitor
        hint = QLabel("Drag to select  ·  ESC to cancel", self)
        hint.setFont(QFont("Courier New", 11))
        hint.setStyleSheet(
            "color:white;"
            "background:rgba(0,0,0,200);"
            "padding:6px 16px;"
            "border-radius:8px;"
        )
        hint.adjustSize()
        primary = QGuiApplication.primaryScreen().geometry()
        hint.move(primary.center().x() - hint.width() // 2, primary.top() + 32)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 160))

    def mousePressEvent(self, e):
        self.origin = e.pos()
        self.band.setGeometry(QRect(self.origin, QSize()))
        self.band.show()

    def mouseMoveEvent(self, e):
        self.band.setGeometry(QRect(self.origin, e.pos()).normalized())

    def mouseReleaseEvent(self, _e):
        sel = self.band.geometry()
        self.band.hide()
        self.close()
        if sel.width() > 4 and sel.height() > 4:
            logger.info("Region selected: %s", sel)
            self.region_selected.emit(sel)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()
