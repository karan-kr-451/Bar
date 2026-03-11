"""Small reusable Qt helpers shared across modules."""

import base64
import os
import tempfile
from datetime import datetime

from PyQt6.QtWidgets import QFrame
from PyQt6.QtCore    import QBuffer, QIODevice
from PyQt6.QtGui     import QPixmap, QImage


def sep_v() -> QFrame:
    """Vertical separator line for the toolbar."""
    s = QFrame()
    s.setFrameShape(QFrame.Shape.VLine)
    s.setStyleSheet("color:rgba(255,255,255,25);")
    s.setFixedHeight(30)
    return s


def ts() -> str:
    """Timestamp string for filenames: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def qimage_to_b64(img: QImage, quality: int = 70, max_dim: int = 1600) -> str:
    """
    Encode QImage to base64 JPEG with compression and auto-resizing.
    JPEG is ~10x smaller than PNG in most cases, essential for Groq/OpenRouter.
    """
    # 1. Convert to RGB (JPEG requirement, avoids issues with transparency/alpha)
    if img.format() != QImage.Format.Format_RGB32:
        img = img.convertToFormat(QImage.Format.Format_RGB32)

    # 2. Auto-resize if too large (e.g. 4K -> 1600px width)
    if img.width() > max_dim or img.height() > max_dim:
        img = img.scaled(
            max_dim, max_dim,
            aspectRatioMode=os.sys.modules['PyQt6.QtCore'].Qt.AspectRatioMode.KeepAspectRatio,
            transformMode=os.sys.modules['PyQt6.QtCore'].Qt.TransformationMode.SmoothTransformation
        )

    # 3. Save to buffer as JPEG
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "JPG", quality=quality)
    return base64.b64encode(buf.data()).decode()


def pixmap_to_b64(px: QPixmap, quality: int = 70, max_dim: int = 1600) -> str:
    """Encode QPixmap to base64 JPEG via QBuffer."""
    return qimage_to_b64(px.toImage(), quality, max_dim)
