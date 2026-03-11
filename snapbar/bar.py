"""
Main floating toolbar (SnapBar).

Always-on-top, frameless, stealth (invisible to screen capture + cameras).
Feeds screenshots directly into AIPanel's image queue.
"""

import os

from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QFileDialog
from PyQt6.QtCore    import Qt, QTimer
from PyQt6.QtGui     import (
    QColor, QFont, QPainter, QPen, QBrush,
    QGuiApplication, QCursor,
)

from snapbar.core.constants  import PANEL_FLAGS, apply_stealth, btn_css
from snapbar.core.utils      import sep_v, ts
from snapbar.core.logging_config import get_logger

logger = get_logger("snapbar.bar")
from snapbar.panels.overlay  import SelectionOverlay
from snapbar.panels.ai_panel import AIPanel


class SnapBar(QWidget):

    def __init__(self):
        super().__init__()
        self.save_dir    = os.path.expanduser("~/Desktop")
        self._drag_pos   = None
        self._shot_count = 0
        self._multi_on   = False
        self._multi_secs = 3
        self._multi_cd   = 0

        self._flash = QTimer(self)
        self._flash.setSingleShot(True)
        self._flash.timeout.connect(self._clear_status)

        self._multi_timer = QTimer(self)
        self._multi_timer.timeout.connect(self._multi_tick)

        self._build_ui()
        self._ai = AIPanel(self)   # created after UI so self.geometry() is valid

    # ── stealth: once after hwnd exists ───────────────────────────
    def showEvent(self, e):
        super().showEvent(e)
        apply_stealth(self)

    # ── custom dark rounded background ────────────────────────────
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        p.setBrush(QBrush(QColor(14, 14, 22)))
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        p.drawRoundedRect(r, 14, 14)

    # ── drag ──────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, e):
        if self._drag_pos and e.buttons() == Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, _e):
        self._drag_pos = None

    # ── UI build ──────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowFlags(PANEL_FLAGS)
        # setWindowOpacity: compatible with SetWindowDisplayAffinity.
        # WA_TranslucentBackground is NOT used (causes UpdateLayeredWindowIndirect).
        self.setWindowOpacity(0.96)
        self.setFixedHeight(58)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(5)

        # drag handle
        h = QLabel("⠿")
        h.setFont(QFont("monospace", 14))
        h.setStyleSheet("color:rgba(255,255,255,60);padding:0 4px;")
        h.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        lay.addWidget(h)

        t = QLabel("SNAP")
        t.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        t.setStyleSheet("color:rgba(255,255,255,220);letter-spacing:3px;")
        lay.addWidget(t)
        lay.addSpacing(4)

        # screenshot buttons
        self.b_full   = self._mk("⬛  Full",   "#3B82F6", self._cap_full,   w=86)
        self.b_region = self._mk("✂  Region", "#8B5CF6", self._cap_region, w=90)
        self.b_window = self._mk("⬜  Window", "#10B981", self._cap_window, w=90)
        lay.addWidget(self.b_full)
        lay.addWidget(self.b_region)
        lay.addWidget(self.b_window)
        lay.addWidget(sep_v())

        # multi-shot
        self.b_multi = self._mk("⏱  Multi", "#F59E0B", self._toggle_multi, w=82)
        lay.addWidget(self.b_multi)
        self.b_minus = self._mk("−", "#94a3b8", self._dec_iv, w=24)
        self.iv_lbl  = QLabel(f"{self._multi_secs}s")
        self.iv_lbl.setFont(QFont("Courier New", 9, QFont.Weight.Bold))
        self.iv_lbl.setStyleSheet("color:rgba(255,255,255,180);min-width:20px;")
        self.iv_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.b_plus  = self._mk("+", "#94a3b8", self._inc_iv, w=24)
        lay.addWidget(self.b_minus)
        lay.addWidget(self.iv_lbl)
        lay.addWidget(self.b_plus)

        # shot counter badge
        self.badge = QLabel("×0")
        self.badge.setFont(QFont("Courier New", 9, QFont.Weight.Bold))
        self.badge.setStyleSheet(
            "color:#F59E0B;background:rgba(245,158,11,0.18);"
            "border:1px solid rgba(245,158,11,0.45);"
            "border-radius:5px;padding:1px 6px;")
        lay.addWidget(self.badge)
        lay.addWidget(sep_v())

        # AI panel controls
        self.b_ai   = self._mk("✦  Ask AI", "#6366F1", self._toggle_ai,  w=88)
        self.b_hide = self._mk("👁  Hide",   "#475569", self._hide_panel, w=80)
        lay.addWidget(self.b_ai)
        lay.addWidget(self.b_hide)
        lay.addWidget(sep_v())

        # utility
        self.b_dir   = self._mk("📁", "#6B7280", self._choose_dir, w=34)
        self.b_close = self._mk("✕",  "#EF4444", self.close,       w=34)
        lay.addWidget(self.b_dir)
        lay.addWidget(self.b_close)
        lay.addSpacing(4)

        # status label
        self.status = QLabel("")
        self.status.setFont(QFont("Courier New", 8))
        self.status.setMinimumWidth(110)
        self.status.setStyleSheet("color:rgba(255,255,255,180);")
        lay.addWidget(self.status)

        scr = QGuiApplication.primaryScreen().geometry()
        self.adjustSize()
        self.move(scr.center().x() - self.width() // 2, scr.bottom() - 94)
        self.show()

    def _mk(self, label: str, color: str, slot, w: int = 100) -> QPushButton:
        b = QPushButton(label)
        b.setFixedSize(w, 36)
        b.setFont(QFont("Courier New", 9, QFont.Weight.Bold))
        b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        b.setStyleSheet(btn_css(color))
        b.clicked.connect(slot)
        return b

    # ── status helpers ────────────────────────────────────────────
    def _set_status(self, text: str, color: str = "rgba(255,255,255,180)"):
        self.status.setText(text)
        self.status.setStyleSheet(
            f"color:{color};font-family:'Courier New';font-size:8pt;")
        self._flash.start(3000)

    def _clear_status(self):
        if not self._multi_on:
            self.status.setText("")

    # ── screenshot capture ────────────────────────────────────────
    def _save(self, px, tag: str = "shot"):
        """Save screenshot to disk and push into AI panel queue."""
        path = os.path.join(self.save_dir, f"{tag}_{ts()}.png")
        px.save(path, "PNG")
        logger.info("Screenshot saved: %s", path)
        self._shot_count += 1
        self.badge.setText(f"×{self._shot_count}")
        self._ai.add_image(px)
        self._set_status(f"✓ queued #{self._shot_count}", "#4ade80")

    def _cap_full(self):
        self.hide()
        QTimer.singleShot(220, self._do_full)

    def _do_full(self):
        self._save(QGuiApplication.primaryScreen().grabWindow(0), "full")
        self.show()

    def _cap_region(self):
        self.hide()
        QTimer.singleShot(180, self._open_overlay)

    def _open_overlay(self):
        self._ov = SelectionOverlay()
        self._ov.region_selected.connect(self._do_region)
        self._ov.destroyed.connect(self.show)
        self._ov.show()

    def _do_region(self, r):
        self._save(
            QGuiApplication.primaryScreen().grabWindow(
                0, r.x(), r.y(), r.width(), r.height()),
            "region")

    def _cap_window(self):
        self.hide()
        QTimer.singleShot(220, self._do_window)

    def _do_window(self):
        self._save(QGuiApplication.primaryScreen().grabWindow(0), "window")
        self.show()

    # ── multi-shot ────────────────────────────────────────────────
    def _toggle_multi(self):
        if self._multi_on:
            self._stop_multi()
        else:
            self._start_multi()

    def _start_multi(self):
        self._multi_on = True
        self._multi_cd = self._multi_secs
        self.b_multi.setStyleSheet(
            "QPushButton{background:#F59E0B;color:#000;"
            "border:1px solid #F59E0B;border-radius:9px;padding:2px 6px;}"
            "QPushButton:hover{background:#d97706;}")
        self.b_multi.setText("⏹  Stop")
        self._multi_timer.start(1000)
        self._do_multi()

    def _stop_multi(self):
        self._multi_on = False
        self._multi_timer.stop()
        self.b_multi.setText("⏱  Multi")
        self.b_multi.setStyleSheet(btn_css("#F59E0B"))
        self._set_status("Multi stopped", "#94a3b8")

    def _multi_tick(self):
        self._multi_cd -= 1
        if self._multi_cd <= 0:
            self._multi_cd = self._multi_secs
            self._do_multi()
        else:
            self._set_status(f"Next in {self._multi_cd}s", "#F59E0B")

    def _do_multi(self):
        self.hide()
        QTimer.singleShot(120, self._grab_multi)

    def _grab_multi(self):
        self._save(QGuiApplication.primaryScreen().grabWindow(0), "multi")
        self.show()

    def _inc_iv(self):
        self._multi_secs = min(60, self._multi_secs + 1)
        self.iv_lbl.setText(f"{self._multi_secs}s")

    def _dec_iv(self):
        self._multi_secs = max(1, self._multi_secs - 1)
        self.iv_lbl.setText(f"{self._multi_secs}s")

    # ── AI panel visibility ───────────────────────────────────────
    def _toggle_ai(self):
        """Open or close the AI panel."""
        self._ai.toggle()
        self._sync_hide_btn()

    def _hide_panel(self):
        """
        Hide the AI panel without destroying any state.
        Queue, transcript, and history are all preserved.
        Click again (now labelled 'Show') to restore.
        """
        if self._ai._open:
            self._ai.hide()
            self._ai._open = False
            self._set_status("Panel hidden — queue intact", "#94a3b8")
        else:
            self._ai.toggle()
        self._sync_hide_btn()

    def _sync_hide_btn(self):
        if self._ai._open:
            self.b_hide.setText("👁  Hide")
            self.b_hide.setStyleSheet(btn_css("#475569"))
        else:
            self.b_hide.setText("👁  Show")
            self.b_hide.setStyleSheet(btn_css("#6366F1"))

    # ── misc ──────────────────────────────────────────────────────
    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Save screenshots to…", self.save_dir)
        if d:
            self.save_dir = d
            self._set_status(f"→ {os.path.basename(d)}")

    def closeEvent(self, e):
        if hasattr(self, "_ai"):
            self._ai.close()
        super().closeEvent(e)
