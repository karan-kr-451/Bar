"""
Shared constants, stealth helpers, button CSS, subject categories.

Stealth strategy (Windows)
--------------------------
WA_TranslucentBackground sets WS_EX_LAYERED (per-pixel alpha).
SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) is incompatible with
layered windows → UpdateLayeredWindowIndirect spam in console.

Fix: use setWindowOpacity() instead (LWA_ALPHA, not per-pixel layered).
Then apply DWM rounded corners (Win11) or setMask() (Win10).
Guard with _stealth_done so showEvent only runs this once.
"""

import sys
import ctypes
from PyQt6.QtCore import Qt
from snapbar.core.logging_config import get_logger

logger = get_logger("snapbar.stealth")

# ── Window flags (used by every top-level widget) ─────────────────
PANEL_FLAGS = (
    Qt.WindowType.FramelessWindowHint |
    Qt.WindowType.WindowStaysOnTopHint |
    Qt.WindowType.Tool
)

# ── Win32 constants ───────────────────────────────────────────────
WDA_EXCLUDEFROMCAPTURE         = 0x00000011  # hide from all capture + camera reflections
WS_EX_TOOLWINDOW               = 0x00000080  # remove from taskbar / Alt+Tab
GWL_EXSTYLE                    = -20
DWMWA_WINDOW_CORNER_PREFERENCE = 33          # Win11 DWM rounded corners
DWMWCP_ROUND                   = 2


def apply_stealth(widget):
    """
    Make widget invisible to screen capture and camera reflections.
    Call inside showEvent(); the _stealth_done guard ensures it runs once.
    Windows 10 build 2004+ / Windows 11 only. No-op on Linux/macOS.
    """
    if sys.platform != "win32":
        return
    if getattr(widget, "_stealth_done", False):
        return
    widget._stealth_done = True

    try:
        u32  = ctypes.windll.user32
        dwm  = ctypes.windll.dwmapi
        hwnd = ctypes.c_void_p(int(widget.winId()))

        # 1. Exclude from all capture (OBS, PrintScreen, Zoom share, camera reflection)
        ok = u32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        if not ok:
            logger.error("SetWindowDisplayAffinity failed: %s", ctypes.get_last_error())
            return

        # 2. Force WS_EX_TOOLWINDOW — no taskbar button, no Alt+Tab entry
        ex = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        u32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_TOOLWINDOW)

        # 3. Rounded corners
        import platform
        build = int(platform.version().split(".")[-1])
        if build >= 22000:
            # Win11: hardware-composited DWM corners
            pref = ctypes.c_int(DWMWCP_ROUND)
            dwm.DwmSetWindowAttribute(
                hwnd,
                DWMWA_WINDOW_CORNER_PREFERENCE,
                ctypes.byref(pref),
                ctypes.sizeof(pref),
            )
        else:
            # Win10: software clip via QRegion mask
            _apply_rounded_mask(widget, radius=14)

        logger.info("hwnd=%d build=%d applied successfully", int(hwnd.value), build)

    except Exception as e:
        logger.exception("Stealth application error: %s", e)


def _apply_rounded_mask(widget, radius: int = 14):
    """Win10 fallback: clips widget to a rounded-rect using QRegion."""
    from PyQt6.QtGui  import QRegion
    from PyQt6.QtCore import QRect
    r   = widget.rect()
    rgn = QRegion(r, QRegion.RegionType.Rectangle)
    d   = radius * 2
    for corner_rect in [
        QRect(0,              0,               radius, radius),
        QRect(r.width()-radius, 0,             radius, radius),
        QRect(0,              r.height()-radius, radius, radius),
        QRect(r.width()-radius, r.height()-radius, radius, radius),
    ]:
        # ellipse that fills the full corner square
        ex = corner_rect.x() if corner_rect.x() == 0 else corner_rect.x() - radius
        ey = corner_rect.y() if corner_rect.y() == 0 else corner_rect.y() - radius
        ellipse = QRegion(QRect(ex, ey, d, d), QRegion.RegionType.Ellipse)
        rgn = rgn.subtracted(QRegion(corner_rect).subtracted(ellipse))
    widget.setMask(rgn)


# ── Button stylesheet helper ──────────────────────────────────────
def btn_css(color: str) -> str:
    return (
        f"QPushButton{{background:rgba(255,255,255,15);color:white;"
        f"border:1px solid rgba(255,255,255,25);border-radius:9px;padding:2px 6px;}}"
        f"QPushButton:hover{{background:{color};border:1px solid {color};}}"
        f"QPushButton:pressed{{background:{color}bb;}}"
    )


# ── Subject categories → system prompts ──────────────────────────
CATEGORIES: dict[str, str] = {
    "🧠 Auto": "",
    "💻 Coding": (
        "You are an expert competitive-programmer.\n"
        "1. State problem understanding + constraints.\n"
        "2. Choose optimal algorithm — justify time/space complexity.\n"
        "3. Write complete, runnable code (Python default unless specified).\n"
        "4. Handle all edge cases including large inputs.\n"
        "5. Walk through one example step-by-step."
    ),
    "🗄️ DB": (
        "You are a senior database engineer (SQL + NoSQL).\n"
        "1. Write the correct, optimised query.\n"
        "2. Explain each clause.\n"
        "3. Suggest indexes or schema changes if relevant.\n"
        "4. Show sample input/output."
    ),
    "🔢 Math": (
        "You are a mathematics tutor.\n"
        "1. Identify the concept/theorem.\n"
        "2. Solve step-by-step showing all working.\n"
        "3. Box the final answer.\n"
        "4. Note common mistakes."
    ),
    "🧩 Aptitude": (
        "You are an aptitude coach for competitive exams.\n"
        "1. Identify question type.\n"
        "2. Apply the fastest shortcut method.\n"
        "3. Show full working + answer.\n"
        "4. Give a one-line tip for similar questions."
    ),
    "📐 Reasoning": (
        "You are a logical/analytical reasoning expert.\n"
        "1. Break down premises and clues.\n"
        "2. Apply deductive/inductive reasoning clearly.\n"
        "3. State the definitive answer with justification.\n"
        "4. Flag any traps."
    ),
    "📝 English": (
        "You are an English language expert.\n"
        "1. Identify the concept tested (tense, preposition, synonym, etc.).\n"
        "2. Give the correct answer.\n"
        "3. Explain the rule simply.\n"
        "4. Give one reinforcing example."
    ),
    "📊 Data Sci": (
        "You are a data scientist and ML engineer.\n"
        "1. Identify the task (EDA, model selection, stats, etc.).\n"
        "2. Provide complete runnable Python/R code.\n"
        "3. Explain methodology and assumptions.\n"
        "4. Suggest alternatives if relevant."
    ),
    "🌐 General": (
        "Answer clearly and concisely. Use headings for multi-part answers. No filler."
    ),
}
