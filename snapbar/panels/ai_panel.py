"""
AI Solver panel.

Displays above the bar. Holds:
- Provider / model / API key selectors
- Audio capture (sounddevice → Google STT)
- Screenshot queue (fed by SnapBar._save)
- Subject category pills (system prompt selector)
- Multi-turn chat with streaming response

Fixes / features
────────────────
1. Vision + Reasoning model auto-select
   Two-pass scan when images are queued:
     Pass 1 — prefer vision + reasoning (gemini-3.x, 2.5+, claude-3-7, o3/o4...)
     Pass 2 — fall back to vision-only model.
   Status banner: "🧠✦ vision + reasoning" or "✦ vision".

2. Groq multi-turn 400 fix
   _build_api_history() collapses prior list-content to plain text for
   providers (groq, openrouter) that reject array content in history.

3. 429 / quota auto-fallback
   On any quota/rate-limit error, _quota_fallback() advances the model
   combo to the next entry and retries automatically.
   "⚡ Quota on X — retrying with Y..." until the list is exhausted.

4. Claude 4.x vision keyword fix
   _VISION_KEYWORDS uses "claude-" (matches claude-opus-4-6,
   claude-sonnet-4-6, etc.) instead of just "claude-3".
"""

import os
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit,
    QComboBox, QButtonGroup, QRadioButton,
    QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui  import (
    QColor, QFont, QPixmap,
    QPainter, QPen, QBrush,
    QGuiApplication, QCursor, QTextCursor,
)

from snapbar.core.constants  import PANEL_FLAGS, CATEGORIES, apply_stealth, btn_css
from snapbar.core.utils      import sep_v, ts
from snapbar.core.logging_config import get_logger

logger = get_logger("snapbar.ui")


def safe_slot(func):
    """
    Wrap every PyQt slot so exceptions hit the log instead of being
    swallowed silently at the C++ signal/slot boundary.
    """
    import functools, traceback, logging
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logging.getLogger("snapbar.ui").critical(
                "SLOT CRASH in %s.%s:\n%s",
                type(args[0]).__name__ if args else "?",
                func.__name__,
                traceback.format_exc(),
            )
    return wrapper

from snapbar.workers.transcriber import AudioSignals, Transcriber
from snapbar.workers.ai_worker   import AIWorker, PROVIDER_MODELS


# ── model capability detection ────────────────────────────────────────────────

# Vision: any model whose name contains one of these (case-insensitive)
_VISION_KEYWORDS = (
    "vision", "vl", "llava", "pixtral",
    "gemini",           # gemini-2.x, gemini-3.x (all multimodal)
    "gpt-4o",
    # Anthropic — ALL Claude models support vision; match any generation
    # (claude-3-*, claude-opus-4-6, claude-sonnet-4-6, claude-haiku-*, ...)
    "claude-",
    "llama-3.2", "llama4", "llama-4", "scout", "maverick",
)

# Reasoning: models with extended chain-of-thought / thinking capability.
_REASONING_KEYWORDS = (
    # Google — 2.5+ have built-in thinking
    "2.5-pro", "2.5-flash", "2.5-flash-lite",
    "gemini-3",         # all gemini-3.x have thinking
    # Anthropic — extended thinking
    "claude-3-7",       # claude-3-7-sonnet
    "claude-3-5-sonnet",
    "claude-opus-4",    # claude-opus-4-6 etc.
    "claude-sonnet-4",  # claude-sonnet-4-6 etc.
    # OpenAI / compatible
    "o1", "o3", "o4",
    # DeepSeek / open-source
    "r1", "r2",
    "thinking",         # explicit tag (qwq, gemma thinking variants)
    "reason",
)

def _is_vision_model(model_name: str) -> bool:
    """Return True if *model_name* is likely to support image input."""
    lower = model_name.lower()
    return any(kw in lower for kw in _VISION_KEYWORDS)

def _is_reasoning_model(model_name: str) -> bool:
    """Return True if *model_name* supports chain-of-thought / thinking."""
    lower = model_name.lower()
    return any(kw in lower for kw in _REASONING_KEYWORDS)

def _model_tier(model_name: str) -> int:
    """
    Priority score for auto-selection when images are queued.
      2 — vision + reasoning  (best for screenshot problem-solving)
      1 — vision only
      0 — neither
    """
    v = _is_vision_model(model_name)
    r = _is_reasoning_model(model_name)
    return 2 if (v and r) else (1 if v else 0)


class AIPanel(QWidget):

    def __init__(self, bar: QWidget):
        super().__init__()
        self.bar         = bar
        self._open       = False
        self._images:    list[QPixmap] = []
        self._transcript = ""
        self._history:   list[dict]   = []
        self._worker:    AIWorker | None = None
        # Strong-ref list: prevents GC of worker/signals between
        # sig.done.emit() in the thread and delivery on the main thread.
        self._workers:   list = []
        self._ai_resp    = ""
        self._recording  = False
        self._pending_send: dict | None = None   # saved args for 429 auto-retry

        # Chunk buffering to prevent UI thread saturation (prevents "UI death")
        self._chunk_buffer = ""
        self._update_timer = QTimer()
        self._update_timer.setInterval(30)  # Update UI every 30ms
        self._update_timer.timeout.connect(self._flush_chunks)

        # Audio
        self._asig = AudioSignals()
        self._asig.transcript.connect(self._on_audio_text)
        self._asig.error.connect(lambda m: self._status(f"Audio: {m}", "#f87171"))
        self._asig.level.connect(self._on_level)
        self._transcriber  = Transcriber(self._asig)
        self._chunk_timer  = QTimer(self)
        self._chunk_timer.timeout.connect(self._transcriber.flush)

        self.setWindowFlags(PANEL_FLAGS)
        self.setWindowOpacity(0.97)
        self.setFixedSize(820, 680)
        self._build()
        self.hide()

    # ── stealth ───────────────────────────────────────────────────
    def showEvent(self, e):
        super().showEvent(e)
        apply_stealth(self)

    # ── background ────────────────────────────────────────────────
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        p.setBrush(QBrush(QColor(10, 10, 20)))
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        p.drawRoundedRect(r, 14, 14)

    # ── UI build ──────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(9)

        # ── row 1: title + new-chat ───────────────────────────────
        r1 = QHBoxLayout()
        ttl = QLabel("✦  AI SOLVER")
        ttl.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        ttl.setStyleSheet("color:rgba(255,255,255,220);letter-spacing:3px;")
        r1.addWidget(ttl)
        r1.addStretch()
        nc = QPushButton("New chat")
        nc.setFixedSize(72, 26)
        nc.setFont(QFont("Courier New", 8))
        nc.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        nc.setStyleSheet(
            "QPushButton{background:rgba(255,255,255,12);color:rgba(255,255,255,170);"
            "border:1px solid rgba(255,255,255,22);border-radius:6px;}"
            "QPushButton:hover{background:rgba(255,255,255,28);color:white;}")
        nc.clicked.connect(self._new_chat)
        r1.addWidget(nc)
        root.addLayout(r1)

        # ── row 2: provider + model + key ─────────────────────────
        r2 = QHBoxLayout()
        r2.setSpacing(6)

        self._env_keys = {
            "anthropic":  os.environ.get("ANTHROPIC_API_KEY",  ""),
            "google":     os.environ.get("GOOGLE_API_KEY",     ""),
            "groq":       os.environ.get("GROQ_API_KEY",       ""),
            "openrouter": os.environ.get("OPENROUTER_API_KEY", ""),
        }

        def _lbl(t):
            l = QLabel(t)
            l.setFont(QFont("Courier New", 8))
            l.setStyleSheet("color:rgba(255,255,255,140);")
            return l

        r2.addWidget(_lbl("Provider:"))
        self.prov_combo = QComboBox()
        self.prov_combo.setFixedSize(110, 26)
        self.prov_combo.setFont(QFont("Courier New", 8))
        self._cs(self.prov_combo)
        for p in PROVIDER_MODELS:
            self.prov_combo.addItem(p)
        self.prov_combo.currentTextChanged.connect(self._on_provider_change)
        r2.addWidget(self.prov_combo)

        r2.addWidget(_lbl("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setFixedHeight(26)
        self.model_combo.setMinimumWidth(240)
        self.model_combo.setFont(QFont("Courier New", 8))
        self._cs(self.model_combo)
        r2.addWidget(self.model_combo, 1)

        r2.addWidget(_lbl("Key:"))
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("API key…")
        self.key_edit.setFont(QFont("Courier New", 8))
        self.key_edit.setFixedHeight(26)
        self.key_edit.setMinimumWidth(180)
        self.key_edit.setStyleSheet(
            "QLineEdit{background:rgba(255,255,255,12);color:white;"
            "border:1px solid rgba(255,255,255,25);border-radius:6px;padding:2px 8px;}"
            "QLineEdit:focus{border:1px solid #6366F1;}")
        r2.addWidget(self.key_edit, 1)

        root.addLayout(r2)
        self._on_provider_change(self.prov_combo.currentText())

        # ── row 3: audio device + rec + VU ───────────────────────
        r3 = QHBoxLayout()
        r3.setSpacing(6)
        r3.addWidget(QLabel("🎙"))

        self.dev_combo = QComboBox()
        self.dev_combo.setFont(QFont("Courier New", 8))
        self.dev_combo.setFixedHeight(26)
        self.dev_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._cs(self.dev_combo)
        r3.addWidget(self.dev_combo, 1)

        ref = QPushButton("↻")
        ref.setFixedSize(26, 26)
        ref.setFont(QFont("Courier New", 10))
        ref.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        ref.setStyleSheet(
            "QPushButton{background:rgba(255,255,255,12);color:white;"
            "border:1px solid rgba(255,255,255,22);border-radius:6px;}"
            "QPushButton:hover{background:rgba(255,255,255,28);}")
        ref.clicked.connect(self._refresh_devices)
        r3.addWidget(ref)

        self.btn_rec = QPushButton("● REC")
        self.btn_rec.setFixedSize(72, 26)
        self.btn_rec.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        self.btn_rec.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_rec.setStyleSheet(btn_css("#EF4444"))
        self.btn_rec.clicked.connect(self._toggle_rec)
        r3.addWidget(self.btn_rec)

        self.vu = QLabel()
        self.vu.setFixedSize(120, 10)
        self.vu.setStyleSheet("background:rgba(255,255,255,18);border-radius:5px;")
        r3.addWidget(self.vu)

        root.addLayout(r3)
        self._refresh_devices()

        # ── category pills (horizontal scroll) ───────────────────
        cat_scroll = QScrollArea()
        cat_scroll.setFixedHeight(42)
        cat_scroll.setWidgetResizable(True)
        cat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        cat_scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
            "QScrollBar:horizontal{height:4px;background:transparent;}"
            "QScrollBar::handle:horizontal{background:rgba(255,255,255,30);"
            "border-radius:2px;}")
        cat_inner = QWidget()
        cat_inner.setStyleSheet("background:transparent;")
        cat_lay = QHBoxLayout(cat_inner)
        cat_lay.setContentsMargins(0, 4, 0, 4)
        cat_lay.setSpacing(4)

        self._cat_group = QButtonGroup(self)
        self._cat_group.setExclusive(True)
        for i, cat in enumerate(CATEGORIES.keys()):
            btn = QPushButton(cat)
            btn.setCheckable(True)
            btn.setFont(QFont("Courier New", 8))
            btn.setFixedHeight(28)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet(
                "QPushButton{background:rgba(255,255,255,10);color:rgba(255,255,255,150);"
                "border:1px solid rgba(255,255,255,20);border-radius:7px;padding:2px 10px;}"
                "QPushButton:checked{background:#6366F1;color:white;"
                "border:1px solid #6366F1;}"
                "QPushButton:hover{background:rgba(255,255,255,22);color:white;}")
            self._cat_group.addButton(btn, i)
            cat_lay.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
        cat_lay.addStretch()
        cat_scroll.setWidget(cat_inner)
        root.addWidget(cat_scroll)

        # ── mode selector + badge + clear ─────────────────────────
        r_mode = QHBoxLayout()
        r_mode.setSpacing(6)
        lbl_mode = QLabel("Mode:")
        lbl_mode.setFont(QFont("Courier New", 8))
        lbl_mode.setStyleSheet("color:rgba(255,255,255,140);")
        r_mode.addWidget(lbl_mode)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for lbl, val in (("🖼 Images", "images"), ("🎙 Audio", "audio"), ("✦ Both", "both")):
            r_mode.addWidget(self._mk_mode(lbl, val))
        self._mode_group.buttons()[0].setChecked(True)
        # re-check vision model whenever mode changes
        self._mode_group.buttonClicked.connect(self._on_mode_changed)

        r_mode.addStretch()

        self.badge = QLabel("0 imgs")
        self.badge.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        self.badge.setStyleSheet(
            "color:#6366F1;background:rgba(99,102,241,0.15);"
            "border:1px solid rgba(99,102,241,0.4);"
            "border-radius:5px;padding:1px 8px;")
        r_mode.addWidget(self.badge)

        b_clr = QPushButton("Clear queue")
        b_clr.setFixedSize(88, 24)
        b_clr.setFont(QFont("Courier New", 8))
        b_clr.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        b_clr.setStyleSheet(
            "QPushButton{background:rgba(255,255,255,10);color:rgba(255,255,255,130);"
            "border:1px solid rgba(255,255,255,22);border-radius:6px;}"
            "QPushButton:hover{background:rgba(239,68,68,0.25);color:#f87171;"
            "border:1px solid rgba(239,68,68,0.5);}")
        b_clr.clicked.connect(self._clear_queue)
        r_mode.addWidget(b_clr)
        root.addLayout(r_mode)

        # ── chat display ──────────────────────────────────────────
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setFont(QFont("Courier New", 9))
        self.chat.setStyleSheet(
            "QTextEdit{background:rgba(255,255,255,6);color:rgba(255,255,255,200);"
            "border:1px solid rgba(255,255,255,14);border-radius:8px;padding:8px;}"
            "QScrollBar:vertical{width:6px;background:transparent;}"
            "QScrollBar::handle:vertical{background:rgba(255,255,255,30);border-radius:3px;}")
        root.addWidget(self.chat, 1)

        # ── transcript preview ────────────────────────────────────
        self.trans_preview = QLabel("🎙 No transcript yet")
        self.trans_preview.setFont(QFont("Courier New", 8))
        self.trans_preview.setStyleSheet(
            "color:rgba(255,255,255,100);background:rgba(255,255,255,6);"
            "border:1px solid rgba(255,255,255,14);border-radius:6px;padding:4px 8px;")
        self.trans_preview.setWordWrap(True)
        self.trans_preview.setFixedHeight(36)
        root.addWidget(self.trans_preview)

        # ── prompt + send ─────────────────────────────────────────
        r_p = QHBoxLayout()
        r_p.setSpacing(6)
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Extra instruction (optional)…")
        self.prompt.setFont(QFont("Courier New", 9))
        self.prompt.setFixedHeight(58)
        self.prompt.setStyleSheet(
            "QTextEdit{background:rgba(255,255,255,9);color:white;"
            "border:1px solid rgba(255,255,255,22);border-radius:8px;padding:6px 8px;}"
            "QTextEdit:focus{border:1px solid #6366F1;}")
        r_p.addWidget(self.prompt, 1)

        self.btn_send = QPushButton("Send ▶")
        self.btn_send.setFixedSize(88, 58)
        self.btn_send.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        self.btn_send.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_send.setStyleSheet(btn_css("#6366F1"))
        self.btn_send.clicked.connect(self._send)
        r_p.addWidget(self.btn_send)
        root.addLayout(r_p)

        # ── status ────────────────────────────────────────────────
        self.status_lbl = QLabel("")
        self.status_lbl.setFont(QFont("Courier New", 8))
        self.status_lbl.setStyleSheet("color:rgba(255,255,255,110);")
        root.addWidget(self.status_lbl)

    # ── combo stylesheet ──────────────────────────────────────────
    def _cs(self, w):
        w.setStyleSheet(
            "QComboBox{background:rgba(255,255,255,12);color:white;"
            "border:1px solid rgba(255,255,255,25);border-radius:6px;padding:2px 8px;}"
            "QComboBox:focus{border:1px solid #6366F1;}"
            "QComboBox QAbstractItemView{background:#1e1e2e;color:white;"
            "border:1px solid rgba(255,255,255,25);"
            "selection-background-color:#6366F1;}"
            "QComboBox::drop-down{border:none;width:18px;}"
            "QComboBox::down-arrow{width:8px;height:8px;}")

    # ── mode radio button ─────────────────────────────────────────
    def _mk_mode(self, label: str, val: str) -> QRadioButton:
        r = QRadioButton(label)
        r.setFont(QFont("Courier New", 8))
        r.setStyleSheet(
            "QRadioButton{color:rgba(255,255,255,160);spacing:4px;}"
            "QRadioButton::indicator{width:12px;height:12px;border-radius:6px;"
            "border:1px solid rgba(255,255,255,40);background:rgba(255,255,255,10);}"
            "QRadioButton::indicator:checked{background:#6366F1;"
            "border:1px solid #6366F1;}")
        r.setProperty("mv", val)
        self._mode_group.addButton(r)
        return r

    def _get_mode(self) -> str:
        b = self._mode_group.checkedButton()
        return b.property("mv") if b else "both"

    def _update_badge(self):
        mode = self._get_mode()
        ni, nc = len(self._images), len(self._transcript)
        self.badge.setText(
            f"{ni} img{'s' if ni != 1 else ''}" if mode == "images" else
            f"{nc} chars"                        if mode == "audio"  else
            f"{ni} imgs · {nc} chars")

    def _status(self, msg: str, color: str = "rgba(255,255,255,110)"):
        self.status_lbl.setText(msg)
        self.status_lbl.setStyleSheet(
            f"color:{color};font-family:'Courier New';font-size:8pt;")

    # ── provider / model ──────────────────────────────────────────
    def _on_provider_change(self, provider: str):
        self.model_combo.clear()
        for m in PROVIDER_MODELS.get(provider, []):
            self.model_combo.addItem(m)
        self.key_edit.setText(self._env_keys.get(provider, ""))
        # After repopulating models, re-apply vision auto-select if needed
        if self._images and self._get_mode() in ("images", "both"):
            self._auto_select_vision_model(silent=True)

    def _on_mode_changed(self, _btn):
        """When the user switches to a mode that includes images, ensure a vision model."""
        if self._images and self._get_mode() in ("images", "both"):
            self._auto_select_vision_model()

    # ── vision + reasoning model auto-select ─────────────────────
    def _auto_select_vision_model(self, *, silent: bool = False) -> bool:
        """
        Ensure the active model can handle images.  When images are queued
        we also prefer a model that can *reason* (extended thinking / CoT),
        since screenshot problem-solving benefits most from it.

        Selection order (two-pass):
          Pass 1 — first model with tier == 2  (vision AND reasoning)
          Pass 2 — first model with tier == 1  (vision only)
          Fail   — no vision model → warn and return False

        Returns True if the current or newly selected model is vision-capable.
        """
        current      = self.model_combo.currentText()
        current_tier = _model_tier(current)

        # Already vision+reasoning — nothing to do
        if current_tier == 2:
            return True

        # Collect all (index, tier) for models with tier >= 1
        candidates: list[tuple[int, int]] = []
        for i in range(self.model_combo.count()):
            t = _model_tier(self.model_combo.itemText(i))
            if t >= 1:
                candidates.append((i, t))

        if not candidates:
            if not silent:
                logger.warning("No vision model found for provider %s",
                               self.prov_combo.currentText())
                self._status(
                    "⚠ No vision model available for this provider — "
                    "switch provider or select manually", "#f87171")
            return False

        # Already vision-only and no reasoning upgrade exists → keep it
        best_idx, best_tier = max(candidates, key=lambda x: x[1])
        if current_tier == 1 and best_tier == 1:
            return True

        # Switch to the best available model
        best_name = self.model_combo.itemText(best_idx)
        self.model_combo.setCurrentIndex(best_idx)

        if not silent:
            if best_tier == 2:
                label = f"🧠✦ vision + reasoning: {best_name.split('/')[-1]}"
                color = "#a78bfa"
            else:
                label = f"✦ vision: {best_name.split('/')[-1]}"
                color = "#818cf8"
            logger.info("Auto-selected model (tier %d): %s → %s",
                        best_tier, current, best_name)
            self._status(f"Auto-switched to {label}", color)

        return True

    # ── audio ─────────────────────────────────────────────────────
    def _refresh_devices(self):
        self.dev_combo.clear()
        for lbl in self._transcriber.list_devices():
            self.dev_combo.addItem(lbl)

    def _toggle_rec(self):
        if self._recording:
            self._stop_rec()
        else:
            self._start_rec()

    def _start_rec(self):
        ok, err = self._transcriber.start(
            self.dev_combo.currentIndex(), 5, self._chunk_timer)
        if not ok:
            logger.error("Failed to start recording: %s", err)
            self._status(f"⚠ {err}", "#f87171")
            return
        logger.info("Recording started on device index %d", self.dev_combo.currentIndex())
        self._recording = True
        self.btn_rec.setText("■ Stop")
        self.btn_rec.setStyleSheet(
            "QPushButton{background:#EF4444;color:white;border:none;"
            "border-radius:9px;padding:2px 6px;}"
            "QPushButton:hover{background:#dc2626;}")
        self._status("Recording…", "#f87171")

    def _stop_rec(self):
        self._transcriber.stop(self._chunk_timer)
        logger.info("Recording stopped")
        self._recording = False
        self.btn_rec.setText("● REC")
        self.btn_rec.setStyleSheet(btn_css("#EF4444"))
        self._status("Stopped")

    @safe_slot
    def _on_audio_text(self, text: str):
        ts_str = datetime.now().strftime("%H:%M:%S")
        self._transcript += f"[{ts_str}] {text}\n"
        preview = self._transcript[-120:].replace("\n", " ")
        self.trans_preview.setText(f"🎙 {preview}")
        self._update_badge()

    @safe_slot
    def _on_level(self, level: float):
        color = "#4ade80" if level < 0.6 else ("#facc15" if level < 0.85 else "#EF4444")
        self.vu.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {color},stop:{max(0.01, level)} {color},"
            f"stop:{min(1.0, level + 0.01)} rgba(255,255,255,12),"
            f"stop:1 rgba(255,255,255,12));border-radius:5px;")

    # ── image queue ───────────────────────────────────────────────
    def add_image(self, px: QPixmap):
        self._images.append(px)
        self._update_badge()
        # Auto-select a vision model whenever an image arrives and the
        # current mode involves images.
        if self._get_mode() in ("images", "both"):
            self._auto_select_vision_model()

    def _clear_queue(self):
        self._images.clear()
        self._transcript = ""
        self.trans_preview.setText("🎙 No transcript yet")
        self._update_badge()
        logger.info("Queue cleared")
        self._status("Queue cleared")

    # ── history sanitization ──────────────────────────────────────
    def _build_api_history(self, provider: str) -> list[dict]:
        """
        Return a copy of _history suitable for *provider*.

        Problem (Groq + others):
          The OpenAI messages spec allows content to be either a string or a
          list[dict] (for multi-modal turns).  Several providers — including
          Groq — accept the list form *only* for the current user message
          where actual image data is present.  Historical messages (prior
          turns) that carry a list cause a 400: "content must be a string".

        Fix:
          For every message that is NOT the final user entry, collapse any
          list content to a plain-text string by joining the "text" parts.
          The final user message is left as-is so the AIWorker can splice in
          the base-64 encoded images.
        """
        if not self._history:
            return []

        # Providers confirmed to need this normalisation.
        # Add others here if they show the same 400 pattern.
        NEEDS_FLAT_PRIOR: set[str] = {"groq", "openrouter"}

        if provider not in NEEDS_FLAT_PRIOR:
            return list(self._history)

        result: list[dict] = []
        last_idx = len(self._history) - 1

        for i, msg in enumerate(self._history):
            content = msg["content"]
            is_last_user = (i == last_idx and msg["role"] == "user")

            if not is_last_user and isinstance(content, list):
                # Collapse to plain text — images from prior turns are not
                # re-sent, so this loses nothing the API would use.
                content = "\n".join(
                    c.get("text", "")
                    for c in content
                    if c.get("type") == "text"
                ).strip()

            result.append({"role": msg["role"], "content": content})

        return result

    # ── send to AI ────────────────────────────────────────────────
    def _send(self):
        provider = self.prov_combo.currentText()
        model    = self.model_combo.currentText()
        api_key  = self.key_edit.text().strip()

        if not api_key:
            self._status("⚠ Enter API key first", "#f87171"); return
        if not model:
            self._status("⚠ Select a model",      "#f87171"); return

        mode  = self._get_mode()
        extra = self.prompt.toPlainText().strip()

        if mode in ("images", "both") and not self._images:
            self._status("⚠ No screenshots queued", "#f87171"); return
        if mode in ("audio", "both") and not self._transcript.strip():
            self._status("⚠ No transcript yet",    "#f87171"); return

        # Guard: if images are involved, make sure we have a vision model
        if mode in ("images", "both") and _model_tier(model) == 0:
            ok = self._auto_select_vision_model()
            if not ok:
                return  # status already set inside _auto_select_vision_model
            model = self.model_combo.currentText()

        cat_id   = self._cat_group.checkedId()
        cat_name = list(CATEGORIES.keys())[cat_id]
        sys_p    = list(CATEGORIES.values())[cat_id]

        # ── content list (images encoded by worker) ───────────────
        content: list[dict] = []
        if mode in ("images", "both"):
            content.append({"type": "text",
                             "text": f"[{len(self._images)} screenshot(s) above]\n"})
        if mode in ("audio", "both"):
            content.append({"type": "text",
                             "text": f"[Audio transcript]\n{self._transcript}\n"})
        content.append({"type": "text", "text": extra or (
            "Solve the problem in the screenshot(s)."        if mode == "images" else
            "Solve / answer the question in the transcript." if mode == "audio"  else
            "Analyse screenshots + transcript and solve.")})

        self._history.append({"role": "user", "content": content})

        short = model.split("/")[-1]
        tag   = {"images": "🖼", "audio": "🎙", "both": "✦"}[mode]
        logger.info("Sending request: provider=%s, model=%s, mode=%s, imgs=%d",
                    provider, model, mode, len(self._images))
        self.chat.append(
            f'<p style="color:#475569;font-size:8pt;margin:6px 0 2px;">'
            f'— You [{tag}] [{cat_name}] [{provider}/{short}] —</p>'
            f'<p style="color:rgba(255,255,255,180);margin:0 0 8px;">'
            f'{extra or "(auto-solve)"}</p>')

        self.btn_send.setEnabled(False)
        self.btn_send.setText("…")
        self._status(f"Asking {provider}/{short}…", "#818cf8")

        self._ai_resp = ""
        qimgs = [px.toImage() for px in self._images] if mode in ("images", "both") else []

        # Save full request state so _quota_fallback() can retry with next model
        self._pending_send = dict(
            provider=provider, api_key=api_key, sys_p=sys_p,
            qimgs=qimgs, mode=mode, extra=extra, cat_name=cat_name,
        )

        # Use sanitised history — collapses prior list-content for providers
        # that require string content in historical messages (e.g. Groq).
        api_history = self._build_api_history(provider)

        self._worker = AIWorker(provider, api_key, model, sys_p, api_history, qimgs)
        # QueuedConnection: guaranteed cross-thread delivery even if the
        # sending QObject is GC'd between emit() and event processing.
        _QC = Qt.ConnectionType.QueuedConnection
        self._worker.sig.chunk.connect(self._on_chunk,    _QC)
        self._worker.sig.done.connect(self._on_done,      _QC)
        self._worker.sig.error.connect(self._on_ai_error, _QC)
        # Keepalive: strong ref until OS thread actually exits
        self._workers.append(self._worker)
        self._worker.finished.connect(
            lambda w=self._worker: self._workers.remove(w) if w in self._workers else None
        )
        self.chat.append(
            f'<p style="color:#a78bfa;font-size:8pt;margin:4px 0 2px;">'
            f'— {provider}/{short} —</p>')
        self._worker.start()

    @safe_slot
    def _on_chunk(self, text: str):
        # Buffer the text instead of updating UI immediately
        self._chunk_buffer += text
        self._ai_resp      += text
        if not self._update_timer.isActive():
            self._update_timer.start()

    @safe_slot
    def _flush_chunks(self):
        """Batched UI update for AI streaming tokens."""
        if not self._chunk_buffer:
            self._update_timer.stop()
            return
            
        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(self._chunk_buffer)
        self.chat.setTextCursor(cursor)
        self.chat.ensureCursorVisible()
        self.chat.viewport().update()
        
        self._chunk_buffer = ""

    @safe_slot
    def _on_done(self):
        # Flush any remaining chunks
        self._flush_chunks()
        self._update_timer.stop()

        # Store assistant reply as plain string
        self._history.append({"role": "assistant", "content": self._ai_resp})
        self._pending_send = None
        self.chat.append("")
        self.btn_send.setEnabled(True)
        self.btn_send.setText("Send ▶")
        self.prompt.clear()
        self._status("✓ Done — queue preserved for follow-ups", "#4ade80")
        logger.info("AI Request completed: %d total chars", len(self._ai_resp))

    @safe_slot
    def _on_ai_error(self, msg: str):
        # Roll back the optimistically-appended user message so a retry
        # doesn't double-up the history.
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()

        # 429 / quota / rate-limit → try the next model automatically
        is_quota = any(k in msg for k in (
            "429", "quota", "ResourceExhausted",
            "rate_limit", "rate limit", "RateLimitError", "RESOURCE_EXHAUSTED",
        ))
        if is_quota and self._pending_send:
            if self._quota_fallback():
                return   # fallback launched a retry — suppress the error display

        self._pending_send = None
        self.chat.append(f'<p style="color:#f87171;">⚠ {msg}</p>')
        self.btn_send.setEnabled(True)
        self.btn_send.setText("Send ▶")
        self._status(f"⚠ {msg[:90]}", "#f87171")

    def _quota_fallback(self) -> bool:
        """
        Advance the model combo past the exhausted model and retry.
        Skips non-vision models when the current mode requires images.
        Returns True if a retry was launched, False if the list is exhausted.
        """
        ps        = self._pending_send
        failed_idx = self.model_combo.currentIndex()
        n          = self.model_combo.count()
        mode       = ps["mode"]

        next_idx = failed_idx + 1
        while next_idx < n:
            if mode in ("images", "both") and _model_tier(
                    self.model_combo.itemText(next_idx)) == 0:
                next_idx += 1
                continue
            break
        else:
            logger.warning("Quota fallback: all models exhausted for %s", ps["provider"])
            self._status("⚠ All models quota-exhausted — try another provider", "#f87171")
            self._pending_send = None
            self.btn_send.setEnabled(True)
            self.btn_send.setText("Send ▶")
            return False

        failed_name = self.model_combo.itemText(failed_idx)
        self.model_combo.setCurrentIndex(next_idx)
        new_model = self.model_combo.itemText(next_idx)
        short     = new_model.split("/")[-1]

        logger.info("Quota fallback: %s -> %s", failed_name, new_model)
        self._status(
            f"⚡ Quota on {failed_name.split('/')[-1]} — retrying with {short}…",
            "#f59e0b")

        # Re-append the user message (was rolled back in _on_ai_error)
        extra   = ps["extra"]
        content: list[dict] = []
        if mode in ("images", "both"):
            content.append({"type": "text",
                             "text": f"[{len(self._images)} screenshot(s) above]\n"})
        if mode in ("audio", "both"):
            content.append({"type": "text",
                             "text": f"[Audio transcript]\n{self._transcript}\n"})
        content.append({"type": "text", "text": extra or (
            "Solve the problem in the screenshot(s)."        if mode == "images" else
            "Solve / answer the question in the transcript." if mode == "audio"  else
            "Analyse screenshots + transcript and solve.")})
        self._history.append({"role": "user", "content": content})

        self._ai_resp = ""
        api_history   = self._build_api_history(ps["provider"])

        self.chat.append(
            f'<p style="color:#a78bfa;font-size:8pt;margin:4px 0 2px;">'
            f'— {ps["provider"]}/{short} (fallback) —</p>')

        self._pending_send = dict(ps)   # keep for possible next fallback

        self._worker = AIWorker(
            ps["provider"], ps["api_key"], new_model,
            ps["sys_p"], api_history, ps["qimgs"],
        )
        _QC = Qt.ConnectionType.QueuedConnection
        self._worker.sig.chunk.connect(self._on_chunk,    _QC)
        self._worker.sig.done.connect(self._on_done,      _QC)
        self._worker.sig.error.connect(self._on_ai_error, _QC)
        self._workers.append(self._worker)
        self._worker.finished.connect(
            lambda w=self._worker: self._workers.remove(w) if w in self._workers else None
        )
        self._worker.start()
        return True

    def _new_chat(self):
        self._history.clear()
        self.chat.clear()
        self.prompt.clear()
        self._status("New chat — queue preserved")

    # ── positioning ───────────────────────────────────────────────
    def _reposition(self):
        bg  = self.bar.geometry()
        scr = QGuiApplication.primaryScreen().geometry()
        x = max(scr.left() + 4,
                min(bg.center().x() - self.width() // 2,
                    scr.right() - self.width() - 4))
        y = max(scr.top() + 4, bg.top() - self.height() - 10)
        self.move(x, y)

    def toggle(self):
        self._reposition()
        if self._open:
            self.hide()
            self._open = False
        else:
            self._update_badge()
            self.show()
            self._open = True

    def closeEvent(self, e):
        if self._recording:
            self._stop_rec()
        super().closeEvent(e)