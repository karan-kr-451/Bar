"""QApplication entry point. Import and call main() from run.py at project root."""
import sys
import traceback
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QKeySequence
from snapbar.bar import SnapBar

from dotenv import load_dotenv
import logging
from snapbar.core.logging_config import setup_logging


class _HotkeyRelay(QObject):
    """
    Bridge between the keyboard library's background thread and the Qt main thread.
    keyboard callbacks must never touch Qt widgets directly — emit a signal instead,
    which Qt will deliver on the main thread via the event loop.
    """
    _triggered = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._triggered.connect(self._run)

    def _run(self, cb):
        try:
            cb()
        except Exception:
            pass  # exception hooks will log it

    def fire(self, cb):
        """Call from any thread — safely dispatches cb() on the main thread."""
        self._triggered.emit(cb)


def _register_global_shortcuts(bar, log):
    try:
        import keyboard
    except ImportError:
        log.error("'keyboard' package not installed. Run: pip install keyboard")
        return [], None

    relay = _HotkeyRelay()

    def make_safe_cb(cb):
        def safe():
            relay.fire(cb)
        return safe

    # Mode buttons
    def make_mode_cb(index):
        def cb():
            buttons = bar._ai._mode_group.buttons()
            if 0 <= index < len(buttons):
                buttons[index].click()
        return cb

    # Category cycling
    def cycle_category(direction):
        def cb():
            buttons = bar._ai._cat_group.buttons()
            if not buttons:
                return
            current = next(
                (i for i, b in enumerate(buttons) if b.isChecked()), 0
            )
            nxt = (current + direction) % len(buttons)
            buttons[nxt].click()
            log.info(
                "Category cycled %s → %s",
                "forward" if direction > 0 else "backward",
                buttons[nxt].text(),
            )
        return cb

    hotkeys = [
        # Screenshots
        ("shift+alt+f9",    bar._cap_full,         "Full screenshot"),
        ("shift+alt+f10",   bar._cap_region,        "Region screenshot"),
        ("shift+alt+f11",   bar._cap_window,        "Window screenshot"),
        # AI panel
        ("shift+alt+a",     bar._toggle_ai,         "Toggle AI panel"),
        ("shift+alt+h",     bar._hide_panel,        "Hide AI panel"),
        # Send
        ("shift+alt+enter", bar._ai._send,          "Send request to AI"),
        ("ctrl+enter",      bar._ai._send,          "Send request to AI (alt)"),
        #clear que
        ("shift+alt+c", bar._ai._clear_queue, "Clear send queue"),
        ("shift+alt+delete", bar._ai._clear_queue, "Clear send queue (alt)"),
        # Modes
        ("shift+alt+1",     make_mode_cb(0),        "Switch to Images mode"),
        ("shift+alt+2",     make_mode_cb(1),        "Switch to Audio mode"),
        ("shift+alt+3",     make_mode_cb(2),        "Switch to Both mode"),
        # Category cycling
        ("shift+alt+right", cycle_category(+1),     "Next category"),
        ("shift+alt+left",  cycle_category(-1),     "Previous category"),
    ]

    registered = 0
    for combo, callback, description in hotkeys:
        try:
            keyboard.add_hotkey(combo, make_safe_cb(callback), suppress=False)
            log.info("Registered global hotkey: %s -> %s", description, combo)
            registered += 1
        except Exception as e:
            log.error("Failed to register hotkey %s (%s): %s", description, combo, e)

    log.info("Registered %d global keyboard shortcuts", registered)
    return hotkeys, relay  # both must stay alive — relay owns the signal connection


def _install_exception_hooks(log):
    """
    Catch ALL unhandled exceptions — including those swallowed silently
    by PyQt's signal/slot mechanism — and write them to the log file.
    """
    def excepthook(exc_type, exc_value, exc_tb):
        log.critical(
            "Unhandled exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = excepthook

    try:
        from PyQt6.QtCore import qInstallMessageHandler, QtMsgType
        def qt_message_handler(mode, context, message):
            if mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
                log.critical("Qt [%s] %s:%d %s",
                             mode.name, context.file, context.line, message)
            elif mode == QtMsgType.QtWarningMsg:
                log.warning("Qt [Warning] %s", message)
            else:
                log.debug("Qt [%s] %s", mode.name, message)
        qInstallMessageHandler(qt_message_handler)
    except Exception:
        log.warning("Could not install Qt message handler")


import gc

def main():
    gc.disable()

    try:
        load_dotenv()
        setup_logging()
        log = logging.getLogger("snapbar.main")
        _install_exception_hooks(log)

        app = QApplication(sys.argv)
        log.info("Application starting (GC disabled)...")
        app.setQuitOnLastWindowClosed(True)
        bar = SnapBar()

        # Guarantee AIWorker threads are cancelled before the process exits,
        # even when the panel is hidden (closeEvent would never fire for a
        # hidden widget, so we hook aboutToQuit directly).
        app.aboutToQuit.connect(bar._ai._shutdown_workers)
        log.info("Connected app.aboutToQuit → AIPanel._shutdown_workers")

        # Keep both hotkeys list AND relay object alive for the entire session
        _shortcuts, _relay = _register_global_shortcuts(bar, log)

        exit_code = app.exec()
        log.info("Application closed with exit code %d", exit_code)

        try:
            import keyboard
            keyboard.unhook_all()
            log.info("Keyboard hooks removed.")
        except Exception as e:
            log.debug("Failed to unhook keyboard: %s", e)

        gc.enable()
        log.info("GC re-enabled on shutdown.")
        sys.exit(exit_code)
    finally:
        gc.enable()


if __name__ == "__main__":
    print("Run from project root:  python run.py")
    main()