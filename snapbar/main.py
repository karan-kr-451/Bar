"""QApplication entry point. Import and call main() from run.py at project root."""
import sys
import traceback
from PyQt6.QtWidgets import QApplication
from snapbar.bar import SnapBar

from dotenv import load_dotenv
import logging
from snapbar.core.logging_config import setup_logging


def _install_exception_hooks(log):
    """
    Catch ALL unhandled exceptions — including those swallowed silently
    by PyQt's signal/slot mechanism — and write them to the log file.

    Without this, any exception raised in a slot connected to a signal
    gets printed to stderr (invisible when running as a windowed app)
    and the widget silently stops working.
    """
    # 1. Standard Python unhandled exception (non-main-thread crashes etc.)
    def excepthook(exc_type, exc_value, exc_tb):
        log.critical(
            "Unhandled exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = excepthook

    # 2. PyQt6 slot exceptions — Qt catches these before Python can,
    #    so we must override the Qt message handler too.
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
    # Stop GC when application starts
    gc.disable()
    
    try:
        load_dotenv()
        setup_logging()
        log = logging.getLogger("snapbar.main")
        _install_exception_hooks(log)
    
        app = QApplication(sys.argv)
        log.info("Application starting (GC disabled)...")
        app.setQuitOnLastWindowClosed(True)
        SnapBar()
        exit_code = app.exec()
        log.info("Application closed with exit code %d", exit_code)
        
        # Explicitly enable before sys.exit in try block
        gc.enable()
        log.info("GC re-enabled on shutdown.")
        sys.exit(exit_code)
    finally:
        # Fallback to ensure it's reactivated even if exceptions bypass sys.exit
        gc.enable()


if __name__ == "__main__":
    print("Run from project root:  python run.py")
    main()