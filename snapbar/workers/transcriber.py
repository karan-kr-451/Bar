"""
Background audio capture + Google Speech-to-Text.

Dependencies: sounddevice, SpeechRecognition, numpy
"""

import os
import threading
import tempfile
import wave

import numpy as np
import sounddevice as sd
import speech_recognition as sr

from PyQt6.QtCore import QObject, pyqtSignal
from snapbar.core.logging_config import get_logger

logger = get_logger("snapbar.audio")


class AudioSignals(QObject):
    """Qt signals bridging the audio thread to the GUI thread."""
    transcript = pyqtSignal(str)    # recognised text chunk
    error      = pyqtSignal(str)    # error message
    level      = pyqtSignal(float)  # RMS level 0.0–1.0 for VU meter


class Transcriber:
    """
    Wraps sounddevice InputStream.
    Call list_devices() to populate a combo box.
    Call start() to begin recording; audio is flushed every chunk_secs seconds.
    Call stop() or flush() manually.
    """

    def __init__(self, sig: AudioSignals):
        self.sig        = sig
        self.recognizer = sr.Recognizer()
        self._recording = False
        self._stream    = None
        self._buf: list = []
        self._rate      = 16000
        self._dev_map: list[int] = []   # combo index → sounddevice device index

    # ── device enumeration ────────────────────────────────────────
    def list_devices(self) -> list[str]:
        """Return display labels for all input devices."""
        labels: list[str] = []
        self._dev_map = []
        try:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    labels.append(f"{d['name']}  [{d['max_input_channels']}ch]")
                    self._dev_map.append(i)
        except Exception as e:
            logger.error("Error listing audio devices: %s", e)
            labels.append(f"Error listing devices: {e}")
        return labels

    # ── start / stop ──────────────────────────────────────────────
    def start(self, combo_idx: int, chunk_secs: int, qt_timer) -> tuple[bool, str]:
        """
        Start recording from the device at combo_idx.
        qt_timer is a QTimer that will call flush() every chunk_secs seconds.
        Returns (ok, error_message).
        """
        if combo_idx < 0 or combo_idx >= len(self._dev_map):
            return False, "No device selected"
        dev = self._dev_map[combo_idx]
        try:
            info        = sd.query_devices(dev)
            self._rate  = int(info["default_samplerate"])
            self._buf   = []
            self._recording = True
            self._stream = sd.InputStream(
                samplerate  = self._rate,
                channels    = min(2, int(info["max_input_channels"])),
                dtype       = "float32",
                device      = dev,
                blocksize   = 4096,
                callback    = self._cb,
            )
            self._stream.start()
            logger.info("Recording started on device: %s", info["name"])
            qt_timer.start(chunk_secs * 1000)
            return True, ""
        except Exception as e:
            logger.exception("Failed to start audio stream: %s", e)
            self._recording = False
            return False, str(e)

    def stop(self, qt_timer) -> None:
        """Stop recording and flush any remaining audio."""
        self._recording = False
        qt_timer.stop()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.flush()

    def flush(self) -> None:
        """Send current buffer to the STT worker thread."""
        if not self._buf:
            return
        buf        = list(self._buf)
        self._buf  = []
        threading.Thread(target=self._worker, args=(buf,), daemon=True).start()

    # ── internal ──────────────────────────────────────────────────
    def _cb(self, indata, frames, t, status):
        """sounddevice callback — runs in audio thread."""
        self._buf.append(indata.copy())
        rms = float(np.sqrt(np.mean(indata ** 2)))
        self.sig.level.emit(min(1.0, rms * 10))

    def _worker(self, buf: list) -> None:
        """Transcription worker — runs in daemon thread."""
        try:
            audio = np.concatenate(buf, axis=0)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)          # stereo → mono
            audio = (audio * 32767).astype(np.int16)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._rate)
                wf.writeframes(audio.tobytes())

            with sr.AudioFile(tmp) as src:
                aud = self.recognizer.record(src)
            os.unlink(tmp)

            text = self.recognizer.recognize_google(aud)
            if text.strip():
                self.sig.transcript.emit(text.strip())

        except sr.UnknownValueError:
            pass    # silence or unintelligible — ignore
        except Exception as e:
            logger.error("STT Worker error: %s", e)
            self.sig.error.emit(str(e))
