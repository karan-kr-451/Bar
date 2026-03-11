"""
Background audio capture + Groq Whisper STT (Google fallback).

VAD behaviour
─────────────
• Continuously monitors audio; transcription is triggered automatically
  — no polling timer needed.
• When speech ends (≥1 s of silence after active voice), the captured
  segment is flushed to the STT worker.
• If the gap between the END of the previous utterance and the START of
  the new one exceeds 3 seconds, the old transcript is discarded and
  replaced by the new one.  Short pauses (<3 s) accumulate in the
  transcript as usual.

Groq / Whisper
──────────────
• whisper-large-v3 is used whenever GROQ_API_KEY is set.
• Falls back to Google STT only if Groq fails or the key is absent.

Dependencies: sounddevice, SpeechRecognition, numpy, groq
"""

import os
import threading
import tempfile
import wave
import time
from collections import deque

import numpy as np
import sounddevice as sd
import speech_recognition as sr
from groq import Groq

from PyQt6.QtCore import QObject, pyqtSignal
from snapbar.core.logging_config import get_logger

logger = get_logger("snapbar.audio")


class AudioSignals(QObject):
    """Qt signals bridging the audio thread to the GUI thread."""
    transcript       = pyqtSignal(str)   # recognised text chunk
    error            = pyqtSignal(str)   # error message
    level            = pyqtSignal(float) # RMS level 0.0–1.0 for VU meter
    clear_transcript = pyqtSignal()      # replace transcript (>3 s silence gap)


class Transcriber:
    """
    Wraps a sounddevice InputStream with always-on Voice Activity Detection.

    Flow
    ────
    1. _cb() runs in the sounddevice audio thread on every ~100 ms block.
    2. RMS is compared to _vad_threshold to determine speech / silence.
    3. When speech begins:
         • If gap from last utterance end > 3 s  → emit clear_transcript.
         • Prepend preroll so the first syllable isn't clipped.
    4. When 1 s of silence follows active speech, flush() is called.
    5. flush() hands the buffer to _worker() (daemon thread) for STT.
    6. _worker() emits transcript (or error) back to the GUI thread.
    """

    # Silence duration (s) after speech that triggers a flush
    _FLUSH_SILENCE   : float = 1.0

    # Gap (s) between two utterances that causes the old transcript to be
    # replaced rather than appended
    _REPLACE_GAP     : float = 3.0

    # RMS level required to count a block as "speech"
    _VAD_THRESHOLD   : float = 0.015

    # Pre-speech preroll (number of ~100 ms blocks = ~1 s)
    _PREROLL_BLOCKS  : int   = 10

    def __init__(self, sig: AudioSignals):
        self.sig        = sig
        self.recognizer = sr.Recognizer()
        self._recording = False
        self._stream    = None
        self._rate      = 16000

        # Protected by Python's GIL (all accesses from one thread at a time)
        self._buf: list  = []
        self._preroll    = deque(maxlen=self._PREROLL_BLOCKS)

        # VAD state  (written only from the audio callback thread)
        self._speaking         : bool  = False
        self._last_speech_time : float = 0.0   # last block that contained speech
        self._last_speech_end  : float = time.time()  # when the last utterance finished

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
        Start always-on VAD recording from the device at *combo_idx*.
        *chunk_secs* and *qt_timer* are accepted for API compatibility but
        are not used — flushing is entirely VAD-driven.
        Returns (ok, error_message).
        """
        if combo_idx < 0 or combo_idx >= len(self._dev_map):
            return False, "No device selected"
        dev = self._dev_map[combo_idx]
        try:
            info       = sd.query_devices(dev)
            self._rate = int(info["default_samplerate"])

            # Reset all state
            self._buf  = []
            self._preroll.clear()
            self._speaking         = False
            self._last_speech_time = 0.0
            self._last_speech_end  = time.time()
            self._recording        = True

            # ~100 ms blocks give snappy VAD response
            blocksize = int(self._rate * 0.1)

            self._stream = sd.InputStream(
                samplerate = self._rate,
                channels   = min(2, int(info["max_input_channels"])),
                dtype      = "float32",
                device     = dev,
                blocksize  = blocksize,
                callback   = self._cb,
            )
            self._stream.start()
            logger.info("Recording started — device: %s", info["name"])
            return True, ""
        except Exception as e:
            logger.exception("Failed to start audio stream: %s", e)
            self._recording = False
            return False, str(e)

    def stop(self, qt_timer) -> None:
        """Stop recording and flush any buffered audio."""
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
        """Dispatch the current audio buffer to the STT worker thread."""
        if not self._buf:
            return
        buf       = list(self._buf)
        self._buf = []
        threading.Thread(target=self._worker, args=(buf,), daemon=True).start()

    # ── audio callback (runs in sounddevice audio thread) ─────────
    def _cb(self, indata, frames, t, status):
        chunk = indata.copy()
        rms   = float(np.sqrt(np.mean(chunk ** 2)))

        # Update VU meter (safe: Qt will queue-deliver to the GUI thread)
        self.sig.level.emit(min(1.0, rms * 10))

        now       = time.time()
        is_speech = rms > self._VAD_THRESHOLD

        if is_speech:
            if not self._speaking:
                # ── Speech onset ──────────────────────────────────
                self._speaking = True

                # Decide whether to replace the old transcript.
                # The signal is emitted BEFORE any new audio is buffered so
                # clear_transcript is guaranteed to arrive at the GUI thread
                # ahead of the upcoming transcript signal.
                gap = now - self._last_speech_end
                if gap > self._REPLACE_GAP:
                    self.sig.clear_transcript.emit()
                    logger.debug("Transcript replaced — %.1f s gap", gap)

                # Prepend pre-speech preroll so first syllables aren't lost
                self._buf.extend(self._preroll)
                self._preroll.clear()

                logger.debug("Speech started")

            self._buf.append(chunk)
            self._last_speech_time = now

        else:
            if self._speaking:
                # Tail of the utterance — keep buffering for a short window
                self._buf.append(chunk)

                if now - self._last_speech_time >= self._FLUSH_SILENCE:
                    # ── Speech end: flush to STT ───────────────────
                    self._speaking        = False
                    self._last_speech_end = now
                    logger.debug("Speech ended — flushing buffer")
                    self.flush()
            else:
                # Pure silence — accumulate in preroll ring buffer
                self._preroll.append(chunk)

    # ── STT worker (runs in daemon thread) ────────────────────────
    def _worker(self, buf: list) -> None:
        """
        Convert the PCM buffer to a WAV, transcribe via Groq Whisper
        (or Google STT as fallback), and emit the result.
        """
        tmp = None
        try:
            # ── Build mono int16 WAV ──────────────────────────────
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

            # ── Groq Whisper (primary) ────────────────────────────
            text      = ""
            groq_key  = os.environ.get("GROQ_API_KEY")

            if groq_key:
                try:
                    client = Groq(api_key=groq_key)
                    with open(tmp, "rb") as fh:
                        transcription = client.audio.transcriptions.create(
                            file            = (tmp, fh.read()),
                            model           = "whisper-large-v3",
                            response_format = "text",
                        )
                    text = transcription if isinstance(transcription, str) else str(transcription)
                    logger.debug("Groq Whisper transcribed %d chars", len(text))
                except Exception as e:
                    logger.error("Groq Whisper error — falling back to Google STT: %s", e)

            # ── Google STT (fallback) ─────────────────────────────
            if not text:
                with sr.AudioFile(tmp) as src:
                    aud = self.recognizer.record(src)
                text = self.recognizer.recognize_google(aud)
                logger.debug("Google STT transcribed %d chars", len(text))

            # ── Emit result ───────────────────────────────────────
            if text and text.strip():
                self.sig.transcript.emit(text.strip())

        except sr.UnknownValueError:
            pass    # silence or unintelligible — ignore quietly
        except Exception as e:
            logger.error("STT worker error: %s", e)
            self.sig.error.emit(str(e))
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass