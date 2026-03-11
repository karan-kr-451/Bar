"""
Multi-provider streaming AI worker (runs in QThread).

Providers
---------
anthropic   pip install anthropic
google      pip install google-genai
groq        pip install groq
openrouter  pip install httpx

All providers stream tokens via AISignals.chunk.

Timeout / cancellation
----------------------
Every provider enforces CONNECT_TIMEOUT (first byte) and READ_TIMEOUT
(per-chunk idle).  The worker also checks self._cancelled between chunks
so AIPanel can call worker.cancel() to abort mid-stream cleanly.
"""

import threading
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from snapbar.core.logging_config import get_logger

logger = get_logger("snapbar.ai")

# ── Timeouts (seconds) ────────────────────────────────────────────
CONNECT_TIMEOUT = 15   # max seconds to establish connection
READ_TIMEOUT    = 30   # max seconds of silence between chunks
TOTAL_TIMEOUT   = 180  # absolute cap for the entire request


class AISignals(QObject):
    chunk = pyqtSignal(str)   # one streamed token / delta
    done  = pyqtSignal()      # stream finished cleanly
    error = pyqtSignal(str)   # error message (worker stopped)


# ── Model catalogue ───────────────────────────────────────────────
PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    "google": [
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ],
    "groq": [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.2-90b-vision-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct",
        "groq/compound",
        "groq/compound-mini",
        "mixtral-8x7b-32768",
    ],
    "openrouter": [
        "meta-llama/llama-3.2-90b-vision-instruct",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-6",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "qwen/qwen2.5-vl-72b-instruct",
        "mistralai/mistral-large",
        "google/gemini-2.0-flash-exp:free",
    ],
}

PROVIDERS: list[str] = list(PROVIDER_MODELS.keys())


class AIWorker(QThread):
    """
    Spawned per send.  Streams tokens via self.sig.chunk.
    Call cancel() from the GUI thread to abort cleanly.
    """

    def __init__(
        self,
        provider:      str,
        api_key:       str,
        model:         str,
        system_prompt: str,
        messages:      list,
        images:        list = None,
    ):
        super().__init__()
        self.provider      = provider
        self.api_key       = api_key
        self.model         = model
        self.system_prompt = system_prompt
        self.messages      = messages
        self.images        = images or []
        self.sig           = AISignals()
        self._cancelled    = False
        self._cancel_event = threading.Event()

    def cancel(self):
        """Call from GUI thread to request cancellation."""
        logger.info("AIWorker.cancel() called")
        self._cancelled = True
        self._cancel_event.set()

    # ── main thread entry ─────────────────────────────────────────
    def run(self):
        logger.info("AI Task started: provider=%s, model=%s", self.provider, self.model)
        try:
            if self.images:
                import concurrent.futures
                from snapbar.core.utils import qimage_to_b64
                
                logger.debug("Encoding %d images for worker payload in parallel...", len(self.images))
                
                for msg in reversed(self.messages):
                    if msg["role"] == "user":
                        img_blocks = []
                        total_bytes = 0
                        
                        # Process images in parallel
                        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.images), 8)) as executor:
                            b64_results = list(executor.map(qimage_to_b64, self.images))
                            
                        for b64_data in b64_results:
                            total_bytes += len(b64_data)
                            img_blocks.append({
                                "type": "image",
                                "source": {
                                    "type":       "base64",
                                    "media_type": "image/jpeg",
                                    "data":       b64_data,
                                }
                            })
                        msg["content"] = img_blocks + msg["content"]
                        logger.info(
                            "Encoders: %d images attached to final user turn "
                            "(approx %.2f MB base64)",
                            len(self.images), total_bytes / (1024 * 1024)
                        )
                        break

            if self._cancelled:
                logger.info("Cancelled before provider call")
                return

            logger.info("Executing %s provider...", self.provider)
            dispatch = {
                "anthropic":  self._run_anthropic,
                "google":     self._run_google,
                "groq":       self._run_groq,
                "openrouter": self._run_openrouter,
            }
            if self.provider not in dispatch:
                raise KeyError(self.provider)
            dispatch[self.provider]()

            if not self._cancelled:
                logger.info("AI Worker task finished successfully.")

        except KeyError:
            msg = f"Unknown provider: {self.provider}"
            logger.error(msg)
            self.sig.error.emit(msg)
        except Exception as e:
            logger.exception("AI Worker unhandled exception: %s", e)
            if not self._cancelled:
                self.sig.error.emit(str(e))

    # ── Anthropic ─────────────────────────────────────────────────
    def _run_anthropic(self):
        import anthropic
        import httpx as _httpx

        client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=_httpx.Timeout(
                connect=CONNECT_TIMEOUT,
                read=READ_TIMEOUT,
                write=30,
                pool=5,
            ),
        )
        kw = dict(model=self.model, max_tokens=4096, messages=self.messages)
        if self.system_prompt:
            kw["system"] = self.system_prompt

        logger.debug("Anthropic: starting stream...")
        chunk_count = 0
        with client.messages.stream(**kw) as stream:
            for text in stream.text_stream:
                if self._cancelled:
                    logger.info("Anthropic: cancelled after %d chunks", chunk_count)
                    return
                self.sig.chunk.emit(text)
                chunk_count += 1
                if chunk_count % 100 == 0:
                    logger.debug("Anthropic: %d chunks so far", chunk_count)

        logger.info("Anthropic: stream finished (%d chunks)", chunk_count)
        self.sig.done.emit()

    # ── Google Gemini ─────────────────────────────────────────────
    def _run_google(self):
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=self.api_key)
        config = gtypes.GenerateContentConfig(
            system_instruction=self.system_prompt or None,
            thinking_config=gtypes.ThinkingConfig(thinking_budget=-1),
        )

        logger.debug("Google: starting generate_content_stream...")
        chunk_count = 0
        for chunk in client.models.generate_content_stream(
            model    = self.model,
            contents = self._to_genai_contents(),
            config   = config,
        ):
            if self._cancelled:
                logger.info("Google: cancelled after %d chunks", chunk_count)
                return
            if chunk.text:
                self.sig.chunk.emit(chunk.text)
                chunk_count += 1

        logger.info("Google: stream finished (%d chunks)", chunk_count)
        self.sig.done.emit()

    def _to_genai_contents(self) -> list:
        import base64
        contents = []
        for i, msg in enumerate(self.messages):
            role     = msg["role"]
            content  = msg["content"]
            sdk_role = "model" if role == "assistant" else "user"
            is_last_user = (i == len(self.messages) - 1 and role == "user")
            parts = []
            if isinstance(content, str):
                if content.strip():
                    parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    btype = block.get("type")
                    if btype == "text" and block.get("text", "").strip():
                        parts.append({"text": block["text"]})
                    elif btype == "image" and is_last_user:
                        from google.genai import types as gtypes
                        raw = base64.b64decode(block["source"]["data"])
                        parts.append(
                            gtypes.Part.from_bytes(data=raw, mime_type="image/jpeg")
                        )
            if parts:
                contents.append({"role": sdk_role, "parts": parts})
        return contents

    # ── Groq ──────────────────────────────────────────────────────
    def _run_groq(self):
        from groq import Groq
        import httpx as _httpx

        # Groq SDK uses httpx internally — explicit timeouts prevent infinite hangs
        # when the API stalls between chunks (READ_TIMEOUT = per-chunk idle limit).
        client = Groq(
            api_key=self.api_key,
            timeout=_httpx.Timeout(
                connect=CONNECT_TIMEOUT,
                read=READ_TIMEOUT,
                write=30,
                pool=5,
            ),
        )

        logger.debug("Groq: opening stream...")
        chunk_count = 0

        try:
            stream = client.chat.completions.create(
                model      = self.model,
                messages   = self._to_openai_messages(),
                max_tokens = 4096,
                stream     = True,
            )
        except Exception as e:
            logger.exception("Groq: failed to open stream: %s", e)
            self.sig.error.emit(f"Groq connection error: {e}")
            return

        logger.debug("Groq: stream opened, reading chunks...")
        try:
            for chunk in stream:
                if self._cancelled:
                    logger.info("Groq: cancelled after %d chunks", chunk_count)
                    return
                try:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        self.sig.chunk.emit(delta)
                        chunk_count += 1
                        if chunk_count % 100 == 0:
                            logger.debug("Groq: %d chunks received", chunk_count)
                except (AttributeError, IndexError) as e:
                    # Heartbeat / usage chunks have no choices — skip silently
                    logger.debug("Groq: skipping malformed chunk (%s)", e)
                    continue
        except Exception as e:
            logger.exception(
                "Groq: stream error after %d chunks: %s", chunk_count, e)
            self.sig.error.emit(f"Groq stream error: {e}")
            return

        logger.info("Groq: stream finished (%d chunks)", chunk_count)
        self.sig.done.emit()

    # ── OpenRouter ────────────────────────────────────────────────
    def _run_openrouter(self):
        import json
        import httpx

        logger.debug("OpenRouter: opening SSE stream...")
        chunk_count = 0

        try:
            with httpx.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  "https://snapbar",
                    "X-Title":       "SnapBar",
                },
                json={
                    "model":      self.model,
                    "messages":   self._to_openai_messages(),
                    "max_tokens": 4096,
                    "stream":     True,
                },
                timeout=httpx.Timeout(
                    connect=CONNECT_TIMEOUT,
                    read=READ_TIMEOUT,
                    write=30,
                    pool=5,
                ),
            ) as r:
                logger.info("OpenRouter: HTTP %d", r.status_code)
                r.raise_for_status()

                for line in r.iter_lines():
                    if self._cancelled:
                        logger.info("OpenRouter: cancelled after %d chunks", chunk_count)
                        return
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        logger.debug("OpenRouter: [DONE]")
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta:
                            self.sig.chunk.emit(delta)
                            chunk_count += 1
                    except Exception as e:
                        logger.debug("OpenRouter: skipping bad SSE line: %s", e)

        except Exception as e:
            logger.exception(
                "OpenRouter: stream error after %d chunks: %s", chunk_count, e)
            self.sig.error.emit(f"OpenRouter error: {e}")
            return

        logger.info("OpenRouter: stream finished (%d chunks)", chunk_count)
        self.sig.done.emit()

    # ── Anthropic-style → OpenAI-style ───────────────────────────
    def _to_openai_messages(self) -> list:
        out = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        for msg in self.messages:
            role, content = msg["role"], msg["content"]
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            parts = []
            for block in content:
                if block["type"] == "text":
                    parts.append({"type": "text", "text": block["text"]})
                elif block["type"] == "image":
                    b64  = block["source"]["data"]
                    mime = block["source"]["media_type"]
                    parts.append({
                        "type":      "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
            out.append({"role": role, "content": parts})
        return out