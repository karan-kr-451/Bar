"""
Drop-in replacement for the "google" entry in PROVIDER_MODELS (ai_worker.py).

Generated from live API output — only models that support generateContent
and are suitable for chat/vision are included.

Capability legend
─────────────────
  🧠✦  vision + reasoning  ← auto-selected when images are queued
  ✦    vision only
  (no mark) text only

Excluded intentionally:
  - TTS / native-audio    (gemini-2.5-*-tts, gemini-2.5-flash-native-audio-*)
  - Embed                 (gemini-embedding-001)
  - Image-gen only        (gemini-2.0-flash-exp-image-generation,
                           gemini-2.5-flash-image, *-image-preview, imagen-*)
  - Video-gen only        (veo-*)
  - Answer-only           (aqa)
  - Robotics              (gemini-robotics-er-1.5-preview)
  - Computer-use          (gemini-2.5-computer-use-preview-10-2025)
  - Deep research         (deep-research-pro-preview-12-2025)

Usage — in ai_worker.py replace the "google" list inside PROVIDER_MODELS:

    PROVIDER_MODELS: dict[str, list[str]] = {
        "anthropic":  [...],
        "google":     GOOGLE_MODELS,   # <- replace existing list
        "groq":       [...],
        "openrouter": [...],
    }
"""

GOOGLE_MODELS = [
    # -- Gemini 3.x -- vision + reasoning (thinking built-in) ----------------
    "gemini-3.1-pro-preview",               # best reasoning + vision
    "gemini-3.1-pro-preview-customtools",   # + tool-use optimised
    "gemini-3.1-flash-lite-preview",        # fast + reasoning
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",               # fast + reasoning

    # -- Gemini 2.5 stable -- vision + reasoning ------------------------------
    "gemini-2.5-pro",                       # highest quality, 1M ctx
    "gemini-2.5-flash",                     # speed/quality sweet spot
    "gemini-2.5-flash-lite",                # lightweight reasoning
    "gemini-2.5-flash-lite-preview-09-2025",

    # -- Gemini 2.0 stable -- vision only ------------------------------------
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",

    # -- Convenience aliases (always point to latest in each tier) ------------
    "gemini-pro-latest",       # -> latest pro (2.5+, reasoning)
    "gemini-flash-latest",     # -> latest flash (2.5+, reasoning)
    "gemini-flash-lite-latest",

    # -- Gemma 3 -- vision only (open-weight, on-device capable) -------------
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "gemma-3-1b-it",
    "gemma-3n-e4b-it",
    "gemma-3n-e2b-it",
]


# -- How auto-select works with these models ----------------------------------
#
# _model_tier() in ai_panel.py scores each model:
#   tier 2  ->  vision AND reasoning  ("2.5-", "gemini-3" in name)
#   tier 1  ->  vision only           (gemini-2.0-*, gemma-3-*)
#   tier 0  ->  neither
#
# On image queue: scan picks max(tier) -> gemini-3.1-pro-preview is chosen.
# Status bar:  "Auto-switched to 🧠✦ vision + reasoning: gemini-3.1-pro-preview"
#
# To change the preferred model, reorder this list — first highest-tier
# entry wins.