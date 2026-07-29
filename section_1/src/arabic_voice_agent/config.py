"""Application configuration loaded once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load project defaults once. Validation below intentionally reads only the
# current process environment, which lets callers and tests override or remove
# values predictably.
load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings."""

    groq_api_key: str
    groq_fallback_api_key: str | None
    deepgram_api_key: str | None
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    stt_model: str
    llm_model: str
    tts_model: str
    tts_voice: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Validate settings from the current process environment."""
        return cls(
            groq_api_key=_required("GROQ_API_KEY"),
            groq_fallback_api_key=os.getenv("GROQ_API_KEY_FALLBACK"),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
            livekit_url=_required("LIVEKIT_URL"),
            livekit_api_key=_required("LIVEKIT_API_KEY"),
            livekit_api_secret=_required("LIVEKIT_API_SECRET"),
            stt_model=os.getenv("STT_MODEL", "whisper-large-v3-turbo"),
            llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
            tts_model=os.getenv("TTS_MODEL", "canopylabs/orpheus-arabic-saudi"),
            tts_voice=os.getenv("TTS_VOICE", "abdullah"),
        )
