"""Groq Orpheus implementation of the TTS interface."""

from __future__ import annotations

from groq import AsyncGroq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from arabic_voice_agent.utils.logger import logger


class GroqTTSProvider:
    """Generate Saudi Arabic WAV bytes with the Orpheus model."""

    max_chars = 200

    def __init__(self, api_key: str, model: str, voice: str = "abdullah") -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._voice = voice

    @retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def synthesize(self, text: str) -> bytes:
        """Return one WAV payload; Orpheus accepts at most 200 characters."""
        normalized = text.strip()
        if not normalized:
            raise ValueError("Cannot synthesize empty text")
        if len(normalized) > self.max_chars:
            raise ValueError(f"Orpheus input exceeds {self.max_chars} characters")
        response = await self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=normalized,
            response_format="wav",
        )
        data = await response.read()
        if not data:
            raise ValueError("TTS returned no audio")
        logger.info("TTS completed (%s bytes)", len(data))
        return data
