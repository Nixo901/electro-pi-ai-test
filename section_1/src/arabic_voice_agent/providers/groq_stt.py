"""Groq Whisper implementation of the STT interface."""

from __future__ import annotations

from pathlib import Path

from groq import AsyncGroq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from arabic_voice_agent.utils.logger import logger


class GroqSTTProvider:
    """Transcribe Arabic audio with Groq Whisper Large V3 Turbo."""

    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo") -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    @retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def transcribe(self, audio_path: Path) -> str:
        """Return an Arabic transcript, rejecting absent or empty inputs."""
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        with audio_path.open("rb") as audio_file:
            result = await self._client.audio.transcriptions.create(
                file=(audio_path.name, audio_file.read()),
                model=self._model,
                language="ar",
                temperature=0,
                response_format="json",
            )
        text = result.text.strip()
        if not text:
            raise ValueError("STT returned an empty transcript")
        logger.info("STT completed (%s characters)", len(text))
        return text
