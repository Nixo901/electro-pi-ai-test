"""Provider-independent audio round-trip orchestration."""

from __future__ import annotations

from pathlib import Path

from arabic_voice_agent.providers.base import STTProvider, TTSProvider


class SpeechService:
    """Coordinates only speech transformations."""

    def __init__(self, stt: STTProvider, tts: TTSProvider) -> None:
        self._stt = stt
        self._tts = tts

    async def transcribe(self, path: Path) -> str:
        """Transcribe input audio."""
        return await self._stt.transcribe(path)

    async def synthesize_to_file(self, text: str, output_path: Path) -> Path:
        """Write synthesized WAV audio to the requested destination."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(await self._tts.synthesize(text))
        return output_path
