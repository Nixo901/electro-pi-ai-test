"""Small vendor-neutral interfaces used by the offline pipeline."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol


class STTProvider(Protocol):
    """Converts an audio file to text."""

    async def transcribe(self, audio_path: Path) -> str: ...


class LLMProvider(Protocol):
    """Produces a response from a list of OpenAI-compatible messages."""

    async def generate(self, messages: list[dict[str, str]]) -> str: ...

    def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


class TTSProvider(Protocol):
    """Converts Arabic text to WAV audio bytes."""

    async def synthesize(self, text: str) -> bytes: ...
