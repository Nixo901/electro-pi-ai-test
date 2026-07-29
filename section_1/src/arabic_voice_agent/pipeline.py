"""Standalone Audio -> STT -> LLM -> TTS pipeline for component testing."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from arabic_voice_agent.config import Settings
from arabic_voice_agent.providers.groq_llm import GroqLLMProvider
from arabic_voice_agent.providers.groq_stt import GroqSTTProvider
from arabic_voice_agent.providers.groq_tts import GroqTTSProvider
from arabic_voice_agent.services.conversation_service import ConversationService
from arabic_voice_agent.services.speech_service import SpeechService
from arabic_voice_agent.utils.logger import logger


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Inspect-able outputs of one speech round trip."""

    transcript: str
    response: str
    audio_path: Path


class VoicePipeline:
    """Composable pipeline used before connecting to a LiveKit room."""

    def __init__(self, speech: SpeechService, conversation: ConversationService) -> None:
        self._speech = speech
        self._conversation = conversation

    async def run(self, input_audio: Path, output_audio: Path) -> PipelineResult:
        """Perform an Arabic audio round trip."""
        transcript = await self._speech.transcribe(input_audio)
        response = await self._conversation.reply(transcript)
        audio_path = await self._speech.synthesize_to_file(response, output_audio)
        return PipelineResult(transcript, response, audio_path)


def build_pipeline(settings: Settings) -> VoicePipeline:
    """Wire Groq implementations into vendor-neutral services."""
    stt = GroqSTTProvider(settings.groq_api_key, settings.stt_model)
    llm = GroqLLMProvider(settings.groq_api_key, settings.llm_model)
    tts = GroqTTSProvider(settings.groq_api_key, settings.tts_model, settings.tts_voice)
    return VoicePipeline(SpeechService(stt, tts), ConversationService(llm))


async def _run_cli(input_file: str, output_file: str) -> None:
    result = await build_pipeline(Settings.from_env()).run(Path(input_file), Path(output_file))
    logger.info("Transcript: %s", result.transcript)
    logger.info("Response: %s", result.response)
    logger.info("Audio saved to: %s", result.audio_path)


def main() -> None:
    """Run the component-test pipeline from a terminal."""
    parser = argparse.ArgumentParser(description="Arabic Groq audio round trip")
    parser.add_argument("input_audio")
    parser.add_argument("--output", default="artifacts/response.wav")
    args = parser.parse_args()
    asyncio.run(_run_cli(args.input_audio, args.output))


if __name__ == "__main__":
    main()
