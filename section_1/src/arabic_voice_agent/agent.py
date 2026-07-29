"""Arabic Groq and English Deepgram LiveKit voice agents."""

from __future__ import annotations

import asyncio
import json
import os
import re

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    function_tool,
    llm as lk_llm,
    stt as lk_stt,
    tts as lk_tts,
)
from livekit.agents.types import APIConnectOptions
from livekit.plugins import deepgram, groq, silero

from arabic_voice_agent.config import Settings
from arabic_voice_agent.prompts import ENGLISH_SYSTEM_PROMPT, SYSTEM_PROMPT
from arabic_voice_agent.services.order_service import OrderService
from arabic_voice_agent.utils.logger import logger


class ArabicFoodDeliveryAgent(Agent):
    """Arabic food-delivery assistant with a restricted order-status tool."""

    def __init__(self, orders: OrderService | None = None) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self._orders = orders or OrderService()

    @function_tool
    async def get_order_status(self, order_id: str) -> str:
        """Look up an order by its numeric order identifier."""
        logger.info("Arabic tool invoked: get_order_status(order_id=%s)", order_id)
        return await self._orders.get_status(order_id)


class EnglishFoodDeliveryAgent(Agent):
    """English food-delivery assistant with the same safe business tool."""

    def __init__(self, orders: OrderService | None = None) -> None:
        super().__init__(instructions=ENGLISH_SYSTEM_PROMPT)
        self._orders = orders or OrderService()

    @function_tool
    async def get_order_status(self, order_id: str) -> str:
        """Look up an order by its numeric order identifier."""
        logger.info("English tool invoked: get_order_status(order_id=%s)", order_id)
        return await self._orders.get_status_english(order_id)


class ArabicGroqTTS(groq.TTS):
    """Groq TTS configured to prevent duplicate rate-limited requests."""

    _MAX_INPUT_CHARS = 180

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = APIConnectOptions()
    ):
        """Bound the utterance and do not retry a provider 429 immediately."""
        del conn_options
        compact_text = " ".join(text.split())[: self._MAX_INPUT_CHARS]
        return super().synthesize(
            compact_text,
            conn_options=APIConnectOptions(max_retry=0, timeout=20.0),
        )


def groq_api_keys(settings: Settings) -> list[str]:
    """Return configured Groq keys in primary-then-fallback order."""
    keys = [settings.groq_api_key]
    if settings.groq_fallback_api_key and settings.groq_fallback_api_key not in keys:
        keys.append(settings.groq_fallback_api_key)
    return keys


def groq_stt_with_failover(settings: Settings, language: str, vad):
    """Use the next Groq key when transcription is temporarily unavailable."""
    providers = [
        groq.STT(model=settings.stt_model, language=language, api_key=key)
        for key in groq_api_keys(settings)
    ]
    if len(providers) == 1:
        return providers[0]
    # Groq Whisper is non-streaming. When more than one provider is wrapped,
    # LiveKit needs VAD to create a streaming adapter around each provider.
    return lk_stt.FallbackAdapter(providers, vad=vad, max_retry_per_stt=0)


def groq_llm_with_failover(settings: Settings):
    """Use the next Groq key when the primary LLM request is rate-limited."""
    providers = [groq.LLM(model=settings.llm_model, api_key=key) for key in groq_api_keys(settings)]
    if len(providers) == 1:
        return providers[0]
    return lk_llm.FallbackAdapter(providers, max_retry_per_llm=0)


def groq_tts_with_failover(settings: Settings):
    """Use the next Groq key when Orpheus reports a transient API failure."""
    providers = [
        ArabicGroqTTS(model=settings.tts_model, voice=settings.tts_voice, api_key=key)
        for key in groq_api_keys(settings)
    ]
    if len(providers) == 1:
        return providers[0]
    return lk_tts.FallbackAdapter(providers, max_retry_per_tts=0)


def build_arabic_session(settings: Settings) -> AgentSession:
    """Groq Whisper Arabic STT + GPT-OSS + Orpheus Arabic TTS."""
    arabic_vad = silero.VAD.load()
    return AgentSession(
        stt=groq_stt_with_failover(settings, "ar", arabic_vad),
        llm=groq_llm_with_failover(settings),
        tts=groq_tts_with_failover(settings),
        vad=arabic_vad,
        allow_interruptions=True,
        min_consecutive_speech_delay=0.15,
        min_endpointing_delay=0.45,
        max_endpointing_delay=2.5,
        preemptive_generation=False,
    )


def build_english_session(settings: Settings) -> AgentSession:
    """Deepgram Nova-3 English STT + Groq GPT-OSS + Deepgram Aura-2 TTS."""
    if not settings.deepgram_api_key:
        raise RuntimeError("Missing required environment variable: DEEPGRAM_API_KEY")
    return AgentSession(
        stt=deepgram.STT(model="nova-3", language="en", api_key=settings.deepgram_api_key),
        llm=groq_llm_with_failover(settings),
        tts=deepgram.TTS(model="aura-2-thalia-en", api_key=settings.deepgram_api_key),
        vad=silero.VAD.load(),
        allow_interruptions=True,
        min_consecutive_speech_delay=0.15,
        min_endpointing_delay=0.45,
        max_endpointing_delay=2.5,
    )


def add_session_logging(session: AgentSession, language: str, ctx: JobContext) -> None:
    """Log pipeline milestones and provide a browser-TTS fallback on a 429."""

    tts_failed = False

    @session.on("user_state_changed")
    def on_user_state_changed(event) -> None:
        logger.info("%s user state: %s -> %s", language, event.old_state, event.new_state)

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event) -> None:
        logger.info(
            "%s STT %s: %r",
            language,
            "final" if event.is_final else "partial",
            event.transcript,
        )

    @session.on("agent_state_changed")
    def on_agent_state_changed(event) -> None:
        logger.info("%s agent state: %s -> %s", language, event.old_state, event.new_state)

    @session.on("error")
    def on_error(event) -> None:
        nonlocal tts_failed
        if event.type == "tts_error":
            tts_failed = True
        logger.error("%s %s: %s", language, event.type, event.error)

    @session.on("conversation_item_added")
    def on_conversation_item_added(event) -> None:
        """Send text only when server-side TTS was unavailable for this turn."""
        nonlocal tts_failed
        item = event.item
        if not tts_failed or getattr(item, "role", None) != "assistant":
            return

        text = "".join(part for part in item.content if isinstance(part, str)).strip()
        tts_failed = False
        if not text:
            return

        payload = json.dumps({"language": language, "text": text}, ensure_ascii=False)
        asyncio.create_task(
            ctx.room.local_participant.publish_data(
                payload,
                reliable=True,
                topic="assistant-tts-fallback",
            )
        )
        logger.warning("%s TTS fallback sent to browser", language)


server = AgentServer()


@server.rtc_session(agent_name=os.getenv("AGENT_NAME", "food-support"))
async def entrypoint(ctx: JobContext) -> None:
    """Run the language requested in the secure LiveKit agent dispatch."""
    settings = Settings.from_env()
    voice_mode = (ctx.job.metadata or "english").strip().lower()
    if voice_mode == "english":
        session = build_english_session(settings)
        agent: Agent = EnglishFoodDeliveryAgent()
        greeting = "Greet the user in English and ask how you can help."
        logger.info("English Deepgram agent started in room %s", ctx.room.name)
    elif voice_mode == "arabic":
        session = build_arabic_session(settings)
        agent = ArabicFoodDeliveryAgent()
        greeting = "رحب بالمستخدم بالعربية واسأله كيف يمكنك مساعدته."
        logger.info("Arabic Groq agent started in room %s", ctx.room.name)
    else:
        raise RuntimeError("VOICE_MODE must be either 'english' or 'arabic'")
    add_session_logging(session, voice_mode, ctx)
    await session.start(room=ctx.room, agent=agent)
    await session.generate_reply(instructions=greeting)


def main() -> None:
    """Expose LiveKit's `dev`, `start`, and `console` commands."""
    cli.run_app(server)


if __name__ == "__main__":
    main()
