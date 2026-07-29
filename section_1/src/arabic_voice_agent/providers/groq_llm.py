"""Groq GPT-OSS implementation of the LLM interface."""

from __future__ import annotations

from collections.abc import AsyncIterator

from groq import AsyncGroq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from arabic_voice_agent.utils.logger import logger


class GroqLLMProvider:
    """Generate Arabic responses with GPT-OSS-120B through Groq."""

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b") -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    @retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate one complete assistant reply."""
        result = await self._client.chat.completions.create(model=self._model, messages=messages, temperature=0.3)
        text = (result.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("LLM returned an empty response")
        logger.info("LLM completed (%s characters)", len(text))
        return text

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield assistant text chunks for non-LiveKit demos."""
        response = await self._client.chat.completions.create(model=self._model, messages=messages, stream=True, temperature=0.3)
        async for chunk in response:
            text = chunk.choices[0].delta.content
            if text:
                yield text
