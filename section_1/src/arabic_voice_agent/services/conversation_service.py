"""Provider-independent text conversation orchestration."""

from __future__ import annotations

from arabic_voice_agent.providers.base import LLMProvider
from arabic_voice_agent.prompts import SYSTEM_PROMPT


class ConversationService:
    """Creates short Arabic answers for the standalone pipeline demonstration."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def reply(self, transcript: str) -> str:
        """Generate a reply without embedding vendor logic."""
        return await self._llm.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ]
        )
