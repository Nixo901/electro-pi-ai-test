"""LangChain LCEL chain for RAG answer generation.

Uses the Groq Python SDK directly (groq>=1.0.0) to avoid the langchain-groq
version conflict (all langchain-groq releases require groq<1.0.0).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from groq import Groq
from langchain_core.documents import Document

from .config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System and human prompt strings (kept here to avoid circular imports)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise, helpful assistant. Your task is to answer the user's question \
using ONLY the context passages provided below.

Rules you MUST follow:
1. Answer solely from the context. Do not add facts from your training data.
2. After your answer, list every source you drew from in a "Sources" section, \
   formatted as:
     Sources:
     - [<filename>] "<short verbatim excerpt>"
3. If the context does not contain enough information to answer the question, \
   respond with exactly:
     I don't have enough context to answer this question.
   Do NOT make up an answer.
"""

_HUMAN_TEMPLATE = """\
Context:
{context}

Question: {question}
"""


def _format_context(chunks: List[Tuple[Document, float]]) -> str:
    """Format retrieved (document, score) pairs into a readable context block."""
    parts: List[str] = []
    for i, (doc, score) in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        parts.append(
            f"[Passage {i} | source: {source} | chunk: {chunk_idx} | score: {score:.3f}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_citations(chunks: List[Tuple[Document, float]]) -> List[dict]:
    """Build a structured list of citation objects from retrieved chunks."""
    citations = []
    for doc, score in chunks:
        source = doc.metadata.get("source", "unknown")
        excerpt = doc.page_content.strip()
        if len(excerpt) > 200:
            excerpt = excerpt[:200].rsplit(" ", 1)[0] + " …"
        citations.append(
            {
                "source": source,
                "chunk_index": doc.metadata.get("chunk_index"),
                "score": round(score, 4),
                "excerpt": excerpt,
            }
        )
    return citations


class RAGChain:
    """Calls the Groq API (groq>=1.0.0 SDK) to generate grounded answers.

    Why not langchain-groq?
    All released versions of langchain-groq require ``groq<1.0.0``, which
    conflicts with section_1's ``groq>=1.0.0`` requirement. We therefore call
    the Groq SDK directly and build prompts as plain dicts, which is simpler
    and has zero version-conflict risk.
    """

    def __init__(self) -> None:
        self._client = Groq(api_key=settings.groq_api_key)

    def answer(
        self,
        question: str,
        chunks: List[Tuple[Document, float]],
    ) -> Tuple[str, List[dict]]:
        """Generate an answer grounded in *chunks* and return (answer, citations).

        Parameters
        ----------
        question : str
            The user's natural-language question.
        chunks : List[Tuple[Document, float]]
            Retrieved (Document, similarity_score) pairs.

        Returns
        -------
        answer : str
            LLM-generated answer grounded in the retrieved context.
        citations : List[dict]
            Structured citation objects: source, chunk_index, score, excerpt.
        """
        context = _format_context(chunks)
        human_msg = _HUMAN_TEMPLATE.format(context=context, question=question)

        logger.debug("Invoking Groq LLM with %d context chunks …", len(chunks))
        response = self._client.chat.completions.create(
            model=settings.groq_model,
            temperature=settings.groq_temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": human_msg},
            ],
        )
        answer_text: str = response.choices[0].message.content or ""
        citations = _extract_citations(chunks)
        return answer_text.strip(), citations
