"""Prompt templates used by the RAG chain."""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# System instructions for the RAG chain
# ---------------------------------------------------------------------------
_SYSTEM = """\
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

_HUMAN = """\
Context:
{context}

Question: {question}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ]
)

# ---------------------------------------------------------------------------
# Canned response returned when the relevance gate fires (no LLM call)
# ---------------------------------------------------------------------------
NO_CONTEXT_RESPONSE = (
    "I don't have enough context to answer this question. "
    "The documents in my knowledge base do not appear to contain relevant information "
    "for your query. Please try rephrasing your question or ask about a topic "
    "covered in the available documents."
)
