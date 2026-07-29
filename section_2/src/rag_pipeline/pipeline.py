"""Top-level RAGPipeline class — the single public API for Task 2.1."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from .chain import RAGChain
from .config import settings
from .embedder import get_embeddings
from .loader import DocumentLoader
from .prompts import NO_CONTEXT_RESPONSE
from .store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Structured result returned by :meth:`RAGPipeline.query`."""

    question: str
    answer: str
    citations: List[dict] = field(default_factory=list)
    no_context: bool = False
    """True if the relevance gate fired and the LLM was NOT called."""
    retrieved_chunks: List[tuple] = field(default_factory=list)
    """Raw (Document, score) pairs for inspection / debugging."""


class RAGPipeline:
    """End-to-end RAG pipeline.

    Typical usage::

        pipeline = RAGPipeline()
        pipeline.build("docs/")           # chunk → embed → index
        result = pipeline.query("What is FAISS?")
        print(result.answer)
        for citation in result.citations:
            print(citation)

    You can also persist and reload the index::

        pipeline.save_index()
        pipeline2 = RAGPipeline()
        pipeline2.load_index()
        result = pipeline2.query("…")
    """

    def __init__(
        self,
        score_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self._score_threshold = (
            score_threshold if score_threshold is not None else settings.score_threshold
        )
        self._top_k = top_k if top_k is not None else settings.top_k

        embeddings = get_embeddings()
        self._store = VectorStoreManager(embeddings)
        self._chain = RAGChain()
        self._loader = DocumentLoader()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build(self, docs_dir: str | Path = "docs/") -> None:
        """Load, chunk, embed, and index all markdown docs from *docs_dir*.

        This is the only method that performs disk/model I/O at build time.
        Subsequent calls to :meth:`query` are fast (embedding + FAISS search +
        single LLM call).

        Parameters
        ----------
        docs_dir:
            Directory containing the source markdown files.
        """
        chunks: List[Document] = self._loader.load(docs_dir)
        self._store.build(chunks)
        logger.info("Pipeline ready. %d chunks indexed.", len(chunks))

    def save_index(self, path: str | Path | None = None) -> None:
        """Persist the FAISS index to disk for later reuse."""
        self._store.save(path)

    def load_index(self, path: str | Path | None = None) -> None:
        """Load a previously saved FAISS index (skips re-embedding)."""
        self._store.load(path)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResult:
        """Answer *question* using the indexed documents.

        The pipeline:
        1. Embeds the question.
        2. Retrieves top-k chunks from the FAISS index.
        3. **Relevance gate**: if the best chunk score < ``score_threshold``,
           returns a canned "no context" response — the LLM is NOT called.
        4. Otherwise, calls the Groq LLM with the context and extracts citations.

        Parameters
        ----------
        question:
            The user's natural-language question.

        Returns
        -------
        RAGResult
            Structured result with ``answer``, ``citations``, and ``no_context`` flag.
        """
        if not self._store.is_ready:
            raise RuntimeError(
                "Pipeline is not initialised. Call build() or load_index() first."
            )

        # 1. Retrieve
        chunks = self._store.similarity_search_with_score(question, k=self._top_k)
        logger.debug(
            "Retrieved %d chunks. Best score: %.4f",
            len(chunks),
            chunks[0][1] if chunks else 0.0,
        )

        # 2. Relevance gate
        if not chunks or chunks[0][1] < self._score_threshold:
            logger.info(
                "Relevance gate fired (best score=%.4f < threshold=%.4f). "
                "Returning no-context response.",
                chunks[0][1] if chunks else 0.0,
                self._score_threshold,
            )
            return RAGResult(
                question=question,
                answer=NO_CONTEXT_RESPONSE,
                citations=[],
                no_context=True,
                retrieved_chunks=chunks,
            )

        # 3. Generate answer with citations
        answer, citations = self._chain.answer(question, chunks)

        return RAGResult(
            question=question,
            answer=answer,
            citations=citations,
            no_context=False,
            retrieved_chunks=chunks,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if the pipeline has a loaded/built index."""
        return self._store.is_ready
