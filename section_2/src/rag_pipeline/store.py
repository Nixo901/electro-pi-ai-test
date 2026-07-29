"""FAISS vector store manager — build, save, load, and query."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages a FAISS vector store: build from documents, persist to disk,
    load from disk, and run similarity search with scores.
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings) -> None:
        self._embeddings = embeddings
        self._store: FAISS | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Document]) -> None:
        """Create a new FAISS index from *chunks* and hold it in memory."""
        if not chunks:
            raise ValueError("Cannot build a vector store from an empty document list.")
        logger.info("Building FAISS index from %d chunks …", len(chunks))
        self._store = FAISS.from_documents(chunks, self._embeddings)
        logger.info("FAISS index built successfully.")

    # ------------------------------------------------------------------
    # Persist / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Serialise the FAISS index and document store to *path*."""
        self._require_store()
        save_path = Path(path or settings.faiss_index_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(save_path))  # type: ignore[union-attr]
        logger.info("FAISS index saved to %s", save_path)

    def load(self, path: str | Path | None = None) -> None:
        """Load a previously saved FAISS index from *path*."""
        load_path = Path(path or settings.faiss_index_path)
        if not load_path.exists():
            raise FileNotFoundError(
                f"FAISS index directory not found: {load_path}. "
                "Run pipeline.build() first."
            )
        self._store = FAISS.load_local(
            str(load_path),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded from %s", load_path)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def similarity_search_with_score(
        self, query: str, k: int | None = None
    ) -> List[Tuple[Document, float]]:
        """Return top-*k* (document, score) pairs for *query*.

        Scores are L2 distances converted to cosine-style similarities
        (higher = more similar) because we normalise embeddings to unit length,
        so L2 distance d and cosine similarity s relate as: s = 1 - d²/2.
        """
        self._require_store()
        k = k or settings.top_k
        results = self._store.similarity_search_with_score(query, k=k)  # type: ignore[union-attr]

        # FAISS returns L2 distances (lower = more similar).
        # Convert to a [0, 1] similarity score so callers don't need to know this detail.
        converted: List[Tuple[Document, float]] = []
        for doc, l2_dist in results:
            # With unit-normalised vectors: cosine_sim = 1 - (l2²/2)
            # Clamp to [0, 1] to handle floating-point edge cases.
            sim = max(0.0, min(1.0, 1.0 - (l2_dist**2) / 2.0))
            converted.append((doc, sim))

        return converted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_store(self) -> None:
        if self._store is None:
            raise RuntimeError(
                "Vector store is not initialised. Call build() or load() first."
            )

    @property
    def is_ready(self) -> bool:
        """True if the store has been built or loaded."""
        return self._store is not None
