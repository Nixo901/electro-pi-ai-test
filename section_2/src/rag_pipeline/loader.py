"""Document loading and chunking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads all Markdown files from a directory, splits them into chunks,
    and attaches rich metadata (source filename, chunk index) to each chunk.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            # Try to split on semantic boundaries first
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load(self, docs_dir: str | Path) -> List[Document]:
        """Load and chunk all *.md files under *docs_dir*.

        Returns
        -------
        List[Document]
            Each document has metadata keys:
            - ``source``: relative path to the source file (e.g. "rag_concepts.md")
            - ``chunk_index``: integer index of the chunk within its source file
        """
        docs_dir = Path(docs_dir).resolve()
        if not docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

        loader = DirectoryLoader(
            str(docs_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
            use_multithreading=False,
        )
        raw_docs: List[Document] = loader.load()
        logger.info("Loaded %d raw documents from %s", len(raw_docs), docs_dir)

        if not raw_docs:
            raise ValueError(f"No markdown files found in {docs_dir}")

        # Normalise source metadata to just the filename for readability
        for doc in raw_docs:
            src = Path(doc.metadata.get("source", "unknown"))
            doc.metadata["source"] = src.name

        chunks: List[Document] = self._splitter.split_documents(raw_docs)

        # Attach chunk index per source file
        source_counts: dict[str, int] = {}
        for chunk in chunks:
            src = chunk.metadata["source"]
            idx = source_counts.get(src, 0)
            chunk.metadata["chunk_index"] = idx
            source_counts[src] = idx + 1

        logger.info(
            "Split into %d chunks (size=%d, overlap=%d)",
            len(chunks),
            self._chunk_size,
            self._chunk_overlap,
        )
        return chunks
