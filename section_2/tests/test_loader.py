"""Tests for DocumentLoader."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow tests to import the package without installing it
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline.loader import DocumentLoader

DOCS_DIR = Path(__file__).parent.parent / "docs"


def test_load_returns_chunks():
    """Loader should produce at least one chunk per markdown file."""
    loader = DocumentLoader(chunk_size=400, chunk_overlap=80)
    chunks = loader.load(DOCS_DIR)
    assert len(chunks) > 0, "No chunks produced — check docs/ directory."


def test_all_chunks_have_source_metadata():
    """Every chunk must carry a 'source' metadata key with a .md filename."""
    loader = DocumentLoader(chunk_size=400, chunk_overlap=80)
    chunks = loader.load(DOCS_DIR)
    for chunk in chunks:
        assert "source" in chunk.metadata, f"Chunk missing 'source': {chunk}"
        assert chunk.metadata["source"].endswith(".md"), (
            f"Expected .md source, got: {chunk.metadata['source']}"
        )


def test_all_chunks_have_chunk_index_metadata():
    """Every chunk must carry a non-negative 'chunk_index' metadata key."""
    loader = DocumentLoader(chunk_size=400, chunk_overlap=80)
    chunks = loader.load(DOCS_DIR)
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["chunk_index"] >= 0


def test_chunks_cover_all_source_files():
    """Chunks from all four expected source files should be present."""
    loader = DocumentLoader(chunk_size=400, chunk_overlap=80)
    chunks = loader.load(DOCS_DIR)
    sources = {c.metadata["source"] for c in chunks}
    expected = {
        "langchain_overview.md",
        "rag_concepts.md",
        "vector_stores.md",
        "retrieval_strategies.md",
    }
    assert expected.issubset(sources), (
        f"Missing sources: {expected - sources}. Found: {sources}"
    )


def test_loader_raises_on_missing_directory():
    """Loader should raise FileNotFoundError for a non-existent directory."""
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/non/existent/path/to/docs")


def test_chunk_content_is_non_empty():
    """Each chunk should contain non-whitespace text."""
    loader = DocumentLoader(chunk_size=400, chunk_overlap=80)
    chunks = loader.load(DOCS_DIR)
    for chunk in chunks:
        assert chunk.page_content.strip(), "Empty chunk found."
