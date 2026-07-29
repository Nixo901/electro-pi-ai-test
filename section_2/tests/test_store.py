"""Tests for VectorStoreManager."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline.embedder import get_embeddings
from rag_pipeline.loader import DocumentLoader
from rag_pipeline.store import VectorStoreManager

DOCS_DIR = Path(__file__).parent.parent / "docs"


@pytest.fixture(scope="module")
def store() -> VectorStoreManager:
    """Build a store once and share it across tests in this module."""
    loader = DocumentLoader(chunk_size=400, chunk_overlap=80)
    chunks = loader.load(DOCS_DIR)
    embeddings = get_embeddings()
    mgr = VectorStoreManager(embeddings)
    mgr.build(chunks)
    return mgr


def test_store_is_ready_after_build(store: VectorStoreManager):
    assert store.is_ready


def test_similarity_search_returns_results(store: VectorStoreManager):
    results = store.similarity_search_with_score("What is FAISS?", k=3)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"


def test_similarity_search_scores_in_range(store: VectorStoreManager):
    """Converted similarity scores must be in [0, 1]."""
    results = store.similarity_search_with_score("vector store comparison", k=4)
    for doc, score in results:
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


def test_top_result_is_relevant_for_faiss_query(store: VectorStoreManager):
    """The top result for 'FAISS in-process library' should be from vector_stores.md."""
    results = store.similarity_search_with_score("FAISS in-process library no server", k=1)
    assert results, "No results returned."
    top_doc, top_score = results[0]
    assert top_doc.metadata["source"] == "vector_stores.md", (
        f"Expected vector_stores.md, got {top_doc.metadata['source']} (score={top_score:.4f})"
    )


def test_store_raises_before_build():
    """Querying an uninitialised store should raise RuntimeError."""
    embeddings = get_embeddings()
    mgr = VectorStoreManager(embeddings)
    with pytest.raises(RuntimeError, match="not initialised"):
        mgr.similarity_search_with_score("anything", k=1)


def test_save_and_load_roundtrip(store: VectorStoreManager, tmp_path: Path):
    """A saved index should produce identical results after reloading."""
    store.save(tmp_path / "idx")

    embeddings = get_embeddings()
    loaded = VectorStoreManager(embeddings)
    loaded.load(tmp_path / "idx")

    q = "cross-encoder re-ranking"
    original_results = store.similarity_search_with_score(q, k=2)
    loaded_results = loaded.similarity_search_with_score(q, k=2)

    assert len(original_results) == len(loaded_results)
    for (orig_doc, orig_score), (load_doc, load_score) in zip(
        original_results, loaded_results
    ):
        assert orig_doc.page_content == load_doc.page_content
        assert abs(orig_score - load_score) < 1e-4
