"""Integration tests for RAGPipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline.pipeline import RAGPipeline, RAGResult

DOCS_DIR = Path(__file__).parent.parent / "docs"


# ---------------------------------------------------------------------------
# Shared fixture — build the pipeline once per test module (expensive step)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    """Fully built RAGPipeline, reused across tests in this module."""
    pl = RAGPipeline()
    pl.build(DOCS_DIR)
    return pl


# ---------------------------------------------------------------------------
# No-context guard tests (no LLM call)
# ---------------------------------------------------------------------------

class TestNoContextGate:
    """Verify the relevance gate fires correctly and the LLM is NOT called."""

    def test_off_topic_query_triggers_no_context(self, pipeline: RAGPipeline):
        """A completely unrelated question should hit the no-context gate."""
        result = pipeline.query("What is the population of Tokyo in 2025?")
        assert result.no_context is True, (
            f"Expected no_context=True for off-topic query. Answer was: {result.answer[:100]}"
        )

    def test_no_context_response_is_non_empty(self, pipeline: RAGPipeline):
        result = pipeline.query("What is the GDP of the United States?")
        assert result.answer.strip(), "No-context answer should be a non-empty string."

    def test_no_context_citations_are_empty(self, pipeline: RAGPipeline):
        result = pipeline.query("Who won the FIFA World Cup in 2022?")
        # Gate may or may not fire for this; only check if it does fire
        if result.no_context:
            assert result.citations == [], "No-context result should have no citations."

    def test_low_threshold_bypasses_gate(self):
        """Setting threshold=0.0 should always pass the gate."""
        pl = RAGPipeline(score_threshold=0.0)
        pl.build(DOCS_DIR)
        # Even an off-topic query should reach the LLM (with a mocked LLM)
        with patch.object(pl._chain, "answer", return_value=("mocked answer", [])) as mock_ans:
            result = pl.query("What is the capital of France?")
            assert not result.no_context, "With threshold=0.0 gate should never fire."
            mock_ans.assert_called_once()


# ---------------------------------------------------------------------------
# Answer quality tests (LLM is called)
# ---------------------------------------------------------------------------

class TestAnswerQuality:
    """Verify that on-topic queries produce answers with citations."""

    def test_answer_is_non_empty_for_on_topic_query(self, pipeline: RAGPipeline):
        result = pipeline.query("What is FAISS and how does it work?")
        assert result.answer.strip(), "Answer should not be empty."

    def test_citations_present_for_on_topic_query(self, pipeline: RAGPipeline):
        result = pipeline.query(
            "What is the difference between FAISS and Chroma for a local prototype?"
        )
        if not result.no_context:
            assert len(result.citations) > 0, "On-topic answer should include citations."

    def test_citation_has_required_keys(self, pipeline: RAGPipeline):
        result = pipeline.query("Explain hybrid search and BM25.")
        if not result.no_context:
            for citation in result.citations:
                assert "source" in citation
                assert "score" in citation
                assert "excerpt" in citation
                assert citation["source"].endswith(".md")

    def test_result_is_rag_result_instance(self, pipeline: RAGPipeline):
        result = pipeline.query("What is cross-encoder re-ranking?")
        assert isinstance(result, RAGResult)

    def test_question_preserved_in_result(self, pipeline: RAGPipeline):
        q = "What is the RecursiveCharacterTextSplitter?"
        result = pipeline.query(q)
        assert result.question == q


# ---------------------------------------------------------------------------
# Pipeline state tests
# ---------------------------------------------------------------------------

class TestPipelineState:
    def test_not_ready_before_build(self):
        pl = RAGPipeline()
        assert not pl.is_ready

    def test_ready_after_build(self, pipeline: RAGPipeline):
        assert pipeline.is_ready

    def test_query_raises_before_build(self):
        pl = RAGPipeline()
        with pytest.raises(RuntimeError, match="not initialised"):
            pl.query("anything")

    def test_save_and_load_index(self, tmp_path: Path):
        pl = RAGPipeline()
        pl.build(DOCS_DIR)
        pl.save_index(tmp_path / "idx")

        pl2 = RAGPipeline()
        pl2.load_index(tmp_path / "idx")
        assert pl2.is_ready

        result = pl2.query("What is FAISS?")
        assert isinstance(result, RAGResult)
