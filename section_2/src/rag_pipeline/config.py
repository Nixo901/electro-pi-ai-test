"""Configuration for the RAG pipeline, loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunable knobs for the pipeline, overridable via environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- LLM ----------
    groq_api_key: str = Field(..., description="Groq API key (required).")
    groq_model: str = Field(
        "llama-3.3-70b-versatile",
        description="Groq model ID to use for answer generation.",
    )
    groq_temperature: float = Field(
        0.0, description="LLM sampling temperature (0 = deterministic)."
    )

    # ---------- Embedding ----------
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model for embeddings.",
    )

    # ---------- Chunking ----------
    chunk_size: int = Field(
        400, description="Maximum characters per document chunk."
    )
    chunk_overlap: int = Field(
        80, description="Character overlap between adjacent chunks."
    )

    # ---------- Retrieval ----------
    top_k: int = Field(4, description="Number of chunks to retrieve per query.")
    score_threshold: float = Field(
        0.25,
        description=(
            "Minimum cosine similarity score for a chunk to be considered relevant. "
            "Queries whose best chunk score is below this threshold receive the "
            "no-context response without calling the LLM."
        ),
    )

    # ---------- Storage ----------
    faiss_index_path: Path = Field(
        Path("faiss_index"),
        description="Directory where the FAISS index is saved / loaded.",
    )


# Module-level singleton — importable as `from rag_pipeline.config import settings`
settings = Settings()  # type: ignore[call-arg]
