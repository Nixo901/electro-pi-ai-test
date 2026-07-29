"""Embedding model wrapper."""

from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings

logger = logging.getLogger(__name__)


def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Return a HuggingFace embeddings instance.

    The model is downloaded to the HuggingFace cache on the first call and
    reused from disk thereafter. No API key is required.

    Parameters
    ----------
    model_name:
        Override the model from settings. Defaults to ``settings.embedding_model``
        (``all-MiniLM-L6-v2``).
    """
    model = model_name or settings.embedding_model
    logger.info("Loading embedding model: %s", model)
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity via dot product
    )
