# Retrieval-Augmented Generation (RAG) Concepts

## What Is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of *parametric* knowledge (what an LLM learned during training) with *non-parametric* knowledge (documents stored in an external corpus). Instead of relying solely on the model's frozen weights, a RAG pipeline:

1. Converts a user's question into a query embedding.
2. Searches a vector store for the most semantically similar document chunks.
3. Injects the retrieved chunks into the LLM prompt as *context*.
4. Asks the LLM to answer *only* from that context.

This approach dramatically reduces hallucinations for domain-specific questions, keeps answers up to date without retraining, and allows explicit citations back to source documents.

## Why RAG Instead of Fine-Tuning?

| | Fine-tuning | RAG |
|---|---|---|
| Knowledge update | Full retrain or LoRA | Update the document store |
| Cost | High (GPU hours) | Low (embedding + storage) |
| Citation | Not native | First-class |
| Hallucination risk | Reduced but not eliminated | Further reduced by grounding |
| Latency | Inference only | Retrieval + inference |

RAG is usually the right first choice when documents change frequently or when citations are required for trust and auditability.

## Document Chunking

Before embedding, source documents must be split into chunks small enough to fit in the LLM's context window alongside the prompt and still leave room for the answer. Key chunking parameters are:

- **Chunk size**: The maximum number of characters (or tokens) per chunk. Common values: 256–1024 tokens. Smaller chunks are more precise but may lose surrounding context; larger chunks provide more context but may dilute relevance.
- **Chunk overlap**: The number of characters repeated at the boundary of adjacent chunks. Overlap (typically 10–20 % of chunk size) prevents important sentences from being cut off at a boundary and lost.
- **Splitting strategy**: `RecursiveCharacterTextSplitter` tries to split on paragraph breaks, then sentence boundaries, then word boundaries, then characters — preserving semantic units as much as possible.

## Embedding Models

Each chunk is converted into a dense vector (embedding) that captures its semantic meaning. Embedding models commonly used with LangChain:

- **all-MiniLM-L6-v2** (sentence-transformers): 384-dimensional, ~90 MB, runs locally, excellent cost-quality trade-off for English text.
- **text-embedding-3-small** (OpenAI): 1536-dimensional, cloud, stronger multilingual performance.
- **nomic-embed-text** (Nomic): 768-dimensional, Apache 2.0, strong open-source alternative.

## The RAG Prompt Pattern

A standard RAG prompt has three sections:

```
System: You are a helpful assistant. Answer the user's question using ONLY
the context below. If the context does not contain enough information to
answer confidently, say "I don't have enough context to answer this question."
Cite the source of each claim.

Context:
[CHUNK 1 — source: docs/foo.md]
...text...

[CHUNK 2 — source: docs/bar.md]
...text...
