# Retrieval Strategies for RAG

## Dense Retrieval (Semantic Search)

Dense retrieval is the standard approach used in most RAG pipelines. A query is converted into an embedding vector by the same model used to embed the document chunks, and the vector store returns the top-k chunks with the highest cosine (or dot-product) similarity.

**Strengths**: Handles paraphrases and synonyms naturally; language-agnostic if a multilingual embedding model is used.

**Weaknesses**: Can miss exact keyword matches; embedding models may not capture rare domain terminology well.

## Sparse Retrieval (Keyword / BM25)

BM25 (Best Match 25) is the classical term-frequency–inverse-document-frequency ranking function used by Elasticsearch and most traditional search engines. It scores documents by how often query terms appear relative to their frequency across the whole corpus.

**Strengths**: Excellent for exact-match queries, handles rare technical terms and model names precisely.

**Weaknesses**: No semantic understanding; "automobile" and "car" are completely different tokens.

## Hybrid Search

Hybrid search combines dense and sparse retrieval by scoring each document with both methods and merging the ranked lists. The most common merging strategy is **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
```

where k is a smoothing constant (typically 60) and rank_i is the document's rank in each individual ranked list. Documents that rank highly in *both* lists are boosted to the top.

**Strengths**: Best of both worlds — captures semantic meaning and exact keywords; robustly outperforms either method alone on heterogeneous query types.

**Weaviate**, **Elasticsearch**, and **Pinecone** support hybrid search natively. For FAISS + BM25, you can combine `BM25Retriever` from `langchain_community` with `FAISS` via `EnsembleRetriever`.

## Maximum Marginal Relevance (MMR)

MMR is a re-ranking technique that trades off relevance against diversity. After retrieving a pool of top-k candidates, MMR iteratively selects the next document that maximises:

```
MMR = argmax [ λ · sim(doc, query) − (1−λ) · max_{d ∈ selected} sim(doc, d) ]
```

Documents very similar to *already-selected* documents are penalised. This prevents the retrieved context from being filled with near-duplicate chunks.

LangChain exposes MMR via `vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=20)`.

## Cross-Encoder Re-ranking

A **cross-encoder** is a transformer model that takes a (query, document) pair as a *single concatenated input* and outputs a single relevance score. This is more accurate than comparing embeddings independently (bi-encoder) because the model sees both texts together and can model cross-attention between them.

Typical pipeline:
1. **Retrieve** top-20 candidates with fast dense/hybrid search.
2. **Re-rank** all 20 with a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
3. **Select** the top-4 re-ranked documents for the final prompt.

**Strengths**: Significantly improves precision, especially for longer documents and ambiguous queries.

**Weaknesses**: Cross-encoders are expensive — O(k) forward passes per query. Must be applied to a small candidate set retrieved by a faster first-stage ranker.

LangChain integration: `langchain.retrievers.ContextualCompressionRetriever` with a `CrossEncoderReranker` from `langchain_community`.

## Contextual Compression

Contextual compression retrieves chunks and then passes each through an LLM (or a smaller extractive model) to extract only the sentences relevant to the query, discarding filler content. This reduces noise in the context window.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
```

## Handling the "No Relevant Context" Case

A robust RAG pipeline must explicitly detect when no retrieved chunk is relevant enough to ground an answer. Common approaches:

1. **Score threshold**: Reject any retrieval result whose similarity score is below a tuned threshold (e.g., cosine similarity < 0.30). Return a canned "I don't have enough information" response without calling the LLM.
2. **LLM self-check**: Provide the chunks to the LLM and instruct it to say "I don't know" if the context is insufficient. Less reliable — the model may still hallucinate.
3. **Classifier**: A lightweight binary classifier trained on (query, chunks) pairs to predict whether the context is relevant. High accuracy, extra training cost.

The score-threshold approach is the most robust and lowest-latency option for most applications.

## Summary Comparison

| Strategy | Accuracy | Latency | Complexity |
|---|---|---|---|
| Dense only | Good | Low | Low |
| Sparse only (BM25) | Good for keywords | Low | Low |
| Hybrid (Dense + BM25) | Better | Low-Medium | Medium |
| + MMR | Better diversity | Low | Low |
| + Cross-encoder re-rank | Best | Medium | Medium |
| + Contextual compression | Best (noisy docs) | High | High |
