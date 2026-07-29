# Vector Stores for RAG

## What Is a Vector Store?

A vector store (also called a vector database) is a data store optimised for storing high-dimensional embedding vectors and performing fast approximate nearest-neighbour (ANN) search. In a RAG pipeline, the vector store answers the question: *"Which of my document chunks are most semantically similar to the user's query?"*

The quality of retrieval directly determines the quality of the final answer. Choosing the right vector store involves trade-offs between speed, accuracy, scalability, infrastructure complexity, and cost.

## FAISS

**Facebook AI Similarity Search (FAISS)** is an open-source C++ library with a Python wrapper, developed by Meta AI Research.

- **Deployment**: In-process library; no server, no network calls.
- **Storage**: In memory (with optional serialisation to disk via `faiss.write_index` / `faiss.read_index`).
- **Index types**: Flat (exact brute force), IVF (inverted file — approximate, clusters vectors), HNSW (hierarchical navigable small world — fast ANN graph).
- **Strengths**: Extremely fast for corpora up to ~1 M vectors, zero infrastructure, deterministic for Flat indices, easy to embed in a Python process.
- **Limitations**: No built-in metadata filtering, no persistence layer, no client-server architecture, not suitable for multi-process write-heavy workloads.
- **Best for**: Local prototypes, research, single-process applications, CI/CD pipelines where simplicity is paramount.

LangChain integration: `langchain_community.vectorstores.FAISS`.

## Chroma

**Chroma** is an open-source, Python-native vector database designed for AI applications.

- **Deployment**: Embedded (in-process, SQLite backend) or client-server (persistent HTTP server).
- **Storage**: Persistent SQLite + parquet files by default; no extra setup for embedded mode.
- **Strengths**: Built-in metadata filtering (`where` clauses on document metadata), persistent by default, easy to run as a local server, first-class LangChain and LlamaIndex support.
- **Limitations**: Slower than FAISS for pure ANN search on large corpora; embedded mode is single-writer.
- **Best for**: Projects that need metadata filtering (e.g., "only search documents tagged `category=finance`") or easy persistence without manual serialisation.

LangChain integration: `langchain_community.vectorstores.Chroma`.

## Weaviate

**Weaviate** is an open-source, production-grade vector database with a GraphQL API.

- **Deployment**: Docker/Kubernetes (self-hosted) or Weaviate Cloud (managed SaaS).
- **Strengths**: Hybrid search (dense + BM25 sparse) built-in, multi-tenancy, RBAC, rich schema with cross-references, real-time updates.
- **Limitations**: Requires running a separate server process; steeper learning curve.
- **Best for**: Production systems that need hybrid search, multi-tenant isolation, or high-availability deployments.

## Pinecone

**Pinecone** is a fully managed, serverless vector database.

- **Deployment**: Cloud SaaS only (AWS/GCP/Azure hosted).
- **Strengths**: Zero infrastructure management, automatic scaling, namespaces for multi-tenancy, metadata filtering, real-time upserts.
- **Limitations**: Paid service, data leaves your infrastructure, network latency on each query.
- **Best for**: Teams that want a production-grade vector store without DevOps overhead and are comfortable with a managed cloud service.

## pgvector

**pgvector** is a PostgreSQL extension that adds vector similarity search directly to Postgres.

- **Strengths**: Reuses existing Postgres infrastructure, SQL joins on vector results, ACID transactions, proven operational tooling.
- **Limitations**: Slower ANN than specialised stores at very large scale without careful index tuning.
- **Best for**: Teams already running Postgres who want to add RAG without introducing a new infrastructure component.

## Comparison Summary

| | FAISS | Chroma | Weaviate | Pinecone | pgvector |
|---|---|---|---|---|---|
| Infrastructure | None | None / Docker | Docker / Cloud | Cloud only | Postgres |
| Persistence | Manual | Automatic | Automatic | Automatic | Automatic |
| Metadata filter | No | Yes | Yes | Yes | Yes (SQL) |
| Hybrid search | No | No | Yes (built-in) | Yes (sparse) | Partial |
| Scale | ~1 M vecs | ~10 M vecs | 100 M+ vecs | Unlimited | Moderate |
| Cost | Free | Free (OSS) | Free (OSS) + Cloud | Paid | Free (OSS) |
