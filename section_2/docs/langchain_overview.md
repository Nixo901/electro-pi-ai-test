# LangChain Overview

## What Is LangChain?

LangChain is an open-source Python (and JavaScript) framework designed to simplify the construction of applications powered by large language models (LLMs). Rather than calling an LLM API directly and managing every prompt, context, and output parser by hand, LangChain provides composable building blocks — **chains**, **agents**, **tools**, **memory**, and **retrievers** — that developers can wire together using a declarative expression language.

The core value proposition is **composability**: each component exposes a uniform interface, so a retriever, an LLM, and a parser can be chained with a `|` operator, just like Unix pipes.

## LangChain Expression Language (LCEL)

LCEL is the modern way to compose LangChain components. It uses Python's bitwise-OR operator (`|`) to declare a data pipeline:

```python
chain = prompt | llm | output_parser
result = chain.invoke({"question": "What is FAISS?"})
```

Every LCEL component is a `Runnable`. Runnables support `.invoke()` for single calls, `.batch()` for parallel execution, and `.stream()` for token streaming. LCEL chains are automatically parallelised where possible and natively support async execution.

## Core Abstractions

### Chains
A chain is a sequence of steps. The simplest chain feeds a prompt template into an LLM and then into an output parser. More complex chains branch, loop, or fan out to multiple LLMs. LCEL replaces the older `LLMChain` class, though `LLMChain` still exists in `langchain.chains` for backward compatibility.

### Agents
An agent uses an LLM to decide *which tool to call next* based on the user's input and the results of prior tool calls. LangChain ships with several agent types:
- **ReAct** — Reason + Act: the LLM produces a Thought, chooses an Action, observes the result, and loops until it has a Final Answer.
- **Tool-calling agent** — uses the LLM's native function-calling API (OpenAI, Groq, Anthropic) to invoke tools structurally.
- **LangGraph agents** — agents built as explicit stateful graphs using the companion LangGraph library.

### Tools
A tool is any callable that an agent can invoke. Tools can wrap web search APIs, database queries, calculators, or custom business-logic functions. They are described by a name, a docstring, and an input schema so the LLM knows when and how to use them.

### Memory
Memory allows a chain or agent to persist information across turns. Short-term memory is often just the chat message history prepended to each new prompt. Long-term memory can use a vector store to retrieve relevant past exchanges.

### Retrievers
A retriever accepts a string query and returns a list of `Document` objects. Retrievers are the bridge between a user's question and a vector store or search engine. They are central to Retrieval-Augmented Generation (RAG) pipelines.

## LangGraph

LangGraph is a companion library for building **stateful, multi-actor** applications as directed acyclic graphs (or even cyclic graphs with conditional edges). It is the recommended approach for complex agents that require loops, branching on intermediate results, or human-in-the-loop steps. LangGraph applications are far easier to debug and test than imperative agent loops.

## Ecosystem Position

LangChain integrates with more than 100 LLM providers (OpenAI, Anthropic, Groq, Cohere, Mistral, Ollama, …), vector stores (FAISS, Chroma, Weaviate, Pinecone, pgvector, …), and document loaders (PDF, Word, Markdown, web pages, databases, …). Integration packages live in `langchain-community` and provider-specific packages like `langchain-openai` or `langchain-groq`.
