# AI Engineer Technical Test — NOTES & Write-ups

This document compiles the half-page architectural write-ups and trade-offs assessments required for each section of the technical test.

---

## 🎙️ Section 1 — LiveKit Agents (Real-time Voice AI)

### 1. Barge-in / Interruption Handling
* **Mechanical Overview**: The LiveKit Agents framework manages interruptions via `allow_interruptions=True` on the `AgentSession`. When active user speech is detected by the Voice Activity Detector (VAD) while the agent is publishing audio, the framework triggers an interruption.
* **Production Extensions**:
  1. **Immediate Stream Cancellation**: Upon interruption, the agent immediately sends a cancellation signal to the active LLM generation stream and the Text-to-Speech (TTS) synthesizer, dropping any queued audio packets to clear the playback buffer.
  2. **VAD Optimization**: Standard VAD models (like Silero VAD) must be tuned to ignore short backchannel sounds (like "uh-huh", "yeah", or coughing) while instantly capturing real speech. We implement this by adjusting the speech detection threshold (typically `0.5` activation probability) and the minimum speech duration (e.g., `250ms`).
  3. **Turn-Taking Delays**: Calibrate the trailing silence threshold (`min_endpointing_delay=600ms`, `max_endpointing_delay=1500ms`) based on user conversational cadence. This ensures the agent does not cut off users who pause briefly to think.
  4. **Transcript Reconciliation**: Keep track of the exact word offset when the audio playback was cut off. This allows the conversation history to capture only what the user actually heard before interrupting, preventing the LLM context window from referencing statements the user never received.

### 2. Safely Adding a Second Tool (`cancel_order`)
To add write-level tools in production without risking security or execution errors, we implement:
* **Caller Authentication**: Never accept a spoken `order_id` in isolation. Bind the agent session to the user's authenticated session (via LiveKit room tokens containing dispatch metadata) and verify ownership of the target order.
* **Schema Validation**: Define rigid input schemas using JSON Schema or Pydantic. Ensure strict regex constraints (e.g., `pattern="^[0-9]{4,12}$"`) to filter out injection attacks or transcription noise before the tool code executes.
* **Explicit User Confirmation**: For destructive actions like cancellation, the agent must collect explicit confirmation (e.g., "Would you like me to go ahead and cancel order 1002?"). The tool should only execute if the conversation state flags a positive user confirmation.
* **Idempotency & Timeout Handling**: Attach a session-unique token to the back-end call to ensure network retries do not result in multiple cancellations. Wrap the execution in an explicit timeout wrapper (e.g., 3.0 seconds) and return a friendly error message (e.g., "I'm having trouble cancelling your order right now; let me transfer you to a human agent") if the backend times out.

### 3. Pipeline Component Swapping (Decoupling)
* **Design Decoupling**: The architecture isolates vendor SDK dependencies from the core agent business logic. This is achieved by creating protocol-based abstractions (`STTProvider`, `LLMProvider`, and `TTSProvider`).
* **Implementation Details**: The agent's pipeline components are wired at the composition root (`build_session` in `agent.py`). Swapping a provider is a one-line config change:
  ```python
  # Swap STT from Groq Whisper to Deepgram Nova-3:
  # Old: stt=groq.STT(model="whisper-large-v3-turbo")
  # New:
  stt = deepgram.STT(model="nova-3", language="ar")
  ```
  The agent class (`ArabicFoodDeliveryAgent`) and its `@function_tool` tools remain untouched, proving the decoupled design.

---

## 📚 Section 2 — LangChain (RAG Pipeline)

### 1. Chunking Strategies for Poor Answer Quality on Longer Documents
When document length increases, simple character-based splitters break semantic coherence, leading to poor retrieval. We would modify our chunking strategy as follows:
* **Semantic Chunking**: Instead of hard-coded splits, calculate moving averages of sentence embeddings and split documents only where the embedding similarity drifts significantly. This keeps logical arguments and context-rich paragraphs grouped together.
* **Hierarchical Chunking (Small-to-Big)**: Split documents into tiny leaf chunks (e.g., 100 tokens) for optimal vector matching, but link each leaf to a larger parent chunk (e.g., 512 tokens). When a leaf matches, we feed the larger parent chunk to the LLM. This yields precise matches while providing full context to the model.
* **Document Summary Embeddings**: For highly verbose files (e.g., PDFs), generate summaries for large sections, embed those summaries for retrieval, but return the full section contents as context.

### 2. Retrieval Upgrades
* **Dense + Sparse Hybrid Search**: Vector similarity (dense search) is excellent at capturing abstract concepts, but poor at matching exact keywords, system codes, or technical keywords (e.g., specific API function names). We combine dense vector search with a sparse keyword search (BM25) using a LangChain `EnsembleRetriever`, merging results using Reciprocal Rank Fusion (RRF).
* **Cross-Encoder Re-ranking**: Dense retrieval uses Bi-encoders, which compare vectors independently. We retrieve a larger pool of candidates (e.g., top-20) and re-rank them using a **Cross-Encoder model** (e.g., `ms-marco-MiniLM-L-6-v2`) which processes the query and chunk together to model token-level cross-attention. This drastically improves precision at the expense of minor latency overhead.
* **Query Expansion (HyDE)**: Generate a hypothetical answer from the query, embed that synthetic answer, and use its embedding to query the vector store. This aligns the query's vocabulary closer to the index documents.

---

## 📉 Section 3 — Quantization

### 1.AWQ/GPTQ vs. bitsandbytes vs. GGUF
For production deployment, the selection of quantization format depends on the serving hardware, latency constraints, and throughput requirements:

| Format | Quantization Method | Execution Engine | Best Production Use Case |
|---|---|---|---|
| **AWQ / GPTQ** | Post-Training (Static) | GPU-bound (vLLM, TGI, TensorRT-LLM) | **High-throughput GPU Serving Clusters** (Enterprise API services) |
| **bitsandbytes** | On-the-fly (Dynamic) | GPU-bound (PyTorch/HF default kernels) | **QLoRA Fine-Tuning** and rapid model evaluation |
| **GGUF** | Post-Training (Static) | CPU / Metal / Mixed (llama.cpp) | **Edge Devices, CPU Servers, & Local Dev Tooling** |

### 2. Selection Heuristics
* **Pick AWQ/GPTQ over bitsandbytes** when building an inference API. AWQ/GPTQ pre-quantize model weights using calibration datasets to generate static files. When loaded, they do not require dynamic runtime dequantization in the critical path. Highly optimized inference engines like vLLM have custom AWQ kernels. For small models like Qwen2.5-1.5B, this eliminates the CPU/GPU dequantization overhead which makes bitsandbytes slower than fp16.
* **Pick GGUF over both** when running on commodity CPU hardware, edge devices (e.g., local coding agents on developer MacBooks with unified memory), or when VRAM is extremely limited and you need partial GPU-to-CPU layers offloading. GGUF is a single-file format designed to work with C++ execution runtimes (like `llama.cpp`) with zero Python overhead.
* **Pick bitsandbytes** strictly for local development prototyping, QLoRA fine-tuning (where weight gradients must be calculated against quantized bases), or when you want zero-friction model loading without running a calibration pass.

---

## 🐳 Section 4 — Model Deployment

### 1. Scaling to 50+ Concurrent Users in Production
A single-worker FastAPI serving a model directly via PyTorch/HuggingFace is limited by sequential GPU execution, causing a queuing bottleneck under concurrency. To serve 50+ concurrent users with sub-second latency, we must implement:

```mermaid
flowchart TD
    U[50+ Concurrent Users] -->|HTTPS/WSS| LB[Envoy/NGINX L7 Load Balancer\nLeast-Request Strategy]
    LB -->|Replicas| K8S[Kubernetes Pod Cluster\nAuto-scaled via KEDA on Queue Metric]
    K8S --> API1[vLLM Inference Server Pod 1\nGPU 1]
    K8S --> API2[vLLM Inference Server Pod 2\nGPU 2]
    
    subgraph Inside Each Pod (vLLM)
        API1 --> CB[Continuous Batching\nIteration-Level Scheduler]
        CB --> PA[PagedAttention\nDynamic VRAM Manager]
        PA --> PC[Prefix / KV Cache\nSkip prompt evaluation]
    end
    
    U -.->|Exact Match| RC[Redis Semantic Cache\nBypass LLM via Vector Similarity]
    RC -.->|Hit| U
```

### 2. Architecture Scaling Blueprint
* **Continuous (Iteration-Level) Batching**: Standard servers process batch requests statically, waiting for the slowest sequence to finish. Continuous batching (offered by vLLM/TGI) schedules requests at the token-iteration level. As soon as a request emits a token, a waiting request joins the active batch. This increases GPU throughput by up to 30x.
* **Horizontal Pod Autoscaling with KEDA**: Scale container replicas horizontally inside Kubernetes. We use KEDA (Kubernetes Event-driven Autoscaling) to scale on custom Prometheus metrics scraped from our inference servers, targeting the number of pending/queued requests (`vllm:num_requests_waiting` > 5), rather than traditional CPU/RAM utilization.
* **Prefix / Prompt Caching**: For applications with long, repetitive prompts (like RAG system instructions or chat history), enabling prefix caching saves the computed Key-Value (KV) cache of prompt prefixes in VRAM. This reduces Time-To-First-Token (TTFT) by up to 90% for subsequent queries.
* **L7 Load Balancing & Request Queuing**: Put a Layer-7 load balancer (like Envoy or NGINX) in front of the cluster configured with a **least-request** load-balancing algorithm. Maintain a request buffer queue at the gateway level to absorb temporary traffic spikes without crashing individual pods.
