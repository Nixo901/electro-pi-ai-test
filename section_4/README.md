# Section 4 — Model Deployment: REST API & Dockerization

This section provides a containerized, production-ready inference API for **Qwen2.5-1.5B-Instruct** using **FastAPI** and **HuggingFace Transformers**.

---

## 🏗️ Architecture and Design Choices

### FastAPI vs. vLLM vs. TGI
For this deployment, **FastAPI + HuggingFace Transformers** was selected over high-throughput inference engines like vLLM or Text Generation Inference (TGI):

1. **Platform Compatibility**: vLLM has native, heavily-optimized custom CUDA kernels (`flash-attention`, custom AWQ kernels) that are built specifically for Linux. Running vLLM inside a Windows WSL2/Docker environment often fails or defaults to slow CPU fallbacks. FastAPI with HuggingFace runs seamlessly across both Windows (with direct CUDA execution) and Linux containers.
2. **Resource Efficiency**: Since we are serving a tiny 1.5B parameter model (which takes only ~3.0 GB of VRAM in fp16 precision), advanced paged attention mechanisms provided by vLLM are not critical for preventing Out-Of-Memory (OOM) errors. 
3. **Control and Customization**: FastAPI allows us to build custom lifecycles, structured error responses, and clean SSE streaming pipelines with minimal overhead.

---

## 📦 Project Structure

```
section_4/
├── Dockerfile                   # Builds runtime environment with CUDA, PyTorch and HuggingFace
├── docker-compose.yml           # Runs the server with GPU passthrough and cache volume
├── requirements.txt             # PyTorch, Transformers, FastAPI, and HTTPX packages
├── .env.example                 # Example environment configuration
├── app/
│   ├── __init__.py
│   ├── main.py                  # API endpoints definition (/health, /generate, /generate/stream)
│   ├── model.py                 # Singleton model & tokenizer loader
│   ├── schemas.py               # Request and response schema definitions
│   └── streaming.py             # Event-stream generator for chunked output
└── scripts/
    └── load_test.py             # Latency & concurrency load tester client
```

---

## 🚀 Setup & Execution

### Prerequisites
- Docker Desktop with **NVIDIA Container Toolkit** installed and configured (to allow GPU passthrough to Docker containers).
- Alternatively, Python 3.10+ and a CUDA-capable local environment.

### Run with Docker Compose (Recommended)

1. Make sure Docker is running.
2. From the `section_4` directory, run:
   ```bash
   docker compose up --build -d
   ```
   *Note: During build, the Dockerfile runs a script to pre-cache the model weights into the image (`/model_cache`). This ensures the container starts instantly when run.*

### Run directly with Python (Local fallback)

If you don't have Docker GPU support set up, you can run locally:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server (runs on port 8000 by default)
# On Windows PowerShell:
$env:MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
$env:DEVICE="cuda"
$env:QUANTIZATION="fp16"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📡 API Endpoints

### 1. Health check: `GET /health`
Verifies server is alive, queries GPU state, and lists VRAM utilization.
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_device": "cuda:0",
  "vram_allocated_mb": 3124.5
}
```

### 2. Generate response (JSON): `POST /generate`
Generates a complete response and returns latency telemetry.
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "max_new_tokens": 50}'
```
**Response:**
```json
{
  "text": "2 + 2 is equal to 4.",
  "tokens_generated": 8,
  "time_to_first_token_ms": 115.42,
  "total_latency_ms": 234.12,
  "tokens_per_sec": 34.17
}
```

### 3. Stream response (SSE): `POST /generate/stream`
Streams response chunks token-by-token using Server-Sent Events.
```bash
curl --no-buffer -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story about a little robot.", "max_new_tokens": 100}'
```
**Response (yields incrementally):**
```text
data: {"token": "Once"}

data: {"token": " upon"}

data: {"token": " a"}

...

data: [DONE]
```

---

## 📊 Latency & Load Testing

A load test client script is included to test the server under concurrency. It sends 10 concurrent requests to measure Time-To-First-Token (TTFT) and total roundtrip latency.

Run the load test:
```bash
python scripts/load_test.py --url http://localhost:8000 --concurrency 10 --tokens 100
```

### Load Test Results Example
Below are typical metrics from running the load test locally on an NVIDIA RTX 4050 GPU:

```
Total Requests: 10
Successes:      10
Failures:       0
Total Wall Time:15.43 seconds

Throughput & Token Metrics:
  - Total generated tokens:   950 tokens
  - Overall request rate:     0.65 req/sec
  - Combined generation rate: 61.57 tok/sec

Latency Percentiles:
  | Metric (ms)              | Average  | Min      | p50 (Med)| p95      | p99      | Max      |
  |--------------------------|----------|----------|----------|----------|----------|----------|
  | Client-Side Latency      |   8123.5 |    845.2 |   8120.4 |  15104.2 |  15423.8 |  15423.8 |
  | Server-Side TTFT (First) |   4120.4 |    112.4 |   4100.2 |   8124.5 |   8142.1 |   8142.1 |
  | Server-Side Gen Latency  |   7940.2 |    820.1 |   7950.4 |  14980.5 |  15220.1 |  15220.1 |
```

> **Why do latency percentiles scale up with concurrency?**
> Since we use a single GPU device mapped to a single Uvicorn server worker, requests are queued and processed sequentially. Request 1 executes immediately (low TTFT/latency), while Request 10 must wait in line for the preceding 9 requests to finish generation. In production, this can be optimized using **Continuous Batching** and **Tensor Parallelism** (e.g., via vLLM running on multi-GPU nodes).

---

## 📈 Write-up: Scaling to 50+ Concurrent Users in Production

Serving 50+ concurrent users with an LLM inference API requires moving beyond the naive FastAPI + single-worker HuggingFace queue structure. If deployed as-is, the 50th request would face catastrophic timeouts (latency > 60 seconds) due to head-of-line blocking on the single GPU queue. 

To achieve low-latency, high-concurrency production serving, the following changes are required:

### 1. Dynamic / Continuous Batching
Instead of waiting for one generation to finish entirely before starting the next (static batching), we must deploy a production inference engine like **vLLM** or **TGI**.
- **Continuous Batching (Iteration-Level Scheduling)**: Combines incoming requests at the token iteration step rather than the request sequence level. As soon as a request generates a token, a new request can join the active batch in the next iteration. This increases GPU tensor utilization from <10% to 90%+ and increases request throughput by 10x–30x without increasing VRAM significantly.
- **PagedAttention**: Resolves KV cache memory fragmentation by allocating virtual blocks of VRAM dynamically (like OS paging), enabling larger batch sizes.

### 2. Load Balancing and Horizontal Autoscaling (KEDA)
- **L7 Load Balancer**: Use **Envoy** or **NGINX** configured for HTTP/2 or WebSockets. Standard round-robin load balancing is insufficient because generation requests vary in runtime. Instead, use a **Least-Request** load balancing strategy.
- **Horizontal Pod Autoscaling (HPA)**: In Kubernetes, standard CPU/RAM metrics are poor indicators of LLM load. Deploy **KEDA** (Kubernetes Event-driven Autoscaling) using custom metrics scraped via Prometheus from the inference engine (e.g., `vllm:num_requests_waiting` or raw GPU VRAM/compute metrics via NVIDIA Triton metrics).
- **Scale-to-Zero**: For cost optimization during idle hours, scale down to zero replicas, using a cold-start buffer queue to handle initial requests.

### 3. KV Caching and Prompt Cache Layer
- **Prompt / Prefix Caching**: LLM prompts often share systemic prefixes (such as system instructions, context documents in RAG, or multi-turn chat history). vLLM's automatic prefix caching saves the KV cache of these prefixes in VRAM, skipping their evaluation phase. This reduces TTFT for subsequent prompts by up to 90%.
- **Semantic Cache (Redis)**: Implement a semantic caching layer in front of the API using vector embeddings. If a user asks a question semantically close to a recently answered query (using cosine similarity threshold > 0.95), serve the cached response directly, bypassing LLM inference entirely.

### 4. Async Task Queueing & Rate Limiting
- **Token Bucket Rate Limiting**: Implement rate-limiting at the gateway (e.g., Kong, FastAPI middleware) based on both Requests-Per-Minute (RPM) and Tokens-Per-Minute (TPM) to prevent resource exhaustion attacks.
- **Request Queueing**: During high-load spikes exceeding GPU capacity, queue requests at the proxy level (e.g., vLLM's built-in pending request queue) or use an async message broker (RabbitMQ/Celery) for non-interactive tasks, returning a task ID to the user for polling or webhook callback.

