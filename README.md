# AI Engineer Technical Test — Electro Pi

This repository contains the practical deliverables for the Mid-Level AI Engineer technical assessment.

## 📂 Repository Structure

The project is structured into four sections, mapping to the key areas assessed:

| Folder | Section & Task | Core Technologies | Description |
|---|---|---|---|
| [**`section_1/`**](file:///c:/Users/nezar/Downloads/technical_test/section_1) | 1.1: LiveKit Voice Agent<br>1.2: Provider Swap (Bonus) | LiveKit Agents SDK, Groq (Whisper/GPT-OSS/Orpheus), Deepgram (Nova-3/Aura-2) | A real-time, multilingual (Arabic/English) voice assistant for a food delivery app with dynamic push-to-talk browser demo and order lookup tool calling. |
| [**`section_2/`**](file:///c:/Users/nezar/Downloads/technical_test/section_2) | 2.1: LangChain RAG Pipeline | LangChain, FAISS Vector Store, SentenceTransformers, Groq LLM | An offline and online RAG pipeline loaded with Markdown concept docs, featuring source citation formatting and double-layered hallucination guards. |
| [**`section_3/`**](file:///c:/Users/nezar/Downloads/technical_test/section_3) | 3.1: Model Quantization | PyTorch, Transformers, BitsAndBytes (NF4) | Subprocess-isolated benchmarking runner evaluating VRAM/RAM footprint and throughput (tok/s) of Qwen2.5-1.5B (fp16 vs 4-bit NF4) on a GPU. |
| [**`section_4/`**](file:///c:/Users/nezar/Downloads/technical_test/section_4) | 4.1: Model Serving & API | FastAPI, Uvicorn, Docker, Docker Compose, HTTPX | A GPU-accelerated Dockerized REST API serving Qwen2.5 with SSE token-by-token streaming, including an automated concurrent load test script. |

---

## 🛠️ Global Prerequisites

To run all sections of this test, you will need:
1. **Operating System**: Windows (or Linux/WSL2).
2. **Python**: Python 3.10 to 3.13. (Python 3.11 is recommended).
3. **Docker**: Docker Desktop with **NVIDIA Container Toolkit** installed (for GPU-accelerated Docker containers).
4. **Hardware**: An NVIDIA GPU (the benchmark and API are configured to run on CUDA; 6 GB VRAM is recommended).
5. **API Keys**:
   - **Groq API Key**: (Required for Section 1 STT/LLM/TTS, Section 2 LLM).
   - **Deepgram API Key**: (Optional, required for Section 1 English STT/TTS).

---

## 🚀 Quick Setup & Run Guide (All Sections in Under 10 Minutes)

Follow these steps to run each section:

### 🔑 Step 0: Set Up Environment Keys
Configure your environment keys at the root of Section 1 and Section 2:
- Copy `section_1/.env.example` to `section_1/.env` and add your `GROQ_API_KEY` (and optionally `DEEPGRAM_API_KEY`).
- Copy `section_2/.env.example` to `section_2/.env` and add your `GROQ_API_KEY`.

---

### 🎙️ Section 1: LiveKit Voice Agent
Runs a local LiveKit media server in Docker and registers a Python agent worker.

```powershell
# 1. Start local LiveKit server
cd section_1
docker compose up -d

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the Voice Agent worker
python -m arabic_voice_agent.agent dev

# 4. In a separate terminal, start the web demo server
python scripts/demo_server.py
# Open http://127.0.0.1:8000 in your browser, connect, and talk!
```

---

### 📚 Section 2: LangChain RAG Pipeline
Builds a local FAISS index from Markdown files and retrieves answers with semantic source citations.

```powershell
cd section_2
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the executable example questions
python scripts/run_examples.py
```

---

### 📉 Section 3: Model Quantization
Measures memory (VRAM/RAM), speed (tokens/sec), and qualitative trade-offs between `fp16` and `4-bit NF4` precisions.

```powershell
cd section_3
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the isolated subprocess benchmark orchestration script
python benchmark.py
```

---

### 🐳 Section 4: Model Serving & REST API
Deploys the served Qwen model behind a Dockerized FastAPI API with SSE streaming support and runs load tests.

```powershell
cd section_4
# 1. Build and start the GPU-accelerated container
docker compose up --build -d

# 2. Run the concurrent load/latency test script (verifies streaming + sends 10 concurrent requests)
python scripts/load_test.py --url http://localhost:8000 --concurrency 10 --tokens 100
```

---

## 📝 Trade-offs and Key Technical Write-ups

All requested half-page architectural write-ups and trade-offs evaluations are compiled in a single consolidated document at the root of the project:
* 📄 [**`NOTES.md`**](file:///c:/Users/nezar/Downloads/technical_test/NOTES.md) (Unified top-level answers covering all sections)

In addition, each section contains context-specific write-ups in its own folder:
* **Section 1**: Detail of barge-in configuration, VAD endpointing, and second-tool design safety is in [**`section_1/NOTES.md`**](file:///c:/Users/nezar/Downloads/technical_test/section_1/NOTES.md).
* **Section 2**: Detailed proposals on chunking strategy adjustments (semantic, summary, hierarchical chunking) and retrieval upgrades (hybrid search, cross-encoder re-ranking) are in [**`section_2/README.md#L200-L230`**](file:///c:/Users/nezar/Downloads/technical_test/section_2/README.md#L200-L230).
* **Section 3**: Comparison of runtime bitsandbytes (NF4) against AWQ/GPTQ and GGUF is in [**`section_3/README.md#L118-L190`**](file:///c:/Users/nezar/Downloads/technical_test/section_3/README.md#L118-L190).
* **Section 4**: Justification of FastAPI vs vLLM/TGI and architectural designs for scaling to 50+ concurrent users are in [**`section_4/README.md#L161-L188`**](file:///c:/Users/nezar/Downloads/technical_test/section_4/README.md#L161-L188).
