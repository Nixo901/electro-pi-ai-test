# Section 3 — Task 3.1: Quantize a Model and Measure the Trade-off

A reproducible benchmark comparing **Qwen2.5-1.5B-Instruct** at full fp16
precision versus **4-bit NF4** quantization via `bitsandbytes`, run locally on
an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM).

---

## Hardware & Software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop |
| VRAM | 6 GB |
| CUDA | 13.3 (driver 610.62) |
| Python | 3.13.5 |
| PyTorch | 2.12.0 |
| transformers | 4.57.3 |
| bitsandbytes | 0.50.0 |
| accelerate | 1.12.0 |
| OS | Windows 11 |

---

## Model

**`Qwen/Qwen2.5-1.5B-Instruct`** — Alibaba's 1.5-billion-parameter
instruction-tuned model, Apache-2.0 licence, no gated access required.

Chosen because:
- Fits entirely in 6 GB VRAM at fp16 (~3 GB) and at 4-bit NF4 (~1 GB)
- Apache-2.0 licence allows unrestricted local use
- Strong quality/size ratio — competitive with larger models on instruction following
- Small enough to complete the benchmark in under 15 minutes on a laptop GPU

---

## Methodology

### Prompts (fixed, identical for both runs)

| ID | Category | Prompt |
|---|---|---|
| P1 | Technical/CS | Explain the difference between a process and a thread in one paragraph. |
| P2 | Coding | Write a Python function that returns the nth Fibonacci number using recursion. |
| P3 | Factual | What is the capital of Australia and why is it not Sydney? |
| P4 | Creative | Write the opening line of a mystery novel set in 1920s Cairo. |
| P5 | Summarization | Summarize the main idea of Einstein's theory of special relativity in two sentences. |

### Measurement procedure

For each precision mode (fp16, then 4-bit NF4):

1. `gc.collect()` + `torch.cuda.empty_cache()` to reset VRAM baseline
2. Record `torch.cuda.memory_allocated()` before loading (baseline)
3. Load model via `AutoModelForCausalLM.from_pretrained(..., device_map="cuda")`
4. Record `torch.cuda.memory_allocated()` after loading → **VRAM delta = model footprint**
5. Warm-up: one generation pass of 10 tokens (not timed)
6. For each of 5 prompts: generate 200 tokens with **greedy decoding** (`do_sample=False`)
   - `torch.cuda.synchronize()` before and after to get accurate wall-clock time
   - `tok/s = 200 / elapsed_seconds`
7. Average throughput over all 5 prompts
8. Unload model, repeat for next precision

Greedy decoding is used for full reproducibility — identical outputs across
runs, removing sampling randomness as a confound.

---

## Results

The benchmark was executed successfully in isolated subprocesses on the NVIDIA GeForce RTX 4050 Laptop GPU, evaluating two different model sizes: **Qwen2.5-1.5B** and **Qwen2.5-0.5B**.

### Trade-off Summary Table

| Model | Precision | VRAM (model) | RAM delta | Avg Throughput | Relative Speed | Qualitative Quality |
|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B** | **fp16** | 2944 MB | 369 MB | 19.3 tok/s | 1.00x (ref) | **Baseline** (Highly accurate & coherent) |
| **Qwen2.5-1.5B** | **4-bit NF4** | 1158 MB | 456 MB | 15.5 tok/s | 0.80x | **Near-identical** (1 minor terminology error) |
| **Qwen2.5-0.5B** | **fp16** | 942 MB | 349 MB | 29.2 tok/s | 1.51x | **Baseline** (Coherent, minor factual errors) |
| **Qwen2.5-0.5B** | **4-bit NF4** | 452 MB | 434 MB | 23.2 tok/s | 1.20x (0.79x ref) | **Degraded** (Severe factual hallucinations) |

### Key Observations

1. **VRAM Footprint**: 
   - **Qwen2.5-1.5B** saw a VRAM reduction of **2.54x** (saving 1.78 GB), allowing it to fit into ~1.16 GB of VRAM.
   - **Qwen2.5-0.5B** saw a VRAM reduction of **2.09x** (saving 490 MB), loading at a tiny **452 MB**.
2. **Speed Overhead (Compute-Bound Quantization)**:
   - For both model sizes, `bitsandbytes` 4-bit NF4 introduced a nearly identical speed overhead of **~20%** compared to its corresponding fp16 baseline (**0.80x** for 1.5B, **0.79x** for 0.5B).
   - Because these models are very small, the GPU's memory bandwidth is not the bottleneck on the RTX 4050. The CPU/GPU overhead of dequantizing weights back to fp16 at runtime dominates, resulting in a net slowdown rather than a speedup.

---

## Qualitative Output Comparison

Raw outputs are saved in [results_qwen_qwen2.5-1.5b-instruct.json](file:///c:/Users/nezar/Downloads/technical_test/section_3/results/results_qwen_qwen2.5-1.5b-instruct.json) and [results_qwen_qwen2.5-0.5b-instruct.json](file:///c:/Users/nezar/Downloads/technical_test/section_3/results/results_qwen_qwen2.5-0.5b-instruct.json).

### 1. Qwen2.5-1.5B-Instruct — Comparison Table

| Prompt ID & Category | Prompt | Qwen2.5-1.5B (fp16) Response | Qwen2.5-1.5B (4-bit NF4) Response | Qualitative Analysis & Differences |
| :--- | :--- | :--- | :--- | :--- |
| **P1**<br>Technical / CS | Explain the difference between a process and a thread in one paragraph. | "A process is an instance of a program that is running on a computer system. It has its own memory space, resources, and can execute concurrently with other processes. A thread, on the other hand, is a lightweight process within a single process. Threads share the same memory space as the parent process and can run concurrently with each other within the same process. Threads allow for more efficient use of CPU resources and enable parallel execution of tasks." | "A process is an independent unit of execution that can be thought of as a program running on a computer system. It consists of multiple threads, which are lightweight processes that share resources with each other. Threads allow for concurrent execution of code within a single process, enabling more efficient use of CPU resources. In contrast, a thread is a single instance of a process that executes independently of others. **Threads have their own memory space**, but they share the same address space with other threads within the same process. Threads provide better resource management and scheduling compared to processes, allowing for more efficient utilization of system resources." | **fp16**: 100% accurate and technically sound.<br><br>**4-bit NF4**: Mostly coherent but introduced a factual error by claiming threads *"have their own memory space"* which directly contradicts the later statement that they *"share the same address space"*. |
| **P2**<br>Coding | Write a Python function that returns the nth Fibonacci number using recursion. | "Certainly! The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones...<br>```python\ndef fibonacci(n):\n    if n <= 0:\n        return \"Input should be a positive integer.\"\n    elif n == 1:\n        return 0\n    elif n == 2:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n```" | "Certainly! The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones...<br>```python\ndef fibonacci(n):\n    if n <= 0:\n        return \"Input should be a positive integer.\"\n    elif n == 1:\n        return 0\n    elif n == 2:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n```" | **Identical & Correct** recursive implementations generated by both versions. Both correctly handled base cases and input verification, showing no regression in basic code generation capability due to quantization. |
| **P3**<br>Factual | What is the capital of Australia and why is it not Sydney? | "The capital city of Australia is Canberra... Sydney: While Sydney is indeed one of the largest cities in Australia, it is not its capital. **The decision to make Canberra the capital was made after World War II due to strategic reasons related to defense and security**..." | "The capital of Australia is Canberra. It's not Sydney because Canberra was established as the national capital in 1908 to serve as the political center... **The decision to move the capital from Melbourne (which had been the capital since 1850) was made due to concerns about the growing population**..." | **Both**: Correctly named Canberra and ruled out Sydney.<br><br>**Both**: Hallucinated historical details. The fp16 version falsely claimed Canberra was chosen *after WWII* for strategic defense, while the NF4 version falsely claimed Melbourne was capital *since 1850* and that Sydney was a port. (Canberra was established under the Seat of Government Act 1908). |
| **P4**<br>Creative | Write the opening line of a mystery novel set in 1920s Cairo. | `"In the sultry heat of Cairo's golden hour, where whispers of secrets and opulence danced on the breeze, there lived an old man with eyes that saw beyond the veil."` | `"In the sultry embrace of Cairo's ancient streets on a sweltering summer evening, a young British diplomat finds himself inexplicably locked in an abandoned mansion with no memory of how he got there or why."` | **Both**: Excellent, atmospheric, and highly creative opening hooks. The quantized NF4 version generated a slightly stronger mystery plot hook (the locked mansion) compared to the descriptive fp16 output. |
| **P5**<br>Summarization | Summarize the main idea of Einstein's theory of special relativity in two sentences. | "Einstein's theory of special relativity introduced the principles that the laws of physics are the same for all observers moving at constant velocity relative to one another and that the speed of light is constant for all observers, regardless of their motion or the motion of the source emitting the light." | "Einstein's theory of special relativity introduced the principles that the speed of light is constant and independent of the observer's motion, and time dilation occurs when objects move at high speeds relative to each other." | **Both**: Excellent, scientifically accurate summaries that strictly followed the two-sentence constraint. The NF4 output explicitly mentioned *time dilation*, which is a strong addition. |

### 2. Qwen2.5-0.5B-Instruct — Comparison Table

| Prompt ID & Category | Prompt | Qwen2.5-0.5B (fp16) Response | Qwen2.5-0.5B (4-bit NF4) Response | Qualitative Analysis & Differences |
| :--- | :--- | :--- | :--- | :--- |
| **P1**<br>Technical / CS | Explain the difference between a process and a thread in one paragraph. | "A process is an instance of a program... It consists of all the code executed... A process can be thought of as a collection of threads... Threads... share some common resources with each other..." | "A process is a collection of resources (such as memory, CPU time, and file handles)... A thread is a single unit of execution... Threads share common data structures called mutexes, which allow threads to coordinate... without contention..." | **fp16**: Mostly correct, though circularly defined process as collection of threads.<br><br>**4-bit NF4**: Confused definitions, claiming mutexes allow coordination "without contention" and circular process/thread definitions. |
| **P2**<br>Coding | Write a Python function that returns the nth Fibonacci number using recursion. | Standard recursive Python function with base cases (`if n <= 0: ... elif n == 1: return 0 ...`). | Standard recursive Python function with cleaner base case (`if n <= 1: return n`). | **Both**: Correct. The quantized 0.5B model generated a slightly cleaner base case structure than its fp16 counterpart. |
| **P3**<br>Factual | What is the capital of Australia and why is it not Sydney? | Named Canberra. Hallucinated that the ACT was formed in 1986 and renamed CCR. | **Catastrophic Hallucination**: "The capital of Australia is Canberra, **which is located in the Northern Territory**... established in 1901 after the British had to relocate from Sydney **due to the outbreak of war with Germany**... **it is now considered the second most populous city in the world, behind Tokyo**." | **4-bit NF4**: Suffered severe degradation of factual weights, leading to extreme hallucinations (Canberra in Northern Territory, war with Germany in 1901, and Canberra as a megacity behind Tokyo). |
| **P4**<br>Creative | Write the opening line of a mystery novel set in 1920s Cairo. | "In the heart of ancient Egypt lay the bustling city of Cairo, where the whispers of history... Nile River flowed gently..." | "In the heart of ancient Egypt, where the pyramids stand as symbols of the past and present... secrets, intrigue, and the dark undercurrents of the Egyptian elite..." | **Both**: Coherent, but relatively generic and wordy compared to the 1.5B model's output. |
| **P5**<br>Summarization | Summarize the main idea of Einstein's theory of special relativity in two sentences. | Summarized relativity of motion. | Summarized relativity of motion. | **Both**: Coherent, but extremely generic and lacked technical depth. |

### ⚠️ The Critical Lesson on Quantization Scaling
Quantizing models below 1B parameters is highly risky. While a 1.5B model degrades gracefully (minor terminology error), a 0.5B model undergoes severe degradation of its factual weights, leading to extreme hallucinations (Canberra as a megacity). For production, **never quantize models below 1B parameters** unless they are exclusively fine-tuned for a narrow task. Use full precision for sub-1B models, as they already fit easily in VRAM (942 MB).

---

## Write-up: When to Use GPTQ/AWQ vs bitsandbytes vs GGUF in Production

### The one-sentence version

Use **GGUF** when you need CPU fallback or wide compatibility; use **GPTQ/AWQ**
when you need the highest quality at a fixed bit-width for GPU serving; use
**bitsandbytes** when you need zero-friction experimentation inside the
HuggingFace ecosystem.

### bitsandbytes (NF4 / INT8)

`bitsandbytes` quantizes weights on-the-fly at load time. There is no separate
calibration step — you swap in a `BitsAndBytesConfig` and you are done. That
makes it by far the fastest path from "full-precision model" to "running
quantized model", which is why it is the standard tool for fine-tuning with QLoRA.

The downside is that it is **compute-bound**, not memory-bandwidth-bound. Every
dequantization happens at inference time, which costs GPU cycles. On a small
RTX 4050, this cost is noticeable: NF4 throughput is typically *slower* than
fp16 for small models because the GPU is fast enough that dequantization
overhead dominates the memory savings. bitsandbytes shines at larger model
scales (≥7B) where VRAM capacity is the hard constraint, not compute.

I would pick bitsandbytes in production **only** for fine-tuning (QLoRA) or for
serving very large models (≥70B) where even GPTQ/AWQ cannot fit the model on
available hardware without it. For pure inference I would not choose it.

### GPTQ / AWQ

Both GPTQ and AWQ are **post-training quantization** methods that run a one-time
calibration pass over a small dataset, then bake the optimal quantized weights
into a new checkpoint. The result is a static file that can be served without
any on-the-fly dequantization overhead.

- **GPTQ** (Frantar et al., 2022) minimises layer-wise reconstruction error using
  the inverse Hessian — accurate but slow to quantize (hours for a 70B model).
- **AWQ** (Lin et al., 2023) observes that only ~1% of weights are salient and
  scales activations to protect them — quantizes faster than GPTQ and often
  achieves slightly better perplexity at 4-bit.

For a production GPU inference service (vLLM, TGI), I would choose **AWQ**
because: (a) the quantized checkpoint loads like a normal model, (b) vLLM has
first-class AWQ support with kernel-level optimizations, and (c) throughput is
5–20% higher than bitsandbytes NF4 at the same bit-width because there is no
runtime dequantization kernel in the critical path.

### GGUF (llama.cpp)

GGUF is the format used by `llama.cpp` and its ecosystem (Ollama, LM Studio,
etc.). It supports mixed-precision quantization (Q4_K_M, Q5_K_M, Q8_0, …) and
— crucially — **CPU inference with optional partial GPU offload**. This makes it
the only practical choice when:

1. No CUDA GPU is available (laptop CPU, edge device, macOS with Metal).
2. The model is too large even for quantized GPU memory (e.g., 70B model on a
   24 GB card — offload the overflow layers to CPU).
3. You want a single binary with zero Python dependencies (llama-server).

In a production context, I would use GGUF for **edge deployment or developer
tooling** (e.g., a local coding assistant running on a developer's MacBook) but
not for a GPU inference cluster, where vLLM + AWQ is superior.

### Decision tree (my personal heuristic)

```
Need to fine-tune? → bitsandbytes (QLoRA)
GPU inference cluster (vLLM / TGI)?
    Model fits in VRAM at fp16/bf16? → Use fp16/bf16 (no quality loss)
    Needs quantization for VRAM?     → AWQ (best throughput + quality)
    AWQ not available for your model?→ GPTQ
No GPU / CPU only / edge device?     → GGUF (Q4_K_M or Q5_K_M)
```

---

## Running the Benchmark

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run (downloads Qwen2.5-1.5B on first run, ~3 GB)
python benchmark.py

# 3. Results are saved to results/results.json
```

Expected runtime: **8–15 minutes** on an RTX 4050 (mostly model download on
the first run).

---

## Project Layout

```
section_3/
├── benchmark.py        # Main benchmark script
├── requirements.txt    # bitsandbytes, psutil, tabulate
├── README.md           # This file
└── results/
    └── results.json    # Auto-generated output (git-ignored until populated)
```
