"""
Section 3 — Task 3.1: Quantization Benchmark
=============================================
Loads Qwen2.5-1.5B-Instruct at fp16 and 4-bit NF4 (bitsandbytes),
measures VRAM footprint, tokens/sec throughput, and qualitative output
quality on 5 fixed prompts.

Uses subprocess isolation with localized ML imports to ensure each benchmark run
has a completely pristine CUDA context and prevents parent-child driver conflicts.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Windows console encoding fix — must happen before any print()
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import psutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 200
RESULTS_DIR = Path(__file__).parent / "results"
TEMP_FP16_PATH = RESULTS_DIR / "temp_fp16.json"
TEMP_NF4_PATH = RESULTS_DIR / "temp_nf4.json"

PROMPTS: list[dict[str, str]] = [
    {
        "id": "P1",
        "category": "Technical / CS",
        "text": "Explain the difference between a process and a thread in one paragraph.",
    },
    {
        "id": "P2",
        "category": "Coding",
        "text": "Write a Python function that returns the nth Fibonacci number using recursion.",
    },
    {
        "id": "P3",
        "category": "Factual",
        "text": "What is the capital of Australia and why is it not Sydney?",
    },
    {
        "id": "P4",
        "category": "Creative",
        "text": "Write the opening line of a mystery novel set in 1920s Cairo.",
    },
    {
        "id": "P5",
        "category": "Summarization",
        "text": "Summarize the main idea of Einstein's theory of special relativity in two sentences.",
    },
]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def get_vram_mb() -> float:
    """Return currently allocated VRAM in MB (GPU 0)."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1024**2
    return 0.0


def get_ram_mb() -> float:
    """Return current process RSS RAM in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def print_section(title: str) -> None:
    width = 70
    print("\n" + "=" * width, flush=True)
    print(f"  {title}", flush=True)
    print("=" * width, flush=True)


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def load_model_fp16(model_id: str):
    """Load model at full fp16 precision."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {model_id} at fp16 ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def load_model_nf4(model_id: str):
    """Load model at 4-bit NF4 via bitsandbytes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"  Loading {model_id} at 4-bit NF4 ...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,  # double quant causes silent crashes with bnb 0.50.0 on Windows
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, float]:
    """
    Generate a response for a single prompt.
    """
    import torch
    messages = [{"role": "user", "content": prompt_text}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).to(model.device)
    except Exception:
        enc = tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for reproducibility
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    new_tokens = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    n_generated = new_tokens.shape[0]
    elapsed = t_end - t_start
    tok_per_sec = n_generated / elapsed if elapsed > 0 else 0.0

    return text, tok_per_sec


def run_benchmark(
    label: str,
    model,
    tokenizer,
    vram_model_mb: float,
    ram_model_mb: float,
) -> dict[str, Any]:
    """Run all 5 prompts and collect results."""
    print_section(f"Running prompts — {label}")

    # Warm-up pass (not timed)
    print("  Warm-up pass ...", flush=True)
    _ = generate_response(model, tokenizer, "Hello!", max_new_tokens=10)

    prompt_results = []
    throughputs = []

    for p in PROMPTS:
        print(f"  [{p['id']}] {p['category']}: {p['text'][:55]}...", flush=True)
        text, tps = generate_response(model, tokenizer, p["text"])
        throughputs.append(tps)
        prompt_results.append(
            {
                "id": p["id"],
                "category": p["category"],
                "prompt": p["text"],
                "response": text,
                "tok_per_sec": round(tps, 2),
            }
        )
        print(f"         -> {tps:.1f} tok/s  |  {len(text.split())} words generated", flush=True)

    avg_tps = sum(throughputs) / len(throughputs)

    return {
        "label": label,
        "vram_model_mb": round(vram_model_mb, 1),
        "ram_model_mb": round(ram_model_mb, 1),
        "avg_tok_per_sec": round(avg_tps, 2),
        "prompts": prompt_results,
    }


# ---------------------------------------------------------------------------
# Execution Modes
# ---------------------------------------------------------------------------


def get_gpu_metadata() -> dict[str, Any]:
    """Fetch GPU properties inside the subprocess."""
    import torch
    if torch.cuda.is_available():
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
        }
    return {"gpu_name": "Unknown GPU", "total_vram_mb": 0.0}


def run_isolated_fp16() -> None:
    print_section("Run 1 of 2 — fp16 (full precision)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    import torch
    # Warm up CUDA context & get baseline memory
    if torch.cuda.is_available():
        torch.cuda.init()
    vram_before = get_vram_mb()
    ram_before = get_ram_mb()

    model, tokenizer = load_model_fp16(MODEL_ID)

    vram_after = get_vram_mb()
    ram_after = get_ram_mb()
    vram_used = vram_after - vram_before
    ram_used = ram_after - ram_before

    print(f"  VRAM used by model: {vram_used:.1f} MB", flush=True)
    print(f"  RAM  used by model: {ram_used:.1f} MB", flush=True)

    results = run_benchmark("fp16", model, tokenizer, vram_used, ram_used)
    
    # Inject GPU metadata
    metadata = get_gpu_metadata()
    results.update(metadata)

    with open(TEMP_FP16_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  fp16 results saved to temp file.", flush=True)


def run_isolated_nf4() -> None:
    print_section("Run 2 of 2 — 4-bit NF4 (bitsandbytes)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    import torch
    # Warm up CUDA context & get baseline memory
    if torch.cuda.is_available():
        torch.cuda.init()
    vram_before = get_vram_mb()
    ram_before = get_ram_mb()

    model, tokenizer = load_model_nf4(MODEL_ID)

    vram_after = get_vram_mb()
    ram_after = get_ram_mb()
    vram_used = vram_after - vram_before
    ram_used = ram_after - ram_before

    print(f"  VRAM used by model: {vram_used:.1f} MB", flush=True)
    print(f"  RAM  used by model: {ram_used:.1f} MB", flush=True)

    results = run_benchmark("4-bit NF4", model, tokenizer, vram_used, ram_used)
    
    # Inject GPU metadata
    metadata = get_gpu_metadata()
    results.update(metadata)

    with open(TEMP_NF4_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  4-bit NF4 results saved to temp file.", flush=True)


def orchestrate_benchmark() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print_section(f"Quantization Benchmark — {MODEL_ID}")
    print("Orchestrator starting up...", flush=True)

    # Clean up any stale temp files
    if TEMP_FP16_PATH.exists():
        TEMP_FP16_PATH.unlink()
    if TEMP_NF4_PATH.exists():
        TEMP_NF4_PATH.unlink()

    # Invoke fp16 in subprocess
    print("\n[Orchestrator] Launching fp16 benchmark subprocess...", flush=True)
    subprocess.run([sys.executable, __file__, "--run-fp16", "--model", MODEL_ID], check=True)

    # Invoke nf4 in subprocess
    print("\n[Orchestrator] Launching 4-bit NF4 benchmark subprocess...", flush=True)
    subprocess.run([sys.executable, __file__, "--run-nf4", "--model", MODEL_ID], check=True)

    # Read back results
    if not TEMP_FP16_PATH.exists() or not TEMP_NF4_PATH.exists():
        print("ERROR: Subprocesses failed to write results.", flush=True)
        sys.exit(1)

    with open(TEMP_FP16_PATH, "r", encoding="utf-8") as f:
        result_fp16 = json.load(f)
    with open(TEMP_NF4_PATH, "r", encoding="utf-8") as f:
        result_nf4 = json.load(f)

    # Clean up temp files
    TEMP_FP16_PATH.unlink()
    TEMP_NF4_PATH.unlink()

    all_results = [result_fp16, result_nf4]

    # Print summary
    print_section("Summary")

    from tabulate import tabulate
    table_rows = []
    for r in all_results:
        table_rows.append(
            [
                r["label"],
                f"{r['vram_model_mb']:.0f} MB",
                f"{r['ram_model_mb']:.0f} MB",
                f"{r['avg_tok_per_sec']:.1f}",
            ]
        )

    headers = ["Precision", "VRAM (model)", "RAM (model)", "Avg tok/s"]
    print(tabulate(table_rows, headers=headers, tablefmt="github"), flush=True)

    # Compute speed-up
    fp16_tps = result_fp16["avg_tok_per_sec"]
    nf4_tps = result_nf4["avg_tok_per_sec"]
    speedup = nf4_tps / fp16_tps if fp16_tps > 0 else 0
    vram_ratio = result_fp16["vram_model_mb"] / result_nf4["vram_model_mb"] if result_nf4["vram_model_mb"] > 0 else 0
    print(f"\n  NF4 speed-up vs fp16:  {speedup:.2f}x", flush=True)
    print(f"  NF4 VRAM reduction:    {vram_ratio:.2f}x smaller", flush=True)

    # Fetch GPU metadata from results
    gpu_name = result_fp16.get("gpu_name", "Unknown GPU")
    total_vram = result_fp16.get("total_vram_mb", 0.0)

    # Save results
    out = {
        "model": MODEL_ID,
        "gpu": gpu_name,
        "total_vram_mb": total_vram,
        "max_new_tokens": MAX_NEW_TOKENS,
        "results": all_results,
        "summary": {
            "fp16_vram_mb": result_fp16["vram_model_mb"],
            "nf4_vram_mb": result_nf4["vram_model_mb"],
            "fp16_avg_tps": result_fp16["avg_tok_per_sec"],
            "nf4_avg_tps": result_nf4["avg_tok_per_sec"],
            "nf4_speedup": round(speedup, 3),
            "vram_reduction_factor": round(vram_ratio, 3),
        },
    }

    model_slug = MODEL_ID.replace("/", "_").lower()
    results_path = RESULTS_DIR / f"results_{model_slug}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved -> {results_path}", flush=True)
    print("\nDone. Use the numbers above to fill in the README table.\n", flush=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-fp16", action="store_true")
    parser.add_argument("--run-nf4", action="store_true")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    MODEL_ID = args.model

    try:
        if args.run_fp16:
            run_isolated_fp16()
        elif args.run_nf4:
            run_isolated_nf4()
        else:
            orchestrate_benchmark()
    except Exception as exc:  # noqa: BLE001
        crash_path = Path(__file__).parent / f"crash_{'fp16' if args.run_fp16 else 'nf4' if args.run_nf4 else 'orch'}.log"
        with open(crash_path, "w", encoding="utf-8") as cf:
            cf.write(traceback.format_exc())
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr, flush=True)
        print(f"Full traceback written to {crash_path}", file=sys.stderr, flush=True)
        sys.exit(1)
