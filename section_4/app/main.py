import os
import time
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from app.model import get_model
from app.schemas import GenerateRequest, GenerateResponse
from app.streaming import token_stream

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        print("Pre-loading model on startup...", flush=True)
        get_model()
        print("Model pre-loaded successfully.", flush=True)
    except Exception as e:
        print(f"Error loading model on startup: {e}", flush=True)
    yield
    # Clean up (if any)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Qwen2.5-1.5B-Instruct Inference API",
    description="FastAPI service for serving Qwen2.5-1.5B-Instruct model with streaming support.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    try:
        model, _ = get_model()
        return {
            "status": "healthy",
            "model_device": str(model.device),
            "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 1) if torch.cuda.is_available() else 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhealthy: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        model, tokenizer = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
    
    # Check if user requested streaming but hit non-streaming endpoint
    if request.stream:
        raise HTTPException(
            status_code=400, 
            detail="Use '/generate/stream' endpoint for streaming responses."
        )

    try:
        # Pre-process prompt
        try:
            messages = [{"role": "user", "content": request.prompt}]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
        except Exception:
            enc = tokenizer(request.prompt, return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            attention_mask = enc.attention_mask.to(model.device)

        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=request.max_new_tokens,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )
        if request.temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = request.temperature
        else:
            generation_kwargs["do_sample"] = False

        # Measure times
        t_start = time.perf_counter()
        
        # Start in thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        tokens = []
        ttft_ms = 0.0
        first = True
        
        for token in streamer:
            if first:
                ttft_ms = (time.perf_counter() - t_start) * 1000.0
                first = False
            tokens.append(token)
            
        t_end = time.perf_counter()
        
        full_text = "".join(tokens)
        total_latency_ms = (t_end - t_start) * 1000.0
        
        # Exact token count generated
        tokens_generated = len(tokenizer.encode(full_text))
        
        if tokens_generated == 0:
            tokens_generated = 1  # prevent division by zero
            
        tokens_per_sec = tokens_generated / (total_latency_ms / 1000.0)

        # If model did not return any tokens, set default ttft
        if first:
            ttft_ms = total_latency_ms

        return GenerateResponse(
            text=full_text,
            tokens_generated=tokens_generated,
            time_to_first_token_ms=round(ttft_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            tokens_per_sec=round(tokens_per_sec, 2)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    try:
        model, tokenizer = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
    
    return StreamingResponse(
        token_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        ),
        media_type="text/event-stream"
    )
