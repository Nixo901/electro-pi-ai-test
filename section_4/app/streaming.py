import asyncio
import json
from threading import Thread
from transformers import TextIteratorStreamer

async def token_stream(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float):
    """
    Asynchronous generator that runs model.generate in a background thread and yields 
    each generated token as a Server-Sent Event (SSE) JSON chunk.
    """
    try:
        messages = [{"role": "user", "content": prompt}]
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
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id
    )
    if temperature > 0.0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    # Start generation in thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    loop = asyncio.get_event_loop()
    
    def get_next(streamer_obj):
        try:
            return next(streamer_obj)
        except StopIteration:
            return None

    # Track if we have sent any tokens for latency measurements
    first_token_sent = False
    
    while True:
        token = await loop.run_in_executor(None, get_next, streamer)
        if token is None:
            break
        
        # Emit SSE data
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0.001)  # small yields

    yield "data: [DONE]\n\n"
