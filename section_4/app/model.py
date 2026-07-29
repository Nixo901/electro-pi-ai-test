import os
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_model = None
_tokenizer = None
_lock = threading.Lock()

def get_model():
    global _model, _tokenizer
    if _model is None:
        with _lock:
            if _model is None:
                model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
                device = os.getenv("DEVICE", "cuda")
                quantization = os.getenv("QUANTIZATION", "fp16").lower()
                
                print(f"Loading tokenizer for {model_id}...", flush=True)
                _tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                print(f"Loading model {model_id} on {device} with quantization {quantization}...", flush=True)
                if device == "cuda" and quantization == "4bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=False,
                    )
                    _model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=bnb_config,
                        device_map="auto"
                    )
                else:
                    # Default load in fp16/bf16
                    torch_dtype = torch.float16 if device == "cuda" else torch.float32
                    _model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        device_map="auto" if device == "cuda" else None
                    )
                
                if device == "cuda":
                    # Warmup CUDA context
                    torch.cuda.init()
                
                _model.eval()
                print("Model loaded successfully.", flush=True)
    return _model, _tokenizer
