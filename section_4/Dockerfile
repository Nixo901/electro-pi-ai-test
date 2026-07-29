FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/model_cache
ENV MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
ENV PORT=8000
ENV DEVICE=cuda
ENV QUANTIZATION=fp16

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up work directory
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Pre-download the model and tokenizer at build time
RUN python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('${MODEL_ID}'); \
    AutoModelForCausalLM.from_pretrained('${MODEL_ID}')"

# Copy the app files
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
