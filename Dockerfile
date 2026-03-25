# nanochat Docker image
# Serves the FastAPI chat server (scripts/chat_web.py) on port 8000.
#
# Build:
#   docker build -t nanochat:latest .
#
# Run (GPU):
#   docker run --gpus all -p 8000:8000 \
#     -v C:/Users/<you>/.cache/nanochat:/checkpoints:ro \
#     -e NANOCHAT_BASE_DIR=/checkpoints \
#     nanochat:latest
#
# Or use docker-compose (recommended):
#   docker compose up

FROM python:3.11-slim

# System build deps 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# work directory 
WORKDIR /app 

# PyTorch (CUDA 12.8) — large layer, cache separately 
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cu128

# install all python dependencies
RUN pip install --no-cache-dir \
    datasets \
    fastapi \
    uvicorn[standard] \
    pydantic \
    rustbpe \
    regex \
    tiktoken \
    tokenizers \
    transformers \
    zstandard \
    psutil \
    matplotlib

# Application code 
COPY nanochat/  ./nanochat/
COPY scripts/   ./scripts/
COPY tasks/     ./tasks/

# Mount your checkpoint directory at /checkpoints (see docker-compose.yml)
ENV NANOCHAT_BASE_DIR=/checkpoints

EXPOSE 8000

# Override with --source, --num-gpus, etc. via docker-compose `command:`
CMD ["python", "-m", "scripts.chat_web", \
     "--source", "sft", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
