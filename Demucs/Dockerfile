# =============================================================================
# Demucs - RunPod Serverless Worker
# =============================================================================

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OUTPUT_DIR=/runpod-volume/demucs_outputs

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Install PyTorch (CUDA compatible)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Demucs
RUN pip install demucs runpod

# Create output dir
RUN mkdir -p /runpod-volume/demucs_outputs

COPY demucs_handler.py /app/demucs_handler.py

CMD ["python", "-u", "/app/demucs_handler.py"]