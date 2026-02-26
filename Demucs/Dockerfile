# =============================================================================
# Demucs Stem Separation - RunPod Serverless Worker
# =============================================================================

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEVICE=cuda

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# PyTorch cu121 — ships torchaudio with soundfile backend (no torchcodec issues)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Demucs + RunPod SDK + boto3 for S3 uploads
RUN pip install demucs runpod boto3

# Pre-bake model weights — avoids cold-start download
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

COPY demucs_handler.py /app/demucs_handler.py

CMD ["python", "-u", "/app/demucs_handler.py"]