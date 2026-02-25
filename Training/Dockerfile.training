# =============================================================================
# ACE-Step 1.5 — RunPod Serverless TRAINING Worker
# =============================================================================
# Target GPU : A100 80GB (recommended) / RTX 3090 24GB (small datasets)
# Storage    : RunPod Network Volume (/runpod-volume)
# Output     : LoRA .safetensors written to Network Volume
# =============================================================================
# BUILD:
#   docker build -f Dockerfile.training \
#     --build-arg HF_TOKEN=hf_xxx \
#     -t your-registry/acestep-training:latest .
#
# PUSH:
#   docker push your-registry/acestep-training:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Model Downloader  (python:3.11-slim — tiny, no GPU needed)
# Downloads: VAE + Qwen3-Embedding-0.6B text encoder (from main bundle)
#            + acestep-v15-base DiT  (the model LoRA trains on top of)
#
# NOT downloaded (saves image space):
#   acestep-v15-turbo  — turbo inference model, not used for training
#   acestep-5Hz-lm-*   — language models, not used during training
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS model-downloader

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN} \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /models

RUN pip install --no-cache-dir "huggingface-hub[cli,hf_transfer]"

# Main bundle: downloads VAE + Qwen3-Embedding-0.6B text encoder
# Skips turbo DiT and all LM weights to keep image lean
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/Ace-Step1.5', local_dir='/models/checkpoints', token=os.environ.get('HF_TOKEN'), ignore_patterns=['acestep-v15-turbo/*','acestep-5Hz-lm-0.6B/*','acestep-5Hz-lm-1.7B/*','acestep-5Hz-lm-4B/*']); print('Main bundle downloaded')"

# Base DiT — the foundation model LoRA adapters train on top of.
# Using base (not turbo/SFT) because it has the best fine-tunability
# per the model zoo docs.
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/acestep-v15-base', local_dir='/models/checkpoints/acestep-v15-base', token=os.environ.get('HF_TOKEN')); print('Base DiT downloaded')"

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# cudnn-devel  — required for backpropagation (training), not just inference
# ubuntu 22.04 — stable, CUDA 12.8 compatible
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # ── These are the only env vars users need to override ──────────────────
    CHECKPOINT_DIR=/app/checkpoints \
    DIT_MODEL=acestep-v15-base \
    DEVICE=cuda \
    LORA_OUTPUT_DIR=/runpod-volume/loras

WORKDIR /app

# ── System packages ──────────────────────────────────────────────────────────
# Python 3.11 via deadsnakes PPA  — Ubuntu 22.04 ships 3.10 by default.
# ACE-Step requires 3.11 (pyproject.toml constraint).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3-pip \
        git \
        curl \
        wget \
        build-essential \
        libsndfile1 \
        libsndfile1-dev \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# ── uv (fast package manager — required by ACE-Step) ────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Clone ACE-Step and install the package ───────────────────────────────────
# Clones latest main. Pin to a specific commit hash for reproducibility:
#   git checkout <COMMIT_SHA>
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /app \
    && rm -rf /app/.git

# Install ACE-Step package (uses pyproject.toml)
RUN uv pip install --system --no-cache .

# ── Training-specific dependencies ───────────────────────────────────────────
# lightning       — Lightning Fabric for training loop (new name, was pytorch-lightning)
# peft            — Parameter-Efficient Fine-Tuning (LoRA implementation)
# runpod          — RunPod serverless SDK
# bitsandbytes    — 8-bit optimizers (optional but useful for large batches)
RUN uv pip install --system --no-cache \
    "lightning>=2.0.0" \
    "peft>=0.7.0" \
    "runpod>=1.6.0" \
    "bitsandbytes>=0.41.0" \
    "safetensors>=0.4.0"

# ── Checkpoint directory layout ──────────────────────────────────────────────
# AceStepHandler.initialize_service(project_root=CHECKPOINT_DIR) internally
# resolves models at:  {project_root}/checkpoints/{model_name}
# So with CHECKPOINT_DIR=/app/checkpoints, the DiT lives at:
#   /app/checkpoints/checkpoints/acestep-v15-base/
#
# The model-downloader stage writes to /models/checkpoints/
# We COPY that into /app/checkpoints/checkpoints/ to match the expected path.
COPY --from=model-downloader /models/checkpoints /app/checkpoints/checkpoints

# ── Placeholder directories ───────────────────────────────────────────────────
# check_main_model_exists() in AceStepHandler ALWAYS looks for
# "acestep-v15-turbo" regardless of which DiT is configured.
# Empty placeholder directory prevents a runtime re-download that fills disk.
RUN mkdir -p /app/checkpoints/checkpoints/acestep-v15-turbo

# Similarly, auto-detection for LM models looks for common LM dirs.
# Empty placeholders suppress auto-download attempts during preprocess init.
RUN mkdir -p \
    /app/checkpoints/checkpoints/acestep-5Hz-lm-0.6B \
    /app/checkpoints/checkpoints/acestep-5Hz-lm-1.7B

# Symlink so the package can also find checkpoints at the package root
RUN ln -sf /app/checkpoints /usr/local/lib/python3.11/dist-packages/checkpoints || true   
# non-fatal if path doesn't exist yet

# ── Working directories ───────────────────────────────────────────────────────
RUN mkdir -p /app/outputs /tmp/training /runpod-volume/loras

# ── Handler ───────────────────────────────────────────────────────────────────
COPY training_handler.py /app/training_handler.py

# Smoke-test imports at build time — catches missing packages immediately
RUN python -c "import torch, torchaudio, lightning, peft, safetensors, runpod; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}'); print(f'lightning={lightning.__version__}  peft={peft.__version__}'); print('All training dependencies import OK')"

# Validate ACE-Step training module is importable
RUN python -c "from acestep.training.dataset_builder import DatasetBuilder, AudioSample; from acestep.training.trainer import LoRATrainer; from acestep.training.configs import LoRAConfig, TrainingConfig; from acestep.training.data_module import PreprocessedTensorDataset; print('ACE-Step training module OK'); import inspect; lora_fields=[f.name for f in LoRAConfig.__dataclass_fields__.values()]; train_fields=[f.name for f in TrainingConfig.__dataclass_fields__.values()]; print(f'LoRAConfig fields:     {lora_fields}'); print(f'TrainingConfig fields: {train_fields}'); train_sig=inspect.signature(LoRATrainer.__init__); print(f'LoRATrainer.__init__ signature: {train_sig}')"


CMD ["python", "-u", "/app/training_handler.py"]