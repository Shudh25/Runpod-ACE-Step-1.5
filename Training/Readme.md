# ACE-Step 1.5 – LoRA Training (RunPod Serverless)

Production-ready LoRA fine-tuning endpoint for **ACE-Step 1.5**, designed for A100 80GB GPUs.

This endpoint:

* Downloads audio files
* Preprocesses to tensors
* Trains LoRA adapters
* Saves results to RunPod Network Volume
* Returns training metrics + saved paths

---

# Architecture

This is part of a 3-endpoint system:

| Endpoint  | GPU       | Purpose          |
| --------- | --------- | ---------------- |
| Inference | A10G      | Music generation |
| Training  | A100 80GB | LoRA fine-tuning |
| Demucs    | T4 / A10G | Vocal separation |

Training runs independently and does not block inference.

---

# 📁 Folder Structure

```bash
ace-step-training/
│
├── Dockerfile
├── training_handler.py
├── sample_request.json
└── README.md
```

---

# 🐳 Docker Build & Push

### 1️⃣ Build Image

```bash
docker build --build-arg HF_TOKEN=hf_xyz -t your-dockerhub-username/ace-step-training:latest .
```

### 2️⃣ Push Image

```bash
docker push your-dockerhub-username/ace-step-training:latest
```

---

# 🚀 RunPod Configuration

When creating the Serverless endpoint:

* **GPU**: A100 80GB
* **Timeout**: 2–6 hours (depending on epochs)
* **Concurrency**: 1
* **Network Volume**: Required
* **Mount Path**: `/runpod-volume`

Environment variables (optional):

```bash
CHECKPOINT_DIR=/app/checkpoints
DIT_MODEL=acestep-v15-base
DEVICE=cuda
LORA_OUTPUT_DIR=/runpod-volume/loras
```

---

# Sample Request

Example request body:

```json
{
  "input": {
    "audio_files": [
      {
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "filename": "song_01.mp3",
        "caption": "Upbeat electronic track with driving synthesizers",
        "lyrics": "[Verse]\nHello world\n\n[Chorus]\nLa la la",
        "bpm": 128,
        "keyscale": "A minor",
        "timesignature": "4"
      }
    ],
    "lora_name": "my_style",
    "lora_rank": 64,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "max_epochs": 100,
    "batch_size": 1,
    "learning_rate": 0.0001,
    "precision": "bf16"
  }
}
```

---

# Sample Response

```json
{
  "output": {
    "status": "success",
    "lora_name": "my_style",
    "lora_path": "/runpod-volume/loras/my_style",
    "lora_files": [
      "adapter_model.safetensors",
      "adapter_config.json"
    ],
    "epochs_trained": 100,
    "final_loss": 0.042,
    "training_time_seconds": 3600
  }
}
```

---

# ⚙️ Training Flow

1. Download audio files
2. Convert to tensors
3. Initialize ACE-Step base model
4. Run LoRA training
5. Save final adapter weights
6. Copy to Network Volume

---

# 💾 Output Location

All trained LoRAs are stored in:

```
/runpod-volume/loras/<lora_name>/
```

Example:

```
/runpod-volume/loras/my_style/adapter_model.safetensors
```

These can then be hot-swapped in the inference endpoint.

---

# Notes

* Use **bf16 precision** on A100 for best performance
* Concurrency must be 1
* Training time scales with:

  * Number of audio files
  * Epoch count
  * Batch size
  * LoRA rank

---

# Production Best Practices

* Always mount a persistent Network Volume
* Monitor disk usage during training
* Keep inference and training endpoints separate
* Use unique `lora_name` per job
