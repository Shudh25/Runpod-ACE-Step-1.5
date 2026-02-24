#!/usr/bin/env python3
"""
RunPod Serverless Training Handler for ACE-Step 1.5
====================================================

Pipeline: download audio → preprocess → train LoRA → save to Network Volume

ENV VARS:
  CHECKPOINT_DIR    Default: /app/checkpoints
  DIT_MODEL         Default: acestep-v15-base
  DEVICE            Default: cuda
  LORA_OUTPUT_DIR   Default: /runpod-volume/loras

INPUT:
{
    "audio_files": [
        {
            "url":           "https://...",
            "filename":      "song1.mp3",
            "caption":       "Indie folk with fingerpicked guitar and soft vocals",
            "lyrics":        "[Verse 1]\nHello world\n\n[Chorus]\nLa la la",
            "bpm":           120,         // optional
            "keyscale":      "C major",   // optional
            "timesignature": "4"          // optional
        }
    ],

    "lora_name":             "my_style",

    "lora_rank":             64,
    "lora_alpha":            64,
    "lora_dropout":          0.05,
    "target_modules":        ["to_q", "to_k", "to_v", "to_out.0"],

    "max_epochs":            500,
    "batch_size":            2,
    "learning_rate":         3e-4,
    "gradient_accumulation": 1,
    "save_every_n_epochs":   100,
    "shift":                 3.0,
    "seed":                  42,
    "precision":             "bf16",
    "optimizer":             "adamw"
}

OUTPUT:
{
    "output": {
        "status":                "success",
        "lora_name":             "my_style",
        "lora_path":             "/runpod-volume/loras/my_style",
        "lora_files":            ["adapter_model.safetensors", "adapter_config.json"],
        "epochs_trained":        500,
        "final_loss":            0.042,
        "training_time_seconds": 3600
    }
}
"""

import os
import shutil
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Optional

import runpod

sys.path.insert(0, "/app")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_DIR  = os.environ.get("CHECKPOINT_DIR",  "/app/checkpoints")
DIT_MODEL       = os.environ.get("DIT_MODEL",       "acestep-v15-base")
DEVICE          = os.environ.get("DEVICE",          "cuda")
LORA_OUTPUT_DIR = os.environ.get("LORA_OUTPUT_DIR", "/runpod-volume/loras")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_audio(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ACE-Step-Trainer/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        with open(dest, "wb") as f:
            f.write(r.read())


def _find_final_lora(output_dir: str) -> Optional[str]:
    """Find the best available LoRA weights directory after training."""
    out = Path(output_dir)
    # 1. Explicit 'final' export directory
    if (out / "final").exists() and any((out / "final").iterdir()):
        return str(out / "final")
    # 2. Latest epoch checkpoint
    epoch_dirs = sorted(out.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    if epoch_dirs:
        return str(epoch_dirs[-1])
    # 3. Weights in root
    if (out / "adapter_model.safetensors").exists():
        return str(out)
    return None


# ---------------------------------------------------------------------------
# Stage 1: Download audio + preprocess to tensors
# ---------------------------------------------------------------------------

def _build_dataset(audio_files: list, work_dir: str) -> str:
    """
    Download audio files and preprocess to .pt tensors.

    Confirmed DatasetBuilder API:
      builder = DatasetBuilder()                              # no constructor args
      output_paths, status = builder.preprocess_to_tensors(  # samples as positional arg
          samples,
          output_dir
      )
    """
    from acestep.training.dataset_builder import DatasetBuilder, AudioSample
    from acestep.handler import AceStepHandler

    audio_dir   = Path(work_dir) / "audio"
    tensors_dir = Path(work_dir) / "tensors"
    audio_dir.mkdir(parents=True)
    tensors_dir.mkdir(parents=True)

    # Download audio
    print(f"[Training] Downloading {len(audio_files)} audio file(s)...")
    samples = []
    for i, af in enumerate(audio_files):
        url      = af.get("url", "")
        filename = af.get("filename", f"audio_{i}.mp3")
        dest     = str(audio_dir / filename)

        if not url:
            raise ValueError(f"audio_files[{i}] is missing 'url'")

        print(f"[Training]   {filename}...")
        _download_audio(url, dest)

        samples.append(AudioSample(
            audio_path    = dest,
            caption       = af.get("caption",       ""),
            lyrics        = af.get("lyrics",        ""),
            bpm           = af.get("bpm",           None),
            keyscale      = af.get("keyscale",      ""),
            timesignature = str(af.get("timesignature", "")),
        ))

    # Initialize AceStepHandler — DatasetBuilder uses this loaded service
    # internally for VAE encoding and text encoding.
    print(f"[Training] Initializing AceStepHandler for preprocessing...")
    dit = AceStepHandler()
    dit.initialize_service(
        project_root=CHECKPOINT_DIR,
        config_path=DIT_MODEL,
        device=DEVICE,
        offload_to_cpu=False,
    )

    # ── KEY FIX ───────────────────────────────────────────────────────────────
    # preprocess_to_tensors(samples, output_dir) — samples is a required
    # positional arg, NOT set via builder.samples attribute.
    print(f"[Training] Preprocessing {len(samples)} sample(s) to tensors...")
    builder = DatasetBuilder()
    output_paths, status = builder.preprocess_to_tensors(samples, str(tensors_dir))

    print(f"[Training] Preprocessing status: {status}")
    print(f"[Training] Tensors at: {tensors_dir}")
    return str(tensors_dir)


# ---------------------------------------------------------------------------
# Stage 2: LoRA training
# ---------------------------------------------------------------------------

def _run_training(tensors_dir: str, lora_out: str, inp: dict) -> dict:
    from acestep.training.trainer import LoRATrainer
    from acestep.training.configs import LoRAConfig, TrainingConfig

    lora_cfg = LoRAConfig(
        rank           = inp.get("lora_rank",       64),
        alpha          = inp.get("lora_alpha",       64),
        dropout        = inp.get("lora_dropout",   0.05),
        target_modules = inp.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
    )

    train_cfg = TrainingConfig(
        dataset_path            = tensors_dir,
        output_dir              = lora_out,
        max_epochs              = inp.get("max_epochs",            500),
        batch_size              = inp.get("batch_size",              2),
        learning_rate           = inp.get("learning_rate",        3e-4),
        accumulate_grad_batches = inp.get("gradient_accumulation",   1),
        save_every_n_epochs     = inp.get("save_every_n_epochs",   100),
        shift                   = inp.get("shift",                 3.0),
        seed                    = inp.get("seed",                   42),
        precision               = inp.get("precision",           "bf16"),
        optimizer               = inp.get("optimizer",         "adamw"),
        checkpoint_dir          = CHECKPOINT_DIR,
        dit_model               = DIT_MODEL,
        device                  = DEVICE,
    )

    print(
        f"[Training] Starting LoRA —"
        f" rank={lora_cfg.rank}"
        f" epochs={train_cfg.max_epochs}"
        f" batch={train_cfg.batch_size}"
        f" precision={train_cfg.precision}"
        f" optimizer={train_cfg.optimizer}"
    )

    t0      = time.time()
    trainer = LoRATrainer(lora_config=lora_cfg, train_config=train_cfg)
    metrics = trainer.train()
    elapsed = int(time.time() - t0)

    print(f"[Training] Done in {elapsed}s — metrics: {metrics}")
    return {**metrics, "training_time_seconds": elapsed}


# ---------------------------------------------------------------------------
# Stage 3: Save weights to network volume
# ---------------------------------------------------------------------------

def _save_to_volume(final_lora_dir: str, lora_name: str) -> dict:
    dest = Path(LORA_OUTPUT_DIR) / lora_name
    dest.mkdir(parents=True, exist_ok=True)

    for f in Path(final_lora_dir).rglob("*"):
        if f.is_file():
            target = dest / f.relative_to(final_lora_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)

    lora_files = [str(f.relative_to(dest)) for f in dest.rglob("*") if f.is_file()]
    print(f"[Training] LoRA saved → {dest}")
    print(f"[Training] Files: {lora_files}")
    return {
        "lora_path":  str(dest),
        "lora_files": lora_files,
    }


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    inp: dict = job.get("input", {})

    audio_files = inp.get("audio_files", [])
    if not audio_files:
        return {"error": "Missing required parameter: 'audio_files'"}

    lora_name = inp.get("lora_name", "lora")

    # Validate network volume is mounted before doing any work
    vol = Path(LORA_OUTPUT_DIR)
    if not vol.parent.exists():
        return {
            "error": (
                f"Network volume not mounted at '{vol.parent}'. "
                "Attach a Network Volume to this endpoint and set "
                f"LORA_OUTPUT_DIR env var (default: {LORA_OUTPUT_DIR})."
            )
        }

    work_dir = tempfile.mkdtemp(prefix="acetrain_")
    lora_out = str(Path(work_dir) / "lora_output")
    Path(lora_out).mkdir()

    try:
        # Stage 1: Preprocess
        print(f"[Training] === Stage 1: Preprocessing ===")
        tensors_dir = _build_dataset(audio_files, work_dir)

        # Stage 2: Train
        print(f"[Training] === Stage 2: Training ===")
        metrics = _run_training(tensors_dir, lora_out, inp)

        # Find output weights
        final_dir = _find_final_lora(lora_out)
        if not final_dir:
            return {"error": "Training finished but no LoRA weights found in output directory."}

        # Stage 3: Save to volume
        print(f"[Training] === Stage 3: Saving to volume ===")
        volume_result = _save_to_volume(final_dir, lora_name)

        return {
            "output": {
                "status":                "success",
                "lora_name":             lora_name,
                "epochs_trained":        metrics.get("epochs_trained"),
                "final_loss":            metrics.get("final_loss"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                **volume_result,
            }
        }

    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc)}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})