#!/usr/bin/env python3
"""
Standalone Training Test Script for ACE-Step 1.5 — Round 2
============================================================
Uses confirmed real API signatures from Step 0 introspection.

Run on the pod:
    python /app/test_training.py
"""

import sys
import os
import json
import shutil
import tempfile
import traceback
import inspect
import time
from pathlib import Path

sys.path.insert(0, "/app")

# =============================================================================
# CONFIG
# =============================================================================
CHECKPOINT_DIR = "/app/checkpoints"   # initialize_service project_root
DIT_MODEL      = "acestep-v15-base"
DEVICE         = "cuda"
OUTPUT_DIR     = "/tmp/test_lora_output"

TEST_AUDIO_URL  = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
TEST_AUDIO_NAME = "test_song.mp3"

TEST_LORA_RANK  = 4
TEST_MAX_EPOCHS = 5
TEST_BATCH_SIZE = 1
# =============================================================================

SEP = "=" * 70
def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def ok(m):   print(f"  ✓  {m}")
def fail(m): print(f"  ✗  {m}")
def info(m): print(f"  →  {m}")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
section("Imports")

from acestep.training.dataset_builder import DatasetBuilder, AudioSample
from acestep.training.trainer import LoRATrainer
from acestep.training.configs import LoRAConfig, TrainingConfig
from acestep.handler import AceStepHandler

ok("All imports successful")

# ---------------------------------------------------------------------------
# Step 1: Download audio
# ---------------------------------------------------------------------------
section("STEP 1: Download audio")

work_dir  = tempfile.mkdtemp(prefix="acetrain_test_")
audio_dir = Path(work_dir) / "audio"
audio_dir.mkdir()
audio_dest = str(audio_dir / TEST_AUDIO_NAME)

info(f"Work dir: {work_dir}")

import urllib.request
req = urllib.request.Request(TEST_AUDIO_URL, headers={"User-Agent": "ACE-Step/1.0"})
with urllib.request.urlopen(req, timeout=60) as r:
    with open(audio_dest, "wb") as f:
        f.write(r.read())
ok(f"Downloaded {TEST_AUDIO_NAME} ({Path(audio_dest).stat().st_size // 1024}KB)")

# ---------------------------------------------------------------------------
# Step 2: Load AceStepHandler
# ---------------------------------------------------------------------------
section("STEP 2: Load AceStepHandler")

dit = AceStepHandler()
dit.initialize_service(
    project_root=CHECKPOINT_DIR,
    config_path=DIT_MODEL,
    device=DEVICE,
    offload_to_cpu=False,
)
ok("AceStepHandler initialized")

# ---------------------------------------------------------------------------
# Step 3: Build AudioSample + preprocess
# ---------------------------------------------------------------------------
section("STEP 3: Preprocess audio -> tensors")

tensors_dir = Path(work_dir) / "tensors"
tensors_dir.mkdir()

# Confirmed real AudioSample fields:
# id, audio_path, filename, caption, genre, lyrics, raw_lyrics,
# formatted_lyrics, bpm, keyscale, timesignature, duration, language,
# is_instrumental, custom_tag, labeled, prompt_override
#
# KEY: labeled=True is required — without it the preprocessor sees
# "No samples to preprocess" and returns an empty list

sample = AudioSample(
    audio_path = audio_dest,
    filename   = TEST_AUDIO_NAME,
    caption    = "Upbeat electronic track with synthesizers and driving drums",
    lyrics     = "[Verse 1]\nTest verse lyrics here\n\n[Chorus]\nLa la la test",
    language   = "en",
    labeled    = True,
)
info(f"AudioSample created: labeled={sample.labeled}")

builder = DatasetBuilder()

# Confirmed real signature:
# preprocess_to_tensors(self, dit_handler, output_dir, max_duration, preprocess_mode, progress_callback)
info("Calling preprocess_to_tensors(dit_handler, output_dir)...")
try:
    output_paths, status = builder.preprocess_to_tensors(
        dit,
        str(tensors_dir),
    )
    info(f"Status: {status}")
    info(f"Output paths: {output_paths}")
except Exception as e:
    fail(f"preprocess_to_tensors failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Verify tensors were created
tensor_files = (
    list(tensors_dir.rglob("*.pt")) +
    list(tensors_dir.rglob("*.safetensors")) +
    list(tensors_dir.rglob("*.json"))
)
if tensor_files:
    ok(f"Tensor files created: {[f.name for f in tensor_files]}")
else:
    fail("Still no tensor files. Full dir tree:")
    for p in sorted(tensors_dir.rglob("*")):
        info(f"  {p}")
    info(f"builder.samples: {getattr(builder, 'samples', 'N/A')}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 4: LoRA training
# ---------------------------------------------------------------------------
section("STEP 4: LoRA training (5 epochs smoke test)")

lora_out = Path(work_dir) / "lora_output"
lora_out.mkdir()

# Confirmed real LoRAConfig fields: r (NOT rank), alpha, dropout, target_modules, bias
lora_cfg = LoRAConfig(
    r              = TEST_LORA_RANK,
    alpha          = TEST_LORA_RANK,
    dropout        = 0.05,
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"],
)
ok(f"LoRAConfig: r={TEST_LORA_RANK}")

# Confirmed real TrainingConfig fields (no checkpoint_dir, no dit_model):
# shift, num_inference_steps, learning_rate, batch_size,
# gradient_accumulation_steps, max_epochs, save_every_n_epochs,
# warmup_steps, weight_decay, max_grad_norm, mixed_precision,
# use_fp8, gradient_checkpointing, seed, output_dir,
# num_workers, pin_memory, prefetch_factor, persistent_workers,
# pin_memory_device, log_every_n_steps, val_split
train_cfg = TrainingConfig(
    output_dir                  = str(lora_out),
    max_epochs                  = TEST_MAX_EPOCHS,
    batch_size                  = TEST_BATCH_SIZE,
    learning_rate               = 3e-4,
    gradient_accumulation_steps = 1,
    save_every_n_epochs         = 5,
    shift                       = 3.0,
    mixed_precision             = "bf16",
    seed                        = 42,
)
ok("TrainingConfig created")

# Confirmed real LoRATrainer signature:
# LoRATrainer(self, dit_handler, lora_config, training_config)
info("Instantiating LoRATrainer(dit, lora_cfg, train_cfg)...")
try:
    trainer = LoRATrainer(dit, lora_cfg, train_cfg)
    ok("LoRATrainer instantiated")
except Exception as e:
    fail(f"LoRATrainer failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Introspect train() before calling — we need to know if it takes dataset_dir
train_sig = inspect.signature(trainer.train)
info(f"trainer.train signature: {train_sig}")

# Try train(tensors_dir) first, fall back to train() if no args needed
info(f"Starting training ({TEST_MAX_EPOCHS} epochs)...")
try:
    t0     = time.time()
    result = trainer.train(str(tensors_dir))
    ok(f"train(tensors_dir) worked — completed in {int(time.time()-t0)}s")
    info(f"Returned: {result}")
except TypeError as e:
    info(f"train(tensors_dir) failed ({e}) — trying train() with no args...")
    try:
        t0     = time.time()
        result = trainer.train()
        ok(f"train() worked — completed in {int(time.time()-t0)}s")
        info(f"Returned: {result}")
    except Exception as e2:
        fail(f"train() also failed: {e2}")
        traceback.print_exc()
        sys.exit(1)
except Exception as e:
    fail(f"Training error: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 5: Verify output
# ---------------------------------------------------------------------------
section("STEP 5: Verify output")

info("Files in lora_output:")
for f in sorted(lora_out.rglob("*")):
    if f.is_file():
        info(f"  {f.relative_to(lora_out)}  ({f.stat().st_size // 1024}KB)")

safetensors = list(lora_out.rglob("*.safetensors"))
if safetensors:
    ok(f"LoRA weights: {[f.name for f in safetensors]}")
else:
    fail("No .safetensors found")

if Path(OUTPUT_DIR).exists():
    shutil.rmtree(OUTPUT_DIR)
shutil.copytree(str(lora_out), OUTPUT_DIR)
ok(f"Output copied to {OUTPUT_DIR}")
shutil.rmtree(work_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("CONFIRMED SIGNATURES — paste output to get final training_handler.py")
info("AudioSample:           labeled=True is required")
info("preprocess_to_tensors: builder.preprocess_to_tensors(dit_handler, output_dir)")
info("LoRAConfig:            r=N  (not rank=N)")
info("TrainingConfig:        mixed_precision='bf16', gradient_accumulation_steps=N")
info("LoRATrainer:           LoRATrainer(dit_handler, lora_config, training_config)")
info("trainer.train:         see above for confirmed call signature")
print()
ok("Done — paste this output back to get the final training_handler.py")