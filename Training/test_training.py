#!/usr/bin/env python3
"""
ACE-Step 1.5 Training Test — Round 3
======================================
Key insight from GitHub issue trace:
  DatasetBuilder.samples is populated by loading a dataset JSON file,
  NOT by passing Python objects directly.

The Gradio UI flow is:
  1. Scan audio folder → creates AudioSample objects
  2. User labels them → sets labeled=True
  3. Save Dataset → writes dataset.json
  4. Load Dataset (for training) → builder.load_dataset(json_path)
  5. Preprocess → builder.preprocess_to_tensors(dit_handler, output_dir)
  6. Train → LoRATrainer(dit_handler, lora_cfg, train_cfg).train(tensors_dir)

We replicate steps 1-6 programmatically.

Run on pod:
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
CHECKPOINT_DIR = "/app/checkpoints"
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
ok("All imports OK")

# Print all DatasetBuilder methods so we can see load_dataset etc.
methods = [m for m in dir(DatasetBuilder) if not m.startswith("__")]
info(f"DatasetBuilder methods: {methods}")

# ---------------------------------------------------------------------------
# Step 1: Download audio
# ---------------------------------------------------------------------------
section("STEP 1: Download audio")

work_dir    = tempfile.mkdtemp(prefix="acetrain_test_")
audio_dir   = Path(work_dir) / "audio"
tensors_dir = Path(work_dir) / "tensors"
lora_out    = Path(work_dir) / "lora_output"
audio_dir.mkdir()
tensors_dir.mkdir()
lora_out.mkdir()

audio_dest = str(audio_dir / TEST_AUDIO_NAME)
info(f"Work dir: {work_dir}")

import urllib.request
req = urllib.request.Request(TEST_AUDIO_URL, headers={"User-Agent": "ACE-Step/1.0"})
with urllib.request.urlopen(req, timeout=60) as r:
    with open(audio_dest, "wb") as f:
        f.write(r.read())
ok(f"Downloaded {TEST_AUDIO_NAME} ({Path(audio_dest).stat().st_size // 1024}KB)")

# ---------------------------------------------------------------------------
# Step 2: Write dataset.json the way the Gradio UI does
# ---------------------------------------------------------------------------
section("STEP 2: Build dataset.json")

# AudioSample fields confirmed: id, audio_path, filename, caption, genre,
# lyrics, raw_lyrics, formatted_lyrics, bpm, keyscale, timesignature,
# duration, language, is_instrumental, custom_tag, labeled, prompt_override

dataset = [
    {
        "audio_path":    audio_dest,
        "filename":      TEST_AUDIO_NAME,
        "caption":       "Upbeat electronic track with synthesizers and driving drums",
        "lyrics":        "[Verse 1]\nTest verse lyrics here\n\n[Chorus]\nLa la la test",
        "language":      "en",
        "bpm":           None,
        "keyscale":      "",
        "timesignature": "",
        "duration":      None,
        "genre":         "",
        "is_instrumental": False,
        "labeled":       True,       # ← gates preprocessing
        "custom_tag":    "",
        "prompt_override": "",
    }
]

dataset_json = str(Path(work_dir) / "dataset.json")
with open(dataset_json, "w") as f:
    json.dump(dataset, f, indent=2)
ok(f"Wrote dataset.json: {dataset_json}")

# ---------------------------------------------------------------------------
# Step 3: Load AceStepHandler
# ---------------------------------------------------------------------------
section("STEP 3: Load AceStepHandler")

dit = AceStepHandler()
dit.initialize_service(
    project_root=CHECKPOINT_DIR,
    config_path=DIT_MODEL,
    device=DEVICE,
    offload_to_cpu=False,
)
ok("AceStepHandler initialized")

# ---------------------------------------------------------------------------
# Step 4: Load dataset into DatasetBuilder + preprocess
# ---------------------------------------------------------------------------
section("STEP 4: Load dataset into builder + preprocess")

builder = DatasetBuilder()
info(f"builder.samples before load: {builder.samples}")

# Try loading via JSON file — this is how Gradio UI populates builder.samples
loaded = False
for method_name in ["load_dataset", "load_from_json", "load_json", "from_json"]:
    if hasattr(builder, method_name):
        info(f"Found method: {method_name}{inspect.signature(getattr(builder, method_name))}")
        try:
            getattr(builder, method_name)(dataset_json)
            info(f"builder.samples after {method_name}: {len(builder.samples)} items")
            loaded = True
            break
        except Exception as e:
            info(f"  {method_name} failed: {e}")

if not loaded:
    # Fallback: set samples directly using AudioSample objects
    info("No load method found — setting builder.samples directly from AudioSample objects")
    samples = []
    for entry in dataset:
        s = AudioSample()
        for k, v in entry.items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, v)
        samples.append(s)
    builder.samples = samples
    info(f"builder.samples set to {len(builder.samples)} items")
    info(f"Sample labeled={builder.samples[0].labeled}, audio_path={builder.samples[0].audio_path}")

# Now preprocess
info("Calling preprocess_to_tensors(dit, tensors_dir)...")
try:
    output_paths, status = builder.preprocess_to_tensors(dit, str(tensors_dir))
    info(f"Status:  {status}")
    info(f"Outputs: {output_paths}")
except Exception as e:
    fail(f"preprocess_to_tensors failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Verify
all_files = list(tensors_dir.rglob("*"))
tensor_files = [f for f in all_files if f.is_file()]
if tensor_files:
    ok(f"Tensor files created:")
    for f in tensor_files:
        info(f"  {f.name}  ({f.stat().st_size // 1024}KB)")
else:
    fail("No tensor files found")
    info(f"Dir contents: {all_files}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 5: LoRA training
# ---------------------------------------------------------------------------
section("STEP 5: LoRA training (5 epochs)")

# Confirmed: r (not rank), alpha, dropout, target_modules, bias
lora_cfg = LoRAConfig(
    r              = TEST_LORA_RANK,
    alpha          = TEST_LORA_RANK,
    dropout        = 0.05,
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"],
)
ok(f"LoRAConfig: r={TEST_LORA_RANK}")

# Confirmed: no checkpoint_dir/dit_model fields
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

# Confirmed: LoRATrainer(dit_handler, lora_config, training_config)
trainer = LoRATrainer(dit, lora_cfg, train_cfg)
ok("LoRATrainer instantiated")

# Introspect train() — need to know if it takes tensors_dir or not
train_sig = inspect.signature(trainer.train)
info(f"trainer.train signature: {train_sig}")

# Try with tensors_dir first, fall back to no args
info(f"Training for {TEST_MAX_EPOCHS} epochs...")
try:
    t0     = time.time()
    result = trainer.train(str(tensors_dir))
    ok(f"train(tensors_dir) succeeded in {int(time.time()-t0)}s → {result}")
except TypeError:
    try:
        t0     = time.time()
        result = trainer.train()
        ok(f"train() succeeded in {int(time.time()-t0)}s → {result}")
    except Exception as e:
        fail(f"train() failed: {e}")
        traceback.print_exc()
        sys.exit(1)
except Exception as e:
    fail(f"Training error: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 6: Verify output
# ---------------------------------------------------------------------------
section("STEP 6: Verify LoRA output")

info("Files in lora_output:")
for f in sorted(lora_out.rglob("*")):
    if f.is_file():
        info(f"  {f.relative_to(lora_out)}  ({f.stat().st_size // 1024}KB)")

safetensors = list(lora_out.rglob("*.safetensors"))
if safetensors:
    ok(f"LoRA weights found: {[f.name for f in safetensors]}")
else:
    fail("No .safetensors found")

if Path(OUTPUT_DIR).exists():
    shutil.rmtree(OUTPUT_DIR)
shutil.copytree(str(lora_out), OUTPUT_DIR)
ok(f"Copied to {OUTPUT_DIR}")
shutil.rmtree(work_dir, ignore_errors=True)

section("ALL DONE — paste output to get final training_handler.py")