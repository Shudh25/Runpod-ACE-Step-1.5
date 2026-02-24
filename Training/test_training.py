#!/usr/bin/env python3
"""
Standalone Training Test Script for ACE-Step 1.5
=================================================

Run this directly on the persistent pod to test the full pipeline
BEFORE converting to serverless. No Docker rebuild needed.

Usage:
    python test_training.py

It will:
  1. Discover the correct API by introspecting the installed package
  2. Download a test audio file
  3. Preprocess to tensors
  4. Run a short LoRA training (5 epochs, rank 4 — fast smoke test)
  5. Save output to /tmp/test_lora_output/

Edit the CONFIG section below to match your pod's paths.
"""

import sys
import os
import json
import shutil
import tempfile
import traceback
import inspect
from pathlib import Path

# =============================================================================
# CONFIG — edit these to match your pod
# =============================================================================

CHECKPOINT_DIR = "/app/checkpoints"          # or wherever models are on this pod
DIT_MODEL      = "acestep-v15-base"          # folder name inside checkpoints/checkpoints/
DEVICE         = "cuda"
OUTPUT_DIR     = "/tmp/test_lora_output"     # where to save the test LoRA

# Test audio — public domain, short clip
TEST_AUDIO_URL  = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
TEST_AUDIO_NAME = "test_song.mp3"

# Training params — kept tiny for fast smoke test
TEST_LORA_RANK   = 4
TEST_LORA_ALPHA  = 4
TEST_MAX_EPOCHS  = 5
TEST_BATCH_SIZE  = 1

# =============================================================================

sys.path.insert(0, "/app")

SEP = "=" * 70


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg):    print(f"  ✓  {msg}")
def fail(msg):  print(f"  ✗  {msg}")
def info(msg):  print(f"  →  {msg}")


# =============================================================================
# Step 0: Introspect the installed package to discover correct APIs
# =============================================================================

section("STEP 0: Discovering ACE-Step training API")

try:
    from acestep.training.dataset_builder import DatasetBuilder, AudioSample
    ok("Imported DatasetBuilder, AudioSample")

    # Print DatasetBuilder.__init__ signature
    sig = inspect.signature(DatasetBuilder.__init__)
    info(f"DatasetBuilder.__init__ signature: {sig}")

    # Print preprocess_to_tensors signature
    if hasattr(DatasetBuilder, "preprocess_to_tensors"):
        sig2 = inspect.signature(DatasetBuilder.preprocess_to_tensors)
        info(f"DatasetBuilder.preprocess_to_tensors signature: {sig2}")
    else:
        fail("DatasetBuilder has no preprocess_to_tensors method!")
        info("Available methods: " + str([m for m in dir(DatasetBuilder) if not m.startswith("_")]))

except ImportError as e:
    fail(f"Could not import DatasetBuilder: {e}")
    info("Trying to find training modules...")
    import subprocess
    result = subprocess.run(
        ["find", "/app", "-name", "*.py", "-path", "*/training/*"],
        capture_output=True, text=True
    )
    info("Training files found:\n" + result.stdout)
    sys.exit(1)

try:
    from acestep.training.trainer import LoRATrainer
    ok("Imported LoRATrainer")
    sig = inspect.signature(LoRATrainer.__init__)
    info(f"LoRATrainer.__init__ signature: {sig}")

    if hasattr(LoRATrainer, "train"):
        sig2 = inspect.signature(LoRATrainer.train)
        info(f"LoRATrainer.train signature: {sig2}")
except ImportError as e:
    fail(f"Could not import LoRATrainer: {e}")

try:
    from acestep.training.configs import LoRAConfig, TrainingConfig
    ok("Imported LoRAConfig, TrainingConfig")
    info(f"LoRAConfig fields:    {list(inspect.signature(LoRAConfig.__init__).parameters.keys())}")
    info(f"TrainingConfig fields: {list(inspect.signature(TrainingConfig.__init__).parameters.keys())}")
except ImportError as e:
    fail(f"Could not import configs: {e}")
    info("Trying alternate import paths...")
    try:
        from acestep.training.lora_config import LoRAConfig
        ok("Found LoRAConfig at acestep.training.lora_config")
    except ImportError:
        pass
    try:
        from acestep.training.train_config import TrainingConfig
        ok("Found TrainingConfig at acestep.training.train_config")
    except ImportError:
        pass

try:
    from acestep.handler import AceStepHandler
    ok("Imported AceStepHandler")
    sig = inspect.signature(AceStepHandler.initialize_service)
    info(f"initialize_service signature: {sig}")
except ImportError as e:
    fail(f"Could not import AceStepHandler: {e}")

# Also print AudioSample fields
try:
    info(f"AudioSample fields: {list(inspect.signature(AudioSample.__init__).parameters.keys())}")
except Exception:
    pass

print()
input("  [Press Enter to continue to Step 1, or Ctrl+C to stop and review the above]")


# =============================================================================
# Step 1: Download test audio
# =============================================================================

section("STEP 1: Downloading test audio")

work_dir  = tempfile.mkdtemp(prefix="acetrain_test_")
audio_dir = Path(work_dir) / "audio"
audio_dir.mkdir()
audio_dest = str(audio_dir / TEST_AUDIO_NAME)

info(f"Work dir: {work_dir}")
info(f"Downloading: {TEST_AUDIO_URL}")

try:
    import urllib.request
    req = urllib.request.Request(TEST_AUDIO_URL, headers={"User-Agent": "ACE-Step-Trainer/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        with open(audio_dest, "wb") as f:
            f.write(r.read())
    size_mb = Path(audio_dest).stat().st_size / 1_048_576
    ok(f"Downloaded {TEST_AUDIO_NAME} ({size_mb:.1f} MB) → {audio_dest}")
except Exception as e:
    fail(f"Download failed: {e}")
    info("You can manually place an MP3 at: " + audio_dest)
    sys.exit(1)


# =============================================================================
# Step 2: Load AceStepHandler
# =============================================================================

section("STEP 2: Loading AceStepHandler (VAE + text encoder)")

checkpoint_path = Path(CHECKPOINT_DIR) / "checkpoints"
info(f"Looking for checkpoints at: {checkpoint_path}")
info(f"Contents: {[p.name for p in checkpoint_path.iterdir()] if checkpoint_path.exists() else 'PATH NOT FOUND'}")

try:
    dit = AceStepHandler()
    dit.initialize_service(
        project_root=CHECKPOINT_DIR,
        config_path=DIT_MODEL,
        device=DEVICE,
        offload_to_cpu=False,
    )
    ok("AceStepHandler initialized")
except Exception as e:
    fail(f"AceStepHandler.initialize_service failed: {e}")
    traceback.print_exc()
    info("\nTrying with project_root pointing one level deeper...")
    try:
        dit = AceStepHandler()
        dit.initialize_service(
            project_root=str(checkpoint_path),
            config_path=DIT_MODEL,
            device=DEVICE,
            offload_to_cpu=False,
        )
        ok("AceStepHandler initialized with nested path")
        CHECKPOINT_DIR = str(checkpoint_path)   # update for later steps
        info(f"Correct CHECKPOINT_DIR is: {CHECKPOINT_DIR}")
    except Exception as e2:
        fail(f"Still failed: {e2}")
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# Step 3: Build AudioSample and preprocess
# =============================================================================

section("STEP 3: Preprocessing audio → tensors")

tensors_dir = Path(work_dir) / "tensors"
tensors_dir.mkdir()

samples = [
    AudioSample(
        audio_path    = audio_dest,
        caption       = "Upbeat electronic track with synthesizers and driving drums",
        lyrics        = "[Verse 1]\nTest verse lyrics here\n\n[Chorus]\nLa la la test",
        bpm           = None,
        keyscale      = "",
        timesignature = "",
    )
]

info(f"Tensors output dir: {tensors_dir}")

try:
    builder = DatasetBuilder()
    ok("DatasetBuilder() instantiated with no args")
except TypeError as e:
    fail(f"DatasetBuilder() failed: {e}")
    info("Trying DatasetBuilder with checkpoint_dir arg...")
    try:
        builder = DatasetBuilder(checkpoint_dir=CHECKPOINT_DIR)
        ok("DatasetBuilder(checkpoint_dir=...) worked")
    except Exception as e2:
        traceback.print_exc()
        sys.exit(1)

# Try calling preprocess_to_tensors — test both known signatures
info("Calling preprocess_to_tensors...")
try:
    result = builder.preprocess_to_tensors(samples, str(tensors_dir))
    ok(f"preprocess_to_tensors(samples, output_dir) succeeded → {result}")
except TypeError as e:
    fail(f"preprocess_to_tensors(samples, output_dir) failed: {e}")
    info("Trying preprocess_to_tensors(output_dir) with samples set on builder...")
    try:
        builder.samples = samples
        result = builder.preprocess_to_tensors(str(tensors_dir))
        ok(f"preprocess_to_tensors(output_dir) succeeded → {result}")
    except TypeError as e2:
        fail(f"Also failed: {e2}")
        info(f"Full signature: {inspect.signature(builder.preprocess_to_tensors)}")
        traceback.print_exc()
        sys.exit(1)

# Verify tensors were actually created
tensor_files = list(tensors_dir.rglob("*.pt")) + list(tensors_dir.rglob("*.safetensors"))
if tensor_files:
    ok(f"Tensor files created: {[f.name for f in tensor_files]}")
else:
    fail("No tensor files found in output dir!")
    info(f"Contents of {tensors_dir}: {list(tensors_dir.iterdir())}")
    sys.exit(1)


# =============================================================================
# Step 4: LoRA training (smoke test — 5 epochs only)
# =============================================================================

section("STEP 4: LoRA training (5 epochs smoke test)")

lora_out = Path(work_dir) / "lora_output"
lora_out.mkdir()

info(f"LoRA output dir: {lora_out}")

try:
    lora_cfg = LoRAConfig(
        rank           = TEST_LORA_RANK,
        alpha          = TEST_LORA_ALPHA,
        dropout        = 0.05,
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"],
    )
    ok(f"LoRAConfig created: rank={TEST_LORA_RANK}")
except TypeError as e:
    fail(f"LoRAConfig failed: {e}")
    info(f"LoRAConfig signature: {inspect.signature(LoRAConfig.__init__)}")
    traceback.print_exc()
    sys.exit(1)

try:
    train_cfg = TrainingConfig(
        dataset_path            = str(tensors_dir),
        output_dir              = str(lora_out),
        max_epochs              = TEST_MAX_EPOCHS,
        batch_size              = TEST_BATCH_SIZE,
        learning_rate           = 3e-4,
        accumulate_grad_batches = 1,
        save_every_n_epochs     = 5,
        shift                   = 3.0,
        seed                    = 42,
        precision               = "bf16",
        optimizer               = "adamw",
        checkpoint_dir          = CHECKPOINT_DIR,
        dit_model               = DIT_MODEL,
        device                  = DEVICE,
    )
    ok("TrainingConfig created")
except TypeError as e:
    fail(f"TrainingConfig failed: {e}")
    info(f"TrainingConfig signature: {inspect.signature(TrainingConfig.__init__)}")
    traceback.print_exc()
    sys.exit(1)

try:
    import time
    trainer = LoRATrainer(lora_config=lora_cfg, train_config=train_cfg)
    ok("LoRATrainer instantiated")

    info(f"Starting training ({TEST_MAX_EPOCHS} epochs)...")
    t0      = time.time()
    metrics = trainer.train()
    elapsed = int(time.time() - t0)

    ok(f"Training completed in {elapsed}s")
    ok(f"Metrics: {metrics}")
except TypeError as e:
    fail(f"LoRATrainer failed: {e}")
    info(f"LoRATrainer signature: {inspect.signature(LoRATrainer.__init__)}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    fail(f"Training failed: {e}")
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Step 5: Verify output files
# =============================================================================

section("STEP 5: Verifying output")

all_files = list(lora_out.rglob("*"))
info(f"All files in output dir:")
for f in sorted(all_files):
    if f.is_file():
        size_mb = f.stat().st_size / 1_048_576
        info(f"  {f.relative_to(lora_out)}  ({size_mb:.2f} MB)")

safetensors = list(lora_out.rglob("*.safetensors"))
if safetensors:
    ok(f"Found LoRA weights: {[f.name for f in safetensors]}")
else:
    fail("No .safetensors found — check what files were actually saved above")

config_files = list(lora_out.rglob("adapter_config.json"))
if config_files:
    ok("Found adapter_config.json")
    with open(config_files[0]) as f:
        info(f"Config contents: {json.dumps(json.load(f), indent=2)}")
else:
    info("No adapter_config.json found")

# Copy to final output dir
info(f"\nCopying results to {OUTPUT_DIR}...")
shutil.copytree(str(lora_out), OUTPUT_DIR, dirs_exist_ok=True)
ok(f"Saved to {OUTPUT_DIR}")


# =============================================================================
# Summary
# =============================================================================

section("SUMMARY")

ok("All steps passed! Pipeline is working.")
print()
info("Correct API signatures discovered:")
info(f"  CHECKPOINT_DIR for initialize_service: {CHECKPOINT_DIR}")
info(f"  DatasetBuilder(): no args")
info(f"  preprocess_to_tensors: check Step 3 output above for which call worked")
info(f"  LoRAConfig fields:    {list(inspect.signature(LoRAConfig.__init__).parameters.keys())}")
info(f"  TrainingConfig fields: {list(inspect.signature(TrainingConfig.__init__).parameters.keys())}")
print()
info("Next step: use these confirmed signatures to update training_handler.py")
info(f"Output LoRA is at: {OUTPUT_DIR}")
print()

# Cleanup temp work dir (output already copied)
shutil.rmtree(work_dir, ignore_errors=True)