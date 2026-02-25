#!/usr/bin/env python3
"""
ACE-Step 1.5 — RunPod Serverless Training Handler
===================================================
Pipeline:
  Stage 1 · Download audio → write dataset JSON → preprocess to .pt tensors
  Stage 2 · Train LoRA adapter via LoRATrainer (Lightning Fabric + PEFT)
  Stage 3 · Copy adapter_model.safetensors to Network Volume

Environment Variables (set on RunPod endpoint):
  CHECKPOINT_DIR   Path to model checkpoints  (default: /app/checkpoints)
  DIT_MODEL        DiT model folder name       (default: acestep-v15-base)
  DEVICE           Torch device                (default: cuda)
  LORA_OUTPUT_DIR  Network Volume destination  (default: /runpod-volume/loras)

Input JSON schema — see INPUT SCHEMA section below.
Output JSON schema — see OUTPUT SCHEMA section below.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import gc
import inspect
import os
import shutil
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Optional

# ── ACE-Step package path ─────────────────────────────────────────────────────
sys.path.insert(0, "/app")

import runpod

# =============================================================================
# Configuration  (all overridable via environment variables)
# =============================================================================
CHECKPOINT_DIR  = os.environ.get("CHECKPOINT_DIR",  "/app/checkpoints")
DIT_MODEL       = os.environ.get("DIT_MODEL",        "acestep-v15-base")
DEVICE          = os.environ.get("DEVICE",           "cuda")
LORA_OUTPUT_DIR = os.environ.get("LORA_OUTPUT_DIR",  "/runpod-volume/loras")

# =============================================================================
# INPUT SCHEMA (reference)
# =============================================================================
# {
#   "audio_files": [                              ← REQUIRED, 1–50 files
#     {
#       "url":           "https://...",           ← REQUIRED
#       "filename":      "track01.mp3",           ← optional, used for logging
#       "caption":       "Cinematic orchestra…",  ← REQUIRED for good results
#       "lyrics":        "[Verse]\n…\n[Chorus]",  ← optional
#       "genre":         "cinematic, epic",       ← optional (helps conditioning)
#       "bpm":           120,                     ← optional int
#       "keyscale":      "D minor",               ← optional string
#       "timesignature": "4",                     ← optional string
#       "is_instrumental": false,                 ← optional bool
#       "language":      "en",                    ← optional, default "en"
#       "custom_tag":    ""                       ← optional activation tag
#     }
#   ],
#
#   "lora_name":             "my_style",          ← output folder name on Volume
#
#   "lora_rank":             64,                  ← LoRA rank (4–128 typical)
#   "lora_alpha":            64,                  ← LoRA scaling (= rank is common)
#   "lora_dropout":          0.05,
#   "target_modules":        ["q_proj","k_proj","v_proj","o_proj"],
#
#   "max_epochs":            500,
#   "batch_size":            1,                   ← 1 is safe; 2 needs ~40GB VRAM
#   "learning_rate":         1e-4,
#   "gradient_accumulation": 1,
#   "save_every_n_epochs":   100,
#   "shift":                 3.0,
#   "seed":                  42
# }

# =============================================================================
# OUTPUT SCHEMA (reference)
# =============================================================================
# {
#   "output": {
#     "status":                "success",
#     "lora_name":             "my_style",
#     "lora_path":             "/runpod-volume/loras/my_style",
#     "lora_files":            ["adapter_model.safetensors", "adapter_config.json"],
#     "epochs_trained":        500,
#     "final_loss":            0.042,
#     "training_time_seconds": 3612
#   }
# }


# =============================================================================
# Utilities
# =============================================================================

def _log(tag: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}][{tag}] {msg}", flush=True)


def _download_file(url: str, dest: str, timeout: int = 300) -> None:
    """Download a file over HTTP(S) with a meaningful User-Agent."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ACE-Step-Trainer/1.5"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        with open(dest, "wb") as fh:
            shutil.copyfileobj(response, fh)


def _find_lora_weights(output_dir: str) -> Optional[str]:
    """
    Return the directory containing the final LoRA weights.

    Confirmed output structure (from live run):
      {output_dir}/
        final/
          adapter/                        ← adapter_model.safetensors lives here
            adapter_model.safetensors
            adapter_config.json
        checkpoints/
          epoch_100/
            adapter/
              adapter_model.safetensors
          epoch_200/
            adapter/
              adapter_model.safetensors

    Priority:
      1. final/adapter/  — the definitive export
      2. final/          — direct safetensors in final (older layout)
      3. Latest checkpoints/epoch_N/adapter/
      4. Latest checkpoints/epoch_N/
      5. Recursive search for any .safetensors under output_dir
    """
    out = Path(output_dir)

    # 1. final/adapter/ — confirmed from live logs
    final_adapter = out / "final" / "adapter"
    if final_adapter.is_dir() and list(final_adapter.glob("*.safetensors")):
        return str(final_adapter)

    # 2. final/ directly (older layout without adapter subfolder)
    final = out / "final"
    if final.is_dir() and list(final.glob("*.safetensors")):
        return str(final)

    # 3. Latest checkpoints/epoch_N/adapter/
    epoch_dirs = sorted(
        [d for d in (out / "checkpoints").glob("epoch_*") if d.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    ) if (out / "checkpoints").is_dir() else []

    if not epoch_dirs:
        # Fallback: epoch_* directly in output_dir (older layout)
        epoch_dirs = sorted(
            [d for d in out.glob("epoch_*") if d.is_dir()],
            key=lambda p: int(p.name.split("_")[1]),
        )

    for epoch_dir in reversed(epoch_dirs):
        adapter_sub = epoch_dir / "adapter"
        if adapter_sub.is_dir() and list(adapter_sub.glob("*.safetensors")):
            return str(adapter_sub)
        if list(epoch_dir.glob("*.safetensors")):
            return str(epoch_dir)

    # 4. Any .safetensors anywhere under output_dir
    all_sf = list(out.rglob("*.safetensors"))
    if all_sf:
        return str(all_sf[0].parent)

    return None


def _reflect_class(cls) -> dict:
    """
    Return {field_name: default} for a dataclass.
    Used to adapt to whatever field names the installed ACE-Step version uses.
    """
    result = {}
    if hasattr(cls, "__dataclass_fields__"):
        for name, field in cls.__dataclass_fields__.items():
            result[name] = field.default
    return result


# =============================================================================
# Stage 1 — Download & Preprocess
# =============================================================================

def _stage1_preprocess(audio_files: list, work_dir: str) -> tuple[str, Any]:
    """
    Download audio files, construct AudioSample objects directly,
    preprocess all samples to .pt tensor files.

    Returns
    -------
    tensors_dir : str           Path to directory containing .pt files + manifest.json
    dit         : AceStepHandler  Initialized handler (reused in Stage 2)
    """
    from acestep.training.dataset_builder import DatasetBuilder, AudioSample
    from acestep.handler import AceStepHandler

    import uuid as _uuid
    audio_dir   = Path(work_dir) / "audio"
    # IMPORTANT: path_safety.py enforces ALL paths passed to LoRATrainer
    # must resolve under root='/app'. tensors_dir is passed to
    # train_from_preprocessed() which calls safe_path() on it.
    # /tmp/ is outside /app — move tensors to /app/outputs/ like lora_out.
    tensors_dir = Path("/app/outputs") / f"tensors_{_uuid.uuid4().hex[:8]}"
    audio_dir.mkdir(parents=True)
    tensors_dir.mkdir(parents=True, exist_ok=True)

    # ── Download audio ───────────────────────────────────────────────────────
    _log("Stage1", f"Downloading {len(audio_files)} audio file(s)...")

    # Reflect AudioSample's actual field names once so we know what's valid
    known_fields = set(_reflect_class(AudioSample).keys())
    _log("Stage1", f"AudioSample fields: {sorted(known_fields)}")

    raw_samples = []   # list of dicts; converted to AudioSample objects below

    for i, af in enumerate(audio_files):
        url      = str(af.get("url", "")).strip()
        filename = af.get("filename", f"audio_{i:03d}.mp3")
        dest     = str(audio_dir / filename)

        if not url:
            raise ValueError(f"audio_files[{i}]: missing required 'url'")

        _log("Stage1", f"  [{i+1}/{len(audio_files)}] {filename}")
        _download_file(url, dest)
        size_kb = Path(dest).stat().st_size // 1024
        _log("Stage1", f"  → {size_kb} KB")

        lyrics = af.get("lyrics", "") or ""

        raw_samples.append({
            "audio_path":      dest,
            "filename":        filename,
            "caption":         af.get("caption",       ""),
            "genre":           af.get("genre",         ""),
            # Both field names for lyrics — different ACE-Step builds use different ones
            "lyrics":          lyrics,
            "raw_lyrics":      lyrics,
            "language":        af.get("language",      "en"),
            "bpm":             af.get("bpm"),           # None = unknown
            "keyscale":        af.get("keyscale",      "") or "",
            "timesignature":   str(af.get("timesignature", "") or ""),
            "duration":        None,
            "is_instrumental": bool(af.get("is_instrumental", False)),
            "custom_tag":      af.get("custom_tag",    "") or "",
            "prompt_override": "",
            # These flags gate preprocessing — both field-name variants set True
            "labeled":         True,
            "is_labeled":      True,
        })

    # ── Initialize AceStepHandler ────────────────────────────────────────────
    # Required for:
    #   • VAE encoding   (audio → latents [T,64])    during preprocessing
    #   • Text encoding  (caption → embeddings [L,D]) during preprocessing
    #   • DiT weights                                  during Stage 2 training
    #
    # lm_model_path="" — explicitly disables LM model loading.
    # We didn't download any LM weights to keep the image lean.
    _log("Stage1", f"Initializing AceStepHandler (model={DIT_MODEL}, device={DEVICE})...")

    dit = AceStepHandler()

    init_sig = inspect.signature(dit.initialize_service)
    init_kwargs: dict = {
        "project_root":   CHECKPOINT_DIR,
        "config_path":    DIT_MODEL,
        "device":         DEVICE,
        "offload_to_cpu": False,
    }
    if "lm_model_path" in init_sig.parameters:
        init_kwargs["lm_model_path"] = ""   # suppress LM auto-detection

    dit.initialize_service(**init_kwargs)
    _log("Stage1", "AceStepHandler ready")

    # ── Build AudioSample objects and assign directly to builder.samples ─────
    # WHY: load_dataset() expects ACE-Step's own serialisation format
    # (a dict with "metadata" + "samples" keys produced by save_dataset()).
    # Passing a bare list makes it silently return 0 samples.
    # Constructing AudioSample objects directly and assigning to builder.samples
    # is the only robust path when we haven't gone through the Gradio UI save flow.
    _log("Stage1", "Constructing AudioSample objects...")

    built_samples = []
    for data in raw_samples:
        s = AudioSample()
        for key, val in data.items():
            # Only set fields that exist on this version of AudioSample
            if key in known_fields:
                # Don't clobber boolean True flags with None
                if val is not None:
                    setattr(s, key, val)
            # For fields not in known_fields, attempt setattr anyway —
            # some builds may have extra fields not captured by __dataclass_fields__
            else:
                try:
                    setattr(s, key, val)
                except (AttributeError, TypeError):
                    pass
        built_samples.append(s)

    builder = DatasetBuilder()
    builder.samples = built_samples

    n = len(builder.samples)
    _log("Stage1", f"Builder has {n} sample(s)")

    if n == 0:
        raise RuntimeError(
            "AudioSample list is empty — this should not happen. "
            f"audio_files had {len(audio_files)} entries."
        )

    # Belt-and-suspenders: confirm labeled flags survived the assignment
    for idx, s in enumerate(builder.samples):
        for attr in ("labeled", "is_labeled"):
            current = getattr(s, attr, None)
            if current is not True:
                try:
                    setattr(s, attr, True)
                    _log("Stage1", f"  sample[{idx}].{attr} force-set True (was {current!r})")
                except (AttributeError, TypeError):
                    pass

    # Log first sample for verification
    s0 = builder.samples[0]
    _log("Stage1", (
        f"Sample[0] check — "
        f"audio_path={getattr(s0,'audio_path','?')}  "
        f"caption={str(getattr(s0,'caption',''))[:60]!r}  "
        f"labeled={getattr(s0,'labeled', getattr(s0,'is_labeled','?'))}"
    ))

    # ── Preprocess ────────────────────────────────────────────────────────────
    # preprocess_to_tensors(dit_handler, output_dir)
    #   • Encodes each sample's audio through the VAE  → target_latents [T,64]
    #   • Encodes caption+lyrics through Qwen3         → encoder_hidden_states [L,D]
    #   • Saves per-sample .pt files + manifest.json
    _log("Stage1", f"Preprocessing {n} sample(s) → {tensors_dir}")

    try:
        output_paths, status = builder.preprocess_to_tensors(dit, str(tensors_dir))
    except TypeError:
        # Some builds may have a different signature — try without dit
        _log("Stage1", "WARN: preprocess_to_tensors(dit, dir) raised TypeError — trying (dir,)")
        output_paths, status = builder.preprocess_to_tensors(str(tensors_dir))

    _log("Stage1", f"Preprocessing status: {status}")

    tensor_files = list(Path(tensors_dir).rglob("*.pt"))
    if not tensor_files:
        manifest = Path(tensors_dir) / "manifest.json"
        _log("Stage1", f"Manifest exists: {manifest.exists()}")
        _log("Stage1", f"tensors_dir contents: {list(Path(tensors_dir).iterdir())}")
        raise RuntimeError(
            f"Preprocessing succeeded ({status}) but produced no .pt tensor files. "
            "Possible causes: all samples had labeled=False, or audio loading failed."
        )

    total_mb = sum(f.stat().st_size for f in tensor_files) / 1e6
    _log("Stage1", f"Created {len(tensor_files)} tensor file(s) ({total_mb:.1f} MB)")

    return str(tensors_dir), dit


# =============================================================================
# Stage 2 — LoRA Training
# =============================================================================

def _stage2_train(
    tensors_dir: str,
    dit: Any,
    work_dir: str,
    inp: dict,
) -> tuple[Any, str, int]:
    """
    Instantiate LoRATrainer and run the full training loop.

    Returns
    -------
    result    : dict | Any    Return value from trainer.train()
    lora_out  : str           Path to the training output directory
    elapsed   : int           Seconds taken
    """
    from acestep.training.trainer import LoRATrainer
    from acestep.training.configs import LoRAConfig, TrainingConfig

    # IMPORTANT: ACE-Step path_safety.py enforces output_dir must be under /app
    # safe_path() in LoRATrainer.__init__ rejects any path outside root='/app'.
    # Writing to /tmp/ raises: "Path escapes safe root ... root='/app'"
    # Fix: use /app/outputs/<unique_id> — always inside /app, globally unique.
    import uuid
    lora_out = f"/app/outputs/lora_{uuid.uuid4().hex[:8]}"
    Path(lora_out).mkdir(parents=True, exist_ok=True)

    # ── Reflect actual field names ────────────────────────────────────────────
    # LoRAConfig uses  'r'  (not 'rank') — confirmed from source inspection.
    # We still reflect to handle potential future renames gracefully.
    lora_fields  = _reflect_class(LoRAConfig)
    train_fields = _reflect_class(TrainingConfig)
    _log("Stage2", f"LoRAConfig fields:     {list(lora_fields.keys())}")
    _log("Stage2", f"TrainingConfig fields: {list(train_fields.keys())}")

    # ── Build LoRAConfig ──────────────────────────────────────────────────────
    # rank field: ACE-Step uses 'r' (PEFT convention), not 'rank'
    rank_val = int(inp.get("lora_rank", 64))
    lora_kwargs: dict = {}

    if "r" in lora_fields:
        lora_kwargs["r"] = rank_val
    elif "rank" in lora_fields:
        lora_kwargs["rank"] = rank_val
    else:
        lora_kwargs["r"] = rank_val      # best guess

    lora_kwargs["alpha"]          = float(inp.get("lora_alpha",   64))
    lora_kwargs["dropout"]        = float(inp.get("lora_dropout", 0.05))
    lora_kwargs["target_modules"] = inp.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj"],   # confirmed from configs.py
    )

    lora_cfg = LoRAConfig(**lora_kwargs)
    _log("Stage2", f"LoRAConfig: {lora_kwargs}")

    # ── Build TrainingConfig ──────────────────────────────────────────────────
    # Field name mapping:
    #   max_epochs              — total training epochs
    #   gradient_accumulation_steps  — (NOT accumulate_grad_batches)
    #   mixed_precision         — (NOT precision)
    #   shift                   — flow-matching timestep shift (3.0 for turbo/base)
    train_kwargs: dict = {
        "output_dir":  lora_out,
        "shift":       float(inp.get("shift",   3.0)),
        "seed":        int(inp.get("seed",      42)),
    }

    # Gracefully map field names that vary across ACE-Step builds
    def _maybe(field_primary, field_fallback, value):
        if field_primary in train_fields:
            train_kwargs[field_primary] = value
        elif field_fallback and field_fallback in train_fields:
            train_kwargs[field_fallback] = value
        else:
            train_kwargs[field_primary] = value   # best-effort

    _maybe("max_epochs",               "num_epochs",              int(inp.get("max_epochs",          500)))
    _maybe("batch_size",               None,                      int(inp.get("batch_size",            1)))
    _maybe("learning_rate",            "lr",                      float(inp.get("learning_rate",    1e-4)))
    _maybe("gradient_accumulation_steps", "accumulate_grad_batches", int(inp.get("gradient_accumulation", 1)))
    _maybe("save_every_n_epochs",      "checkpoint_every",        int(inp.get("save_every_n_epochs", 100)))
    _maybe("mixed_precision",          "precision",               inp.get("precision", "bf16"))

    train_cfg = TrainingConfig(**train_kwargs)
    _log("Stage2", (
        f"TrainingConfig: epochs={inp.get('max_epochs',500)} "
        f"batch={inp.get('batch_size',1)} "
        f"lr={inp.get('learning_rate',1e-4)} "
        f"precision={inp.get('precision','bf16')}"
    ))

    # ── Instantiate LoRATrainer ───────────────────────────────────────────────
    # Confirmed signature: LoRATrainer(dit_handler, lora_config, training_config)
    trainer = LoRATrainer(dit, lora_cfg, train_cfg)
    _log("Stage2", "LoRATrainer instantiated")

    # Confirmed from live logs (run 4):
    #   LoRATrainer methods: ['_train_basic', '_train_with_fabric', 'stop', 'train_from_preprocessed']
    #   _train_with_fabric(data_module: PreprocessedDataModule,
    #                      training_state: Optional[Dict],
    #                      resume_from: Optional[str] = None) -> Generator
    all_methods = [m for m in dir(trainer) if callable(getattr(trainer, m, None)) and not m.startswith("__")]
    _log("Stage2", f"LoRATrainer methods: {all_methods}")

    from acestep.training.data_module import PreprocessedDataModule

    # training_state mirrors the Gradio UI's stop-signal dict
    training_state = {"is_training": True, "should_stop": False}

    # ── Try train_from_preprocessed first — the clean public API ─────────────
    t0 = time.time()

    if hasattr(trainer, "train_from_preprocessed"):
        sig = inspect.signature(trainer.train_from_preprocessed)
        _log("Stage2", f"train_from_preprocessed sig: {sig}")
        params = list(sig.parameters.keys())
        # Likely: train_from_preprocessed(tensor_dir, training_state=None, ...)
        try:
            if "training_state" in params:
                result = trainer.train_from_preprocessed(tensors_dir, training_state=training_state)
            else:
                result = trainer.train_from_preprocessed(tensors_dir)
            _log("Stage2", "train_from_preprocessed() called OK")
        except TypeError as exc:
            _log("Stage2", f"train_from_preprocessed failed ({exc}) — falling through to _train_with_fabric")
            result = None
    else:
        result = None

    # ── Fall back to _train_with_fabric(data_module, training_state) ─────────
    if result is None:
        _log("Stage2", "Building PreprocessedDataModule for _train_with_fabric...")
        data_module = PreprocessedDataModule(tensor_dir=tensors_dir, batch_size=train_cfg.batch_size)
        _log("Stage2", f"PreprocessedDataModule ready, calling _train_with_fabric")
        result = trainer._train_with_fabric(data_module, training_state)

    # Exhaust generator — both methods return Generator[Tuple[int, float, str], None, None]
    # Tuple is (epoch: int, loss: float, log_message: str)
    final_result = {}
    if inspect.isgenerator(result) or hasattr(result, "__next__"):
        _log("Stage2", "Exhausting training generator...")
        last_epoch = 0
        last_loss  = None
        step_count = 0
        for item in result:
            step_count += 1
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                epoch, loss = item[0], item[1]
                log_msg = item[2] if len(item) > 2 else ""
                last_epoch = epoch
                last_loss  = loss
                if step_count == 1 or step_count % 50 == 0:
                    _log("Stage2", f"  epoch={epoch}  loss={loss:.6f}  {log_msg}")
            elif isinstance(item, dict):
                last_epoch = item.get("epoch", last_epoch)
                last_loss  = item.get("loss",  item.get("train_loss", last_loss))
                if step_count == 1 or step_count % 50 == 0:
                    _log("Stage2", f"  epoch={last_epoch}  loss={last_loss}")
        _log("Stage2", f"Generator done — {step_count} yields  final_loss={last_loss}")
        final_result = {"epochs_trained": last_epoch, "final_loss": last_loss}
    elif isinstance(result, dict):
        final_result = result

    elapsed = int(time.time() - t0)   # measured AFTER generator is exhausted
    _log("Stage2", f"Training complete in {elapsed}s — {final_result}")
    return final_result, lora_out, elapsed


# =============================================================================
# Stage 3 — Copy to Network Volume
# =============================================================================

def _stage3_save(lora_weights_dir: str, lora_name: str) -> tuple[str, list]:
    """
    Copy LoRA weights from temp working dir to the Network Volume.

    Returns (destination_path, list_of_relative_file_paths)
    """
    dest = Path(LORA_OUTPUT_DIR) / lora_name
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_file in Path(lora_weights_dir).rglob("*"):
        if src_file.is_file():
            rel = src_file.relative_to(lora_weights_dir)
            dst_file = dest / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            copied += 1

    file_list = [
        str(f.relative_to(dest))
        for f in sorted(dest.rglob("*"))
        if f.is_file()
    ]
    _log("Stage3", f"Saved {copied} file(s) → {dest}")
    _log("Stage3", f"Files: {file_list}")
    return str(dest), file_list


# =============================================================================
# Main RunPod handler
# =============================================================================

def handler(job: dict) -> dict:
    """
    RunPod serverless entry point.

    Receives a job dict with an 'input' key containing the training config.
    Returns a result dict (see OUTPUT SCHEMA above).
    """
    inp: dict = job.get("input", {})

    # ── Validate required inputs ─────────────────────────────────────────────
    audio_files = inp.get("audio_files", [])
    if not audio_files:
        return {"error": "Missing required field: 'audio_files' (list of audio dicts with 'url')"}
    if not isinstance(audio_files, list):
        return {"error": "'audio_files' must be a list"}

    lora_name = inp.get("lora_name", f"lora_{int(time.time())}")

    # ── Validate Network Volume ───────────────────────────────────────────────
    # The Network Volume must be mounted before we do any heavy work.
    vol_parent = Path(LORA_OUTPUT_DIR).parent
    if not vol_parent.exists():
        return {
            "error": (
                f"Network Volume not mounted at '{vol_parent}'. "
                "Attach a Network Volume to this RunPod endpoint and ensure "
                f"LORA_OUTPUT_DIR is set correctly (current: {LORA_OUTPUT_DIR})."
            )
        }

    _log("Handler", f"Job started — lora_name='{lora_name}'  files={len(audio_files)}")
    _log("Handler", f"Config: DIT={DIT_MODEL}  device={DEVICE}  checkpoints={CHECKPOINT_DIR}")

    # Verify training packages are importable before any heavy work
    try:
        import lightning  # noqa: F401
        import peft       # noqa: F401
        _log("Handler", f"lightning={lightning.__version__}  peft={peft.__version__}")
    except ImportError as exc:
        return {"error": f"Missing training dependency: {exc}. Rebuild Docker image."}

    # ── Working directory (cleaned up in finally) ─────────────────────────────
    work_dir = tempfile.mkdtemp(prefix="acetrain_")
    _log("Handler", f"Work dir: {work_dir}")

    try:
        # ── Stage 1: Download + Preprocess ───────────────────────────────────
        _log("Handler", "=== Stage 1: Download & Preprocess ===")
        t1 = time.time()
        tensors_dir, dit = _stage1_preprocess(audio_files, work_dir)
        _log("Handler", f"Stage 1 done in {int(time.time()-t1)}s")

        # ── Stage 2: Train ────────────────────────────────────────────────────
        _log("Handler", "=== Stage 2: LoRA Training ===")
        t2 = time.time()
        train_result, lora_out, train_elapsed = _stage2_train(
            tensors_dir, dit, work_dir, inp
        )
        _log("Handler", f"Stage 2 done in {train_elapsed}s")

        # Free GPU memory before copying files
        del dit
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # ── Locate LoRA weights ───────────────────────────────────────────────
        lora_weights_dir = _find_lora_weights(lora_out)
        if lora_weights_dir is None:
            # List what's in the output dir to help debug
            contents = list(Path(lora_out).rglob("*"))
            return {
                "error": (
                    "Training completed but no LoRA weights (.safetensors) found. "
                    f"Output dir contents: {[str(p.relative_to(lora_out)) for p in contents]}"
                )
            }
        _log("Handler", f"LoRA weights located at: {lora_weights_dir}")

        # ── Stage 3: Save to Volume ───────────────────────────────────────────
        _log("Handler", "=== Stage 3: Save to Network Volume ===")
        dest_path, file_list = _stage3_save(lora_weights_dir, lora_name)

        # ── Extract metrics from train result ─────────────────────────────────
        epochs_trained = None
        final_loss     = None
        if isinstance(train_result, dict):
            epochs_trained = (
                train_result.get("epochs_trained")
                or train_result.get("epoch")
                or train_result.get("num_epochs")
            )
            final_loss = (
                train_result.get("final_loss")
                or train_result.get("train_loss")
                or train_result.get("loss")
            )

        total_elapsed = int(time.time() - t1)
        _log("Handler", f"Total time: {total_elapsed}s")

        return {
            "output": {
                "status":                "success",
                "lora_name":             lora_name,
                "lora_path":             dest_path,
                "lora_files":            file_list,
                "epochs_trained":        epochs_trained,
                "final_loss":            final_loss,
                "training_time_seconds": total_elapsed,
            }
        }

    except Exception as exc:
        tb = traceback.format_exc()
        _log("Handler", f"FATAL: {exc}")
        print(tb, flush=True)
        return {
            "error":     str(exc),
            "traceback": tb,
        }

    finally:
        # Clean up audio + tensor temp files (always in /tmp via work_dir)
        shutil.rmtree(work_dir, ignore_errors=True)
        _log("Handler", f"Cleaned up work dir: {work_dir}")
        # lora_out is under /app/outputs/ (outside work_dir) — clean it up too.
        # It only exists if Stage 2 started; guard with try/except NameError.
        try:
            if lora_out and Path(lora_out).exists():
                shutil.rmtree(lora_out, ignore_errors=True)
                _log("Handler", f"Cleaned up lora_out: {lora_out}")
        except NameError:
            pass
        # tensors_dir is also under /app/outputs/ — clean it up too
        try:
            if tensors_dir and Path(str(tensors_dir)).exists():
                shutil.rmtree(str(tensors_dir), ignore_errors=True)
                _log("Handler", f"Cleaned up tensors_dir: {tensors_dir}")
        except NameError:
            pass


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    _log("Startup", f"ACE-Step Training Handler starting")
    _log("Startup", f"CHECKPOINT_DIR  = {CHECKPOINT_DIR}")
    _log("Startup", f"DIT_MODEL       = {DIT_MODEL}")
    _log("Startup", f"DEVICE          = {DEVICE}")
    _log("Startup", f"LORA_OUTPUT_DIR = {LORA_OUTPUT_DIR}")

    # Verify model files exist before registering with RunPod
    base_model = Path(CHECKPOINT_DIR) / "checkpoints" / DIT_MODEL
    if not base_model.exists():
        _log("Startup", f"WARNING: DiT model not found at {base_model}")
    else:
        _log("Startup", f"DiT model found: {base_model}")

    runpod.serverless.start({"handler": handler})