#!/usr/bin/env python3
"""
RunPod Serverless Handler for ACE-Step 1.5

Key fix: pass save_dir to generate_music() so audio files are written to disk.
Fallback: if path is still empty, save the audio tensor manually via torchaudio.

ENV VARS:
  CHECKPOINT_DIR   Default: /app/checkpoints
  DIT_MODEL        Default: acestep-v15-turbo
  LM_MODEL         Default: acestep-5Hz-lm-1.7B
  DEVICE           Default: cuda

INPUT:
{
    "caption":         "Upbeat indie pop with jangly guitars",  // required
    "lyrics":          "[Verse 1]\nHello\n\n[Chorus]\nLa la la",
    "duration":        90,
    "seed":            -1,
    "batch_size":      1,
    "bpm":             null,
    "keyscale":        "",
    "timesignature":   "",
    "vocal_language":  "unknown",
    "inference_steps": 8,
    "guidance_scale":  7.0,
    "shift":           3.0,
    "infer_method":    "ode",
    "thinking":        true,
    "audio_format":    "mp3",
    "return_base64":   true
}
"""

import base64
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import runpod

sys.path.insert(0, "/app")

print("[Handler] Importing ACE-Step 1.5 modules…")
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.gpu_config import get_gpu_config
print("[Handler] Imports OK.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")
DIT_MODEL      = os.environ.get("DIT_MODEL",      "acestep-v15-turbo")
LM_MODEL       = os.environ.get("LM_MODEL",       "acestep-5Hz-lm-1.7B")
DEVICE         = os.environ.get("DEVICE",         "cuda")

# ---------------------------------------------------------------------------
# Global model handles
# ---------------------------------------------------------------------------
_dit_handler: AceStepHandler | None = None
_llm_handler: LLMHandler | None = None


def _init_models() -> tuple[AceStepHandler, LLMHandler]:
    global _dit_handler, _llm_handler
    if _dit_handler is not None and _llm_handler is not None:
        return _dit_handler, _llm_handler

    gpu_cfg     = get_gpu_config()
    cpu_offload = gpu_cfg.gpu_memory_gb < 16.0

    print(f"[Handler] GPU: {gpu_cfg.gpu_memory_gb:.1f} GB  |  cpu_offload={cpu_offload}")
    print(f"[Handler] CHECKPOINT_DIR : {CHECKPOINT_DIR}")
    print(f"[Handler] DiT            : {DIT_MODEL}")
    print(f"[Handler] LM             : {LM_MODEL}")

    dit = AceStepHandler()
    dit.initialize_service(
        project_root=CHECKPOINT_DIR,
        config_path=DIT_MODEL,
        device=DEVICE,
        offload_to_cpu=cpu_offload,
    )

    llm = LLMHandler()
    llm.initialize(
        checkpoint_dir=CHECKPOINT_DIR,
        lm_model_path=LM_MODEL,
        backend="pytorch",
        device=DEVICE,
    )

    _dit_handler = dit
    _llm_handler = llm
    print("[Handler] All models initialised successfully.")
    return _dit_handler, _llm_handler


try:
    _init_models()
except Exception as _exc:
    print(f"[Handler] WARNING: model pre-load failed – will retry on first request.\n  {_exc}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _save_tensor(tensor, sample_rate: int, path: str, fmt: str) -> None:
    """Save a torch audio tensor to disk."""
    import torchaudio
    # torchaudio uses 'mp3' as format string directly
    torchaudio.save(path, tensor.cpu(), sample_rate, format=fmt)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    inp: dict = job.get("input", {})

    caption: str = inp.get("caption", "")
    if not caption:
        return {"error": "Missing required parameter: 'caption'"}

    # GenerationParams fields
    lyrics:          str        = inp.get("lyrics",          "")
    duration:        float      = float(inp.get("duration",       -1))
    seed:            int        = int(inp.get("seed",             -1))
    bpm_raw                     = inp.get("bpm",               None)
    bpm:    int | None          = int(bpm_raw) if bpm_raw else None
    keyscale:        str        = inp.get("keyscale",            "")
    timesignature:   str        = str(inp.get("timesignature",   ""))
    vocal_language:  str        = inp.get("vocal_language",  "unknown")
    inference_steps: int        = int(inp.get("inference_steps",  8))
    guidance_scale:  float      = float(inp.get("guidance_scale", 7.0))
    shift:           float      = float(inp.get("shift",          3.0))
    infer_method:    str        = inp.get("infer_method",      "ode")
    thinking:        bool       = bool(inp.get("thinking",      True))

    # GenerationConfig fields
    batch_size:      int        = int(inp.get("batch_size",        1))
    audio_format:    str        = inp.get("audio_format",       "mp3")

    # Handler-only
    return_base64:   bool       = bool(inp.get("return_base64",  True))

    print(f"[Handler] caption='{caption[:60]}…'  duration={duration}s  batch={batch_size}")

    try:
        dit, llm = _init_models()
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Model init failed: {exc}"}

    # Build params and config
    params = GenerationParams(
        caption=caption,
        lyrics=lyrics,
        duration=duration,
        seed=seed,
        bpm=bpm,
        keyscale=keyscale,
        timesignature=timesignature,
        vocal_language=vocal_language,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        infer_method=infer_method,
        thinking=thinking,
    )

    use_random_seed = (seed == -1)
    config = GenerationConfig(
        batch_size=batch_size,
        seeds=None if use_random_seed else [seed] * batch_size,
        use_random_seed=use_random_seed,
        audio_format=audio_format,
    )

    # Use a temp dir so generate_music saves files to disk
    tmp_dir = tempfile.mkdtemp()
    try:
        result = generate_music(
            dit_handler=dit,
            llm_handler=llm,
            params=params,
            config=config,
            save_dir=tmp_dir,       # ← critical: tells the model to write files
        )
    except Exception as exc:
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"error": f"Generation failed: {exc}"}

    if not result.success:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"error": result.error or "Generation failed with unknown error"}

    # Collect results
    audio_files = []
    for i, audio_dict in enumerate(result.audios):
        audio_path: str = audio_dict.get("path") or audio_dict.get("audio_path", "")

        # Fallback: path empty → save tensor ourselves
        if not audio_path or not Path(audio_path).exists():
            tensor     = audio_dict.get("tensor")
            samplerate = audio_dict.get("sample_rate", 48000)
            if tensor is not None:
                ext = audio_format if audio_format in ("wav", "flac") else "mp3"
                audio_path = str(Path(tmp_dir) / f"output_{i}.{ext}")
                _save_tensor(tensor, samplerate, audio_path, ext)
                print(f"[Handler] Saved tensor manually → {audio_path}")
            else:
                print(f"[Handler] WARNING: no path and no tensor for item {i}, skipping.")
                continue

        entry: dict = {"filename": Path(audio_path).name}

        if return_base64:
            entry["data"]      = _b64(audio_path)
            entry["mime_type"] = _MIME.get(audio_format, "audio/mpeg")

        net_vol = os.environ.get("RUNPOD_NETWORK_VOLUME_PATH")
        if net_vol:
            dest = Path(net_vol) / "outputs" / Path(audio_path).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio_path, dest)
            entry["network_volume_path"] = str(dest)

        for k in ("bpm", "keyscale", "timesignature", "duration", "seed"):
            if k in audio_dict:
                entry[k] = audio_dict[k]

        audio_files.append(entry)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not audio_files:
        return {"error": "Generation produced no output files."}

    print(f"[Handler] Returning {len(audio_files)} file(s).")
    return {"output": {"status": "success", "audio_files": audio_files}}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})