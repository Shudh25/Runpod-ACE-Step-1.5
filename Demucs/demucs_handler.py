#!/usr/bin/env python3
"""
RunPod Serverless Handler — Demucs Stem Separation

Storage strategy (automatic):
  - If /runpod-volume is mounted → copy files directly to volume
  - Else → upload via S3 SDK using env var credentials

Output is always S3 keys (no /runpod-volume prefix).
Your backend constructs the full URL:
  https://s3api-eu-ro-1.runpod.io/e429rfhzfg/<s3_key>

REQUIRED ENV VARS (always needed):
  RUNPOD_S3_ENDPOINT   — e.g. https://s3api-eu-ro-1.runpod.io
  RUNPOD_S3_BUCKET     — e.g. e429rfhzfg

REQUIRED ONLY when volume is NOT mounted:
  RUNPOD_S3_ACCESS_KEY
  RUNPOD_S3_SECRET_KEY
  RUNPOD_S3_REGION     — e.g. eu-ro-1

INPUT:
{
    "input": {
        "audio_url": "https://..."   // or "audio_base64": "..."
    }
}

OUTPUT:
{
    "output": {
        "job_id": "abc12345",
        "stems": {
            "drums":  "demucs/abc12345/drums.wav",
            "bass":   "demucs/abc12345/bass.wav",
            "other":  "demucs/abc12345/other.wav",
            "vocals": "demucs/abc12345/vocals.wav"
        },
        "no_vocals": "demucs/abc12345/no_vocals.wav"
    }
}
"""

import base64
import os
import shutil
import subprocess
import tempfile
import traceback
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

import runpod

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
VOLUME_MOUNT      = "/runpod-volume"
VOLUME_OUTPUT_DIR = os.environ.get("VOLUME_OUTPUT_DIR", f"{VOLUME_MOUNT}/demucs")

S3_ENDPOINT   = os.environ.get("RUNPOD_S3_ENDPOINT",   "https://s3api-eu-ro-1.runpod.io")
S3_BUCKET     = os.environ.get("RUNPOD_S3_BUCKET",     "e429rfhzfg")
S3_ACCESS_KEY = os.environ.get("RUNPOD_S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("RUNPOD_S3_SECRET_KEY", "")
S3_REGION     = os.environ.get("RUNPOD_S3_REGION",     "eu-ro-1")


def _is_volume_mounted() -> bool:
    """Check if the RunPod network volume is actually mounted."""
    return os.path.ismount(VOLUME_MOUNT) or os.path.isdir(VOLUME_MOUNT)


def _get_s3_client():
    """Lazy-init S3 client — only created when volume is not mounted."""
    import boto3
    from botocore.client import Config
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(
            signature_version="s3v4",
            # Disable multipart uploads — RunPod S3 doesn't support them reliably
            multipart_threshold=10 * 1024 * 1024 * 1024,  # 10GB (effectively disabled)
            multipart_chunksize=10 * 1024 * 1024 * 1024,
        ),
        region_name=S3_REGION,
    )


def _save_file(local_path: str, s3_key: str, use_volume: bool, s3=None):
    """Copy to volume OR upload via S3 SDK. Returns the s3_key either way."""
    if use_volume:
        dst = Path(VOLUME_OUTPUT_DIR) / Path(s3_key).relative_to("demucs").parent / Path(s3_key).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)
    else:
        s3.upload_file(local_path, S3_BUCKET, s3_key)
    return s3_key


def _download_audio(url: str, dest: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Demucs-Worker/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        with open(dest, "wb") as f:
            f.write(r.read())


def handler(job):
    inp = job.get("input", {})
    audio_url = inp.get("audio_url", "")
    audio_b64 = inp.get("audio_base64", "")

    if not audio_url and not audio_b64:
        return {"error": "Either audio_url or audio_base64 is required"}

    use_volume = _is_volume_mounted()
    print(f"[Demucs] Storage mode: {'volume (direct copy)' if use_volume else 'S3 upload'}")

    # Validate S3 creds if we need them
    if not use_volume and (not S3_ACCESS_KEY or not S3_SECRET_KEY):
        return {"error": "Volume not mounted and S3 credentials missing. Set RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY."}

    s3 = None if use_volume else _get_s3_client()

    job_id = str(uuid.uuid4())[:8]
    work_dir = tempfile.mkdtemp(prefix="demucs_")

    try:
        # ----------------------------------------------------------
        # 1. Fetch input audio
        # ----------------------------------------------------------
        if audio_url:
            print("[Demucs] Downloading audio...")
            ext = os.path.splitext(urlparse(audio_url).path)[1] or ".mp3"
            audio_path = os.path.join(work_dir, f"input_audio{ext}")
            _download_audio(audio_url, audio_path)
        else:
            print("[Demucs] Decoding base64 audio...")
            audio_path = os.path.join(work_dir, "input_audio.wav")
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_b64))

        # ----------------------------------------------------------
        # 2. Run demucs CLI — all 4 stems
        # ----------------------------------------------------------
        print("[Demucs] Running separation...")
        subprocess.run([
            "demucs", "-n", "htdemucs",
            "-o", work_dir,
            audio_path
        ], check=True)

        stem_dir = Path(work_dir) / "htdemucs" / Path(audio_path).stem

        # ----------------------------------------------------------
        # 3. Mix drums + bass + other → no_vocals via ffmpeg
        # ----------------------------------------------------------
        instrumental = ["drums", "bass", "other"]
        no_vocals_tmp = os.path.join(work_dir, "no_vocals.wav")
        ffmpeg_cmd = ["ffmpeg", "-y"]
        for name in instrumental:
            ffmpeg_cmd += ["-i", str(stem_dir / f"{name}.wav")]
        ffmpeg_cmd += [
            "-filter_complex", f"amix=inputs={len(instrumental)}:normalize=0",
            no_vocals_tmp
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        # ----------------------------------------------------------
        # 4. Save stems → return S3 keys
        # ----------------------------------------------------------
        print("[Demucs] Saving stems...")
        stems = {}
        for name in ["drums", "bass", "other", "vocals"]:
            local = str(stem_dir / f"{name}.wav")
            key   = f"demucs/{job_id}/{name}.wav"
            stems[name] = _save_file(local, key, use_volume, s3)
            print(f"[Demucs] ✓ {name} → {key}")

        no_vocals_key = _save_file(no_vocals_tmp, f"demucs/{job_id}/no_vocals.wav", use_volume, s3)
        print(f"[Demucs] ✓ no_vocals → {no_vocals_key}")

        print(f"[Demucs] Done. job_id={job_id}")
        return {
            "output": {
                "job_id": job_id,
                "stems": stems,
                "no_vocals": no_vocals_key
            }
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Separation failed: {e}"}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})