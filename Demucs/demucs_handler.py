import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
import subprocess
import traceback

import runpod

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/runpod-volume/demucs_outputs")


def _download_audio(url: str, dest: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Demucs-Worker/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        with open(dest, "wb") as f:
            f.write(r.read())


def handler(job):
    try:
        inp = job.get("input", {})
        audio_url = inp.get("audio_url")

        if not audio_url:
            return {"error": "Missing 'audio_url'"}

        work_dir = tempfile.mkdtemp(prefix="demucs_")
        audio_path = os.path.join(work_dir, "input_audio.wav")

        print("[Demucs] Downloading audio...")
        _download_audio(audio_url, audio_path)

        print("[Demucs] Running separation...")
        subprocess.run([
            "demucs",
            "--two-stems", "vocals",
            "-o", work_dir,
            audio_path
        ], check=True)

        separated_dir = Path(work_dir) / "htdemucs" / "input_audio"
        vocals_file = separated_dir / "vocals.wav"
        accompaniment_file = separated_dir / "no_vocals.wav"

        if not vocals_file.exists():
            raise RuntimeError("Demucs failed to generate vocals.")

        output_folder = Path(OUTPUT_DIR)
        output_folder.mkdir(parents=True, exist_ok=True)

        final_vocals = output_folder / f"vocals_{os.path.basename(work_dir)}.wav"
        final_instr = output_folder / f"instrumental_{os.path.basename(work_dir)}.wav"

        shutil.copy2(vocals_file, final_vocals)
        shutil.copy2(accompaniment_file, final_instr)

        shutil.rmtree(work_dir, ignore_errors=True)

        return {
            "output": {
                "status": "success",
                "vocals_path": str(final_vocals),
                "instrumental_path": str(final_instr)
            }
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})