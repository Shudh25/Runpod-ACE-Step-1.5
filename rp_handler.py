import runpod
import os
import subprocess
import time
import base64
import requests

CONFIG_PATH = os.environ.get("ACESTEP_CONFIG_PATH", "/app/checkpoints/acestep-v15-base")
LM_MODEL_PATH = os.environ.get("ACESTEP_LM_MODEL_PATH", "/app/checkpoints/acestep-5Hz-lm-1.7B")
API_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = os.environ.get("ACESTEP_OUTPUT_DIR", "/app/outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Start ACE-Step API server as subprocess ──
print("Starting ACE-Step API server...")
server_proc = subprocess.Popen(
    ["acestep-api", "--host", "127.0.0.1", "--port", "8000"],
    env={**os.environ, "ACESTEP_CONFIG_PATH": CONFIG_PATH, "ACESTEP_LM_MODEL_PATH": LM_MODEL_PATH},
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

# ── Wait for server to be ready ──
print("Waiting for API server to become ready...")
for i in range(180):
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            print(f"API server ready after {i}s")
            break
    except Exception:
        pass
    time.sleep(1)
else:
    raise RuntimeError("ACE-Step API server failed to start within 180 seconds")


def poll_result(task_id: str, timeout: int = 600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.post(
            f"{API_URL}/query_result",
            json={"task_ids": [task_id]},
            timeout=10
        )
        r.raise_for_status()
        results = r.json()

        # Response is typically a dict keyed by task_id
        task_data = results.get(task_id) or (results[0] if isinstance(results, list) else None)
        if not task_data:
            time.sleep(2)
            continue

        status = task_data.get("status", "")
        if status in ("completed", "success", "done"):
            return task_data
        elif status in ("failed", "error"):
            raise RuntimeError(f"Task failed: {task_data.get('error', task_data)}")

        time.sleep(2)
    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")


def handler(job):
    job_input = job["input"]

    caption = job_input.get("prompt") or job_input.get("caption", "")
    lyrics = job_input.get("lyrics", "")
    duration = float(job_input.get("duration", 30.0))

    if not caption and not lyrics:
        return {"error": "At least one of 'prompt'/'caption' or 'lyrics' must be provided."}

    payload = {
        "caption": caption,
        "lyrics": lyrics,
        "audio_duration": duration,
        "infer_step": int(job_input.get("steps", 60)),
        "guidance_scale": float(job_input.get("guidance_scale", 4.5)),
        "scheduler_type": job_input.get("scheduler_type", "ddim"),
        "cfg_type": job_input.get("cfg_type", "apg"),
        "omega_scale": float(job_input.get("omega_scale", 10.0)),
        "seed": int(job_input.get("seed", -1)),
        "use_erg_tag": job_input.get("use_erg_tag", True),
        "use_erg_lyric": job_input.get("use_erg_lyric", True),
        "use_erg_diffusion": job_input.get("use_erg_diffusion", True),
    }

    try:
        # ── Submit task ──
        r = requests.post(f"{API_URL}/release_task", json=payload, timeout=30)
        r.raise_for_status()
        resp = r.json()
        task_id = resp.get("task_id") or resp.get("id")
        if not task_id:
            return {"error": f"No task_id in response: {resp}"}
        print(f"Submitted task: {task_id}")

        # ── Poll until done ──
        result = poll_result(task_id, timeout=int(job_input.get("timeout", 600)))

        # ── Get audio ──
        # Try fetching via /v1/audio endpoint first
        audio_url = f"{API_URL}/v1/audio?task_id={task_id}"
        audio_resp = requests.get(audio_url, timeout=30)

        if audio_resp.status_code == 200:
            audio_b64 = base64.b64encode(audio_resp.content).decode("utf-8")
            content_type = audio_resp.headers.get("content-type", "audio/wav")
            fmt = "mp3" if "mp3" in content_type else "wav"
        else:
            # Fallback: look for file in output dir
            candidates = [
                f for f in os.listdir(OUTPUT_DIR)
                if task_id in f and f.endswith((".wav", ".mp3"))
            ]
            if not candidates:
                # Try path from result dict
                output_path = result.get("output_path") or result.get("audio_path") or result.get("file_path")
                if output_path and os.path.exists(output_path):
                    candidates = [output_path]
                else:
                    return {"error": "Audio file not found", "result": result}

            file_path = candidates[0] if "/" in candidates[0] else os.path.join(OUTPUT_DIR, candidates[0])
            with open(file_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            fmt = file_path.rsplit(".", 1)[-1]
            try:
                os.remove(file_path)
            except Exception:
                pass

        return {
            "audio_base64": audio_b64,
            "format": fmt,
            "task_id": task_id,
            "duration": duration,
            "caption": caption,
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})