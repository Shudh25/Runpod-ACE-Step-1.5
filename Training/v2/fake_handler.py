import os
import json
import time
from pathlib import Path

import boto3
from botocore.config import Config

# ENV
LORA_OUTPUT_DIR = os.environ.get("LORA_OUTPUT_DIR", "/runpod-volume/loras")


def upload_to_s3(filepath: str, s3_config: dict, s3_key: str):
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_config["endpoint"],
        aws_access_key_id=s3_config["access_key"],
        aws_secret_access_key=s3_config["secret_key"],
        region_name=s3_config.get("region", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )

    s3.upload_file(filepath, s3_config["bucket"], s3_key)
    print(f"✓ Uploaded: {s3_key}")


def fake_handler():
    print("=== FAKE HANDLER START ===")

    # -------------------------------------------------------------------------
    # 1. Hardcoded payload
    # -------------------------------------------------------------------------
    payload = {
        "lora_name": "test_fake_lora",
        "s3_config": {
            "endpoint": "https://s3api-eu-ro-1.runpod.io",
            "bucket": "tm28qb0qoo",
            "region": "eu-ro-1",
            "access_key": "user_39Q097ZjwHHovTDbd2jTotAW0yR",
            "secret_key": "rps_6CTPILC0460KPOBB7ASM7JAZ175RJY9177U2A52Nz4c7w4"
        }
    }

    lora_name = payload["lora_name"]
    s3_config = payload["s3_config"]

    # -------------------------------------------------------------------------
    # 2. Create fake LoRA output
    # -------------------------------------------------------------------------
    fake_output_dir = f"/tmp/fake_lora_{int(time.time())}"
    fake_adapter_dir = Path(fake_output_dir) / "final" / "adapter"
    fake_adapter_dir.mkdir(parents=True, exist_ok=True)

    fake_weights = fake_adapter_dir / "adapter_model.safetensors"
    fake_config = fake_adapter_dir / "adapter_config.json"

    fake_weights.write_bytes(b"FAKE SAFETENSORS CONTENT")
    fake_config.write_text(json.dumps({"fake": True}))

    print(f"Created fake LoRA at: {fake_adapter_dir}")

    # -------------------------------------------------------------------------
    # 3. Copy to volume (same as before)
    # -------------------------------------------------------------------------
    dest_dir = Path(LORA_OUTPUT_DIR) / lora_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for src in fake_adapter_dir.glob("*"):
        dst = dest_dir / src.name
        dst.write_bytes(src.read_bytes())
        files.append(dst)

    print(f"Saved to volume: {dest_dir}")

    # -------------------------------------------------------------------------
    # 4. Upload using boto3
    # -------------------------------------------------------------------------
    print("Uploading to S3...")
    for f in files:
        s3_key = f"{lora_name}/{f.name}"
        try:
            upload_to_s3(str(f), s3_config, s3_key)
        except Exception as e:
            print(f"✗ Upload failed for {s3_key}: {e}")

    print("=== DONE ===")


if __name__ == "__main__":
    fake_handler()