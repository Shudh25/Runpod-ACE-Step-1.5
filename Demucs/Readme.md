# 🎵 Demucs Serverless – RunPod Endpoint

Lightweight Demucs deployment for vocal separation using RunPod Serverless.

This endpoint:

* Downloads audio from URL
* Separates vocals + instrumental
* Saves output to RunPod Network Volume
* Returns file paths

---

# 📁 Folder Structure

```id="kq3d2m"
demucs-serverless/
│
├── Dockerfile
├── demucs_handler.py
├── sample_request.json
└── README.md
```

---

# 🐳 Build & Push Docker Image

### 1️⃣ Build Image

```
docker build -t your-dockerhub-username/demucs-serverless:latest .
```

### 2️⃣ Push Image

```
docker push your-dockerhub-username/demucs-serverless:latest
```

### 3️⃣ Use Image in RunPod

In RunPod → Serverless → Create Endpoint
Use:

```
your-dockerhub-username/demucs-serverless:latest
```

Mount Network Volume at:

```
/runpod-volume
```

---

# 📤 Sample Request JSON

```
{
  "input": {
    "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
  }
}
```

---

# 📥 Sample Response

```
{
  "delayTime": 7900,
  "executionTime": 17169,
  "id": "1be6c6c3-d0cd-4b54-ad8f-0bb12390b594-e1",
  "output": {
    "output": {
      "instrumental_path": "/runpod-volume/demucs_outputs/instrumental_demucs_iwdoe04k.wav",
      "status": "success",
      "vocals_path": "/runpod-volume/demucs_outputs/vocals_demucs_iwdoe04k.wav"
    }
  },
  "status": "COMPLETED",
  "workerId": "55rua6f45vj5bm"
}
```

---

# ⚙️ Recommended RunPod Settings

* GPU: T4 or A10G
* Timeout: 15–30 minutes
* Concurrency: 1–2
* Network Volume required