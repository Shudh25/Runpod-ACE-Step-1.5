# ACE-Step 1.5 API Usage Guide

This guide covers how to use the ACE-Step API for music generation, including practical examples and troubleshooting tips.

## Base URL

When running locally or via Docker:
```
http://localhost:8000
```

For RunPod deployments:
```
https://<POD_ID>-8000.proxy.runpod.net
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/release_task` | POST | Create music generation task |
| `/query_result` | POST | Query task results |
| `/format_input` | POST | Format and enhance lyrics/caption via LLM |
| `/v1/audio` | GET | Download generated audio file |

## Python CLI Script (Recommended)

The easiest way to generate music is using the included Python CLI script. It handles task submission, polling, and file download automatically.

### Basic Usage

```bash
python generate_music.py \
  --api-url https://your-pod-8000.proxy.runpod.net \
  --caption "Upbeat indie pop with jangly guitars and energetic vocals" \
  --lyrics "[Verse 1]\nWalking down the street\nMusic in my feet\n\n[Chorus]\nWe are alive tonight" \
  --output my_song.mp3
```

### Using a Lyrics File

```bash
python generate_music.py \
  --api-url http://localhost:8000 \
  --caption "Dreamy folk ballad with acoustic guitar" \
  --lyrics-file my_lyrics.txt \
  --duration 120 \
  --output ballad.mp3
```

### Generate Multiple Variations

```bash
python generate_music.py \
  --api-url https://your-api.runpod.net \
  --caption "Electronic dance track with driving beat" \
  --batch-size 2 \
  --output dance.mp3  # Creates dance_1.mp3, dance_2.mp3
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--api-url` | Base URL of the ACE-Step API (required) |
| `--caption`, `-c` | Music style description (required) |
| `--lyrics`, `-l` | Song lyrics with structure tags (use `\n` for newlines) |
| `--lyrics-file`, `-f` | Path to a text file containing lyrics |
| `--duration`, `-d` | Duration in seconds (default: 90) |
| `--batch-size`, `-b` | Number of variations to generate (default: 1) |
| `--output`, `-o` | Output filename (default: output.mp3) |
| `--poll-interval` | Seconds between status checks (default: 5) |
| `--timeout` | Maximum seconds to wait (default: 600) |
| `--quiet`, `-q` | Suppress progress output |
| `--show-lyrics-help` | Show detailed lyrics writing guide |
| `--show-caption-help` | Show detailed style/caption guide |

### Writing Effective Captions

The caption describes the musical style. Include:
- **Genre**: pop, rock, folk, electronic, jazz, hip-hop, etc.
- **Instruments**: guitar, piano, synths, drums, strings, etc.
- **Mood**: upbeat, melancholic, energetic, peaceful, dark, hopeful
- **Vocal style**: soft female vocals, raspy male voice, choir, etc.
- **Production**: lo-fi, polished, atmospheric, reverb-heavy

**Example captions:**
```
"Upbeat indie pop with jangly guitars, bright synths, and energetic female vocals"
"Dark atmospheric electronic with deep bass, haunting pads, and whispered vocals"
"Warm acoustic folk ballad with fingerpicked guitar, soft harmonies, and gentle strings"
"High-energy rock anthem with distorted guitars, pounding drums, and powerful male vocals"
```

### Structuring Lyrics

Use structure tags to organize your song:

```
[Intro]        - Instrumental opening
[Verse 1]      - First verse (use [Verse 2], [Verse 3] for more)
[Pre-Chorus]   - Build-up before chorus
[Chorus]       - Main hook, usually repeated
[Bridge]       - Contrasting section
[Outro]        - Ending section
[Drop]         - For electronic music
[Instrumental] - Non-vocal sections
```

**Example lyrics file:**
```
[Verse 1]
Walking down the empty street
Shadows dancing at my feet

[Chorus]
We are the dreamers of the night
Chasing stars until the light

[Verse 2]
Memories like falling rain
Washing away all the pain

[Chorus]
We are the dreamers of the night
Chasing stars until the light

[Bridge]
And when the morning comes around
We'll still be here, we won't back down

[Outro]
Dreamers of the night...
```

**Tips:**
- Keep verses 2-4 lines each for best results
- Repeat the chorus for emphasis
- Leave blank lines between sections

---

## Quick Start Example (curl)

For direct API access without the CLI script, use curl:

### 1. Check API Health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "data": {"status": "ok", "service": "ACE-Step API", "version": "1.0"},
  "code": 200,
  "error": null
}
```

### 2. Generate Music

Submit a generation task with a caption (style description) and lyrics:

```bash
curl -X POST http://localhost:8000/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "caption": "Warm acoustic folk song with gentle fingerpicked guitar, soft piano, and a cozy nostalgic atmosphere",
    "lyrics": "[Verse 1]\nLines of code like poetry\nPatterns dancing endlessly\n\n[Chorus]\nIn the logic we find grace\nEvery problem has its place",
    "duration": 90
  }'
```

Response:
```json
{
  "data": {
    "task_id": "ea782b05-87d3-428f-a269-27a7bea32c94",
    "status": "queued",
    "queue_position": 1
  },
  "code": 200
}
```

### 3. Query Task Result

Poll for the result using the task ID. **Important:** The parameter is `task_id_list` (not `task_ids`):

```bash
curl -X POST http://localhost:8000/query_result \
  -H "Content-Type: application/json" \
  -d '{"task_id_list": ["ea782b05-87d3-428f-a269-27a7bea32c94"]}'
```

Response (when complete):
```json
{
  "data": [
    {
      "task_id": "ea782b05-87d3-428f-a269-27a7bea32c94",
      "status": 1,
      "result": "[{\"file\": \"/v1/audio?path=%2Fapp%2Foutputs%2Fapi_audio%2Fc3ababa4-7d39-827c-21ca-a7ef8add99d8.mp3\", ...}]"
    }
  ],
  "code": 200
}
```

**Key points:**
- Use `task_id_list` parameter (NOT `task_ids` - wrong parameter name returns empty results)
- The `result` field is a JSON string containing an array of generated audio files
- Each item has a `file` field with the download URL (URL-encoded path)
- The **audio filename is different from the task ID** - they are separate UUIDs
- With `batch_size: 2` (default), you get 2 audio files in the result

**Status codes in result:**
- `0` = in progress
- `1` = success
- `2` = failed

### 4. Download Generated Audio

Extract the `file` URL from the query result and download. The path is URL-encoded, so you can use it directly:

```bash
# Using the file URL from query_result (already URL-encoded)
curl -o song.mp3 "http://localhost:8000/v1/audio?path=%2Fapp%2Foutputs%2Fapi_audio%2Fc3ababa4-7d39-827c-21ca-a7ef8add99d8.mp3"

# Or with the decoded path (must be full absolute path)
curl -o song.mp3 "http://localhost:8000/v1/audio?path=/app/outputs/api_audio/c3ababa4-7d39-827c-21ca-a7ef8add99d8.mp3"
```

**⚠️ IMPORTANT - Common Mistakes:**

1. **Wrong parameter name:** Using `task_ids` instead of `task_id_list` returns empty results

2. **The filename is NOT the task ID.** The task ID (e.g., `945bb822-b563-44bb-8a41-9a952cc0e9e6`) is different from the audio filename (e.g., `c3ababa4-7d39-827c-21ca-a7ef8add99d8.mp3`)

3. **You MUST use the full absolute path.** The path must start with `/app/outputs/api_audio/`

Example:
```bash
# WRONG - using task ID as filename (WILL NOT WORK)
curl -o song.mp3 "http://localhost:8000/v1/audio?path=/app/outputs/api_audio/945bb822-b563-44bb-8a41-9a952cc0e9e6.mp3"

# CORRECT - using actual filename from query_result
curl -o song.mp3 "http://localhost:8000/v1/audio?path=/app/outputs/api_audio/c3ababa4-7d39-827c-21ca-a7ef8add99d8.mp3"
```

## Request Parameters

### `/release_task` Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `caption` | string | Yes | Music style description (genre, instruments, mood) |
| `lyrics` | string | No | Song lyrics with structure tags like `[Verse]`, `[Chorus]` |
| `duration` | int | No | Duration in seconds (default varies by GPU tier) |
| `batch_size` | int | No | Number of variations to generate (default: 2) |

### Lyrics Format

Use structure tags to organize your lyrics:
```
[Verse 1]
First verse lyrics here
More lines...

[Chorus]
Catchy chorus lyrics
Repeat section...

[Verse 2]
Second verse lyrics

[Bridge]
Bridge section

[Outro]
Ending lyrics
```

## LLM-Powered Features

### Format/Enhance Lyrics

Use the LLM to improve or format your lyrics:

```bash
curl -X POST http://localhost:8000/format_input \
  -H "Content-Type: application/json" \
  -d '{
    "caption": "rock song",
    "lyrics": "some rough lyrics here"
  }'
```

## Troubleshooting

### Issue: Empty query results

**Symptom:** `/query_result` returns `{"data": []}` even after waiting.

**Cause:** You're using the wrong parameter name. The correct parameter is `task_id_list`, NOT `task_ids`.

**Solution:** 
```bash
# WRONG - returns empty data
curl -X POST http://localhost:8000/query_result \
  -d '{"task_ids": ["..."]}'  # Wrong parameter name!

# CORRECT - returns results
curl -X POST http://localhost:8000/query_result \
  -d '{"task_id_list": ["..."]}'  # Correct parameter name
```

If still empty with correct parameter, the task may still be processing (typically 30-120 seconds).

### Issue: Audio file not found

**Symptom:** `/v1/audio` returns `{"detail": "Audio file not found: ..."}`

**Cause:** Either:
1. The `path` parameter format is incorrect, OR
2. You're using the **task ID** instead of the **actual audio filename** (these are different!)

**Solution:**

1. **Get the correct filename from `/query_result`** using `task_id_list` parameter - the `file` field contains the download URL
2. **Use the full absolute path** including `/app/outputs/api_audio/`

```bash
# WRONG - using task ID as filename
curl "http://localhost:8000/v1/audio?path=/app/outputs/api_audio/945bb822-b563-44bb-8a41-9a952cc0e9e6.mp3"
# This will fail because the task ID is NOT the audio filename!

# WRONG - relative path
curl "http://localhost:8000/v1/audio?path=e0a7dddd.mp3"

# WRONG - partial path  
curl "http://localhost:8000/v1/audio?path=api_audio/e0a7dddd.mp3"

# CORRECT - full absolute path with actual filename from query_result
curl "http://localhost:8000/v1/audio?path=/app/outputs/api_audio/e0a7dddd-9a3a-b5e0-c98b-b07bd36e2e2f.mp3"
```

### Issue: Model checkpoint not found

**Symptom:** Server fails to start with error about missing model files in `/opt/venv/lib/python3.11/site-packages/checkpoints/`

**Cause:** ACE-Step looks for checkpoints relative to its installation directory. When installed in a virtualenv, it looks in the wrong location.

**Solution:** This was fixed by installing ACE-Step directly into `/app` instead of using a virtualenv. The Dockerfile now:
1. Clones ACE-Step into `/app`
2. Installs with `uv pip install --system`
3. Copies models to `/app/checkpoints`

This ensures `./checkpoints` resolves to `/app/checkpoints` where the models are baked in.

### Issue: VLC cannot play downloaded MP3

**Symptom:** VLC shows "cannot open file" errors even though the file exists.

**Cause:** This can be a permissions issue or VLC path handling quirk.

**Solution:** 
1. Verify the file is valid: `ffprobe ./song.mp3`
2. Copy to home directory: `cp ./song.mp3 ~/song.mp3`
3. Use alternative player: `mpv ./song.mp3` or `ffplay ./song.mp3`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACESTEP_CONFIG_PATH` | `/app/checkpoints/acestep-v15-base` | Path to DiT model |
| `ACESTEP_LM_MODEL_PATH` | `/app/checkpoints/acestep-5Hz-lm-1.7B` | Path to LM model |
| `ACESTEP_OUTPUT_DIR` | `/app/outputs` | Output directory for generated audio |
| `ACESTEP_DEVICE` | `cuda` | Device (cuda, cpu, mps) |
| `ACESTEP_LM_BACKEND` | `pt` | LLM backend (vllm, pt) |
| `ACESTEP_API_HOST` | `0.0.0.0` | Server host |
| `ACESTEP_API_PORT` | `8000` | Server port |

## Complete Workflow Example

```bash
#!/bin/bash
API_URL="http://localhost:8000"

# 1. Submit generation task
RESPONSE=$(curl -s -X POST "$API_URL/release_task" \
  -H "Content-Type: application/json" \
  -d '{
    "caption": "Upbeat electronic dance music with synthesizers and driving beat",
    "lyrics": "[Drop]\nFeel the rhythm take control\nLet the music free your soul",
    "duration": 60
  }')

TASK_ID=$(echo $RESPONSE | jq -r '.data.task_id')
echo "Task submitted: $TASK_ID"

# 2. Poll for completion (30-120 seconds depending on duration and GPU)
# IMPORTANT: Use task_id_list, NOT task_ids!
echo "Waiting for generation..."
while true; do
  RESULT=$(curl -s -X POST "$API_URL/query_result" \
    -H "Content-Type: application/json" \
    -d "{\"task_id_list\": [\"$TASK_ID\"]}")
  
  STATUS=$(echo $RESULT | jq -r '.data[0].status // 0')
  if [ "$STATUS" = "1" ]; then
    echo "Generation complete!"
    break
  elif [ "$STATUS" = "2" ]; then
    echo "Generation failed!"
    exit 1
  fi
  echo "Still processing..."
  sleep 10
done

# 3. Extract audio file URLs from result
# The result field is a JSON string containing array of files
# Each file has a "file" field with the download URL
FILES=$(echo $RESULT | jq -r '.data[0].result' | jq -r '.[].file')
echo "Audio files:"
echo "$FILES"

# 4. Download audio files
# Note: The file URLs are already properly formatted for the /v1/audio endpoint
COUNTER=1
for FILE_URL in $FILES; do
  curl -s -o "output_$COUNTER.mp3" "$API_URL$FILE_URL"
  echo "Downloaded output_$COUNTER.mp3"
  COUNTER=$((COUNTER + 1))
done
```

## GPU Requirements

| GPU VRAM | Max Duration | Recommended LM Model |
|----------|--------------|---------------------|
| 8GB | 120s | acestep-5Hz-lm-0.6B |
| 16GB | 300s | acestep-5Hz-lm-1.7B |
| 24GB+ | 600s | acestep-5Hz-lm-4B |

## Links

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step-1.5)
- [Official API Documentation](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md)
- [Docker Image Source](https://github.com/ValyrianTech/ace-step-1.5)
