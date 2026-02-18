import json
import base64

with open("response.json", "r") as f:
    data = json.load(f)

audio_files = data["output"]["output"]["audio_files"]

for audio in audio_files:
    filename = audio["filename"]
    audio_bytes = base64.b64decode(audio["data"])
    
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    
    print(f"Saved: {filename}")