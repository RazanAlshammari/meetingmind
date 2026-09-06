import sys
import os
import json

audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/jfk.flac"
base_name = os.path.splitext(os.path.basename(audio_path))[0]

input_path = f"output/{base_name}_transcript_with_speakers.json"
output_path = f"output/{base_name}_chunks.json"

with open(input_path, "r", encoding="utf-8") as f:
    segments = json.load(f)

chunks = []
current_chunk = None

for seg in segments:
    if current_chunk and current_chunk["speaker"] == seg["speaker"]:
        current_chunk["end"] = seg["end"]
        current_chunk["text"] += " " + seg["text"]
    else:
        if current_chunk:
            chunks.append(current_chunk)
        current_chunk = {
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg["speaker"],
            "text": seg["text"]
        }

if current_chunk:
    chunks.append(current_chunk)

for c in chunks:
    print(f"[{c['start']:.2f}s -> {c['end']:.2f}s] {c['speaker']}: {c['text']}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"\nتم حفظ {len(chunks)} قطعة بملف: {output_path}")