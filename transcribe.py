import sys
import os
import json
from faster_whisper import WhisperModel

# 1. نقرأ اسم ملف الصوت من الترمنال، أو نستخدم jfk.flac كافتراضي لو ما انكتب شي
audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/jfk.flac"

model = WhisperModel("small", device="cpu", compute_type="int8")

segments, info = model.transcribe(audio_path)

print(f"اللغة المكتشفة: {info.language} (نسبة الثقة: {info.language_probability:.2f})")

transcript_segments = []
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    transcript_segments.append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip()
    })

os.makedirs("output", exist_ok=True)

# 2. نشتق اسم ملف الحفظ تلقائياً من اسم ملف الصوت
base_name = os.path.splitext(os.path.basename(audio_path))[0]
output_path = f"output/{base_name}_transcript.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        {"language": info.language, "segments": transcript_segments},
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"\nتم حفظ الترانسكربت بملف: {output_path}")