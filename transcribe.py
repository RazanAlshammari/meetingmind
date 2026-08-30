from faster_whisper import WhisperModel
import json
import os

model = WhisperModel("small", device="cpu", compute_type="int8")

segments, info = model.transcribe("data/jfk.flac")

print(f"اللغة المكتشفة: {info.language} (نسبة الثقة: {info.language_probability:.2f})")

# نجمع كل مقطع بقائمة بدل ما نطبعه بس، عشان نقدر نحفظه بملف
transcript_segments = []
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    transcript_segments.append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip()
    })

# ننشئ مجلد output لو مو موجود (exist_ok=True يمنع الخطأ لو موجود أصلاً)
os.makedirs("output", exist_ok=True)

# نحفظ الترانسكربت كامل بملف JSON منظم
with open("output/jfk_transcript.json", "w", encoding="utf-8") as f:
    json.dump(
        {"language": info.language, "segments": transcript_segments},
        f,
        ensure_ascii=False,  # مهم لاحقًا للعربي، يمنع تحويل الأحرف لرموز غريبة
        indent=2
    )

print("\nتم حفظ الترانسكربت بملف: output/jfk_transcript.json")