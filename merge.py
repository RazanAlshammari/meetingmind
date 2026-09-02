import sys
import os
import json

# 1. نقرأ اسم ملف الصوت الأصلي (نفس الاسم اللي استخدمناه بـ transcribe.py و diarize.py)
audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/jfk.flac"
base_name = os.path.splitext(os.path.basename(audio_path))[0]

transcript_path = f"output/{base_name}_transcript.json"
diarization_path = f"output/{base_name}_diarization.json"
output_path = f"output/{base_name}_transcript_with_speakers.json"

# 2. نقرأ الترانسكربت ونتيجة الديارايزيشن
with open(transcript_path, "r", encoding="utf-8") as f:
    transcript_data = json.load(f)

with open(diarization_path, "r", encoding="utf-8") as f:
    diarization_segments = json.load(f)

# 3. دالة تحسب مقدار التداخل الزمني بين مقطعين
def overlap(start1, end1, start2, end2):
    return max(0, min(end1, end2) - max(start1, start2))

# 4. لكل مقطع نص، نلاقي المتحدث صاحب أكبر تداخل زمني
merged_segments = []
for segment in transcript_data["segments"]:
    best_speaker = None
    best_overlap = 0.0

    for dia in diarization_segments:
        ov = overlap(segment["start"], segment["end"], dia["start"], dia["end"])
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = dia["speaker"]

    merged_segments.append({
        "start": segment["start"],
        "end": segment["end"],
        "speaker": best_speaker if best_speaker else "UNKNOWN",
        "text": segment["text"]
    })

# 5. نطبع ونحفظ النتيجة النهائية المدمجة
for seg in merged_segments:
    print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_segments, f, ensure_ascii=False, indent=2)

print(f"\nتم حفظ الترانسكربت المدمج بملف: {output_path}")