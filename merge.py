import json

# 1. نقرأ الترانسكربت (من Whisper) ونتيجة الديارايزيشن (من pyannote)
with open("output/jfk_transcript.json", "r", encoding="utf-8") as f:
    transcript_data = json.load(f)

with open("output/jfk_diarization.json", "r", encoding="utf-8") as f:
    diarization_segments = json.load(f)

# 2. دالة تحسب مقدار التداخل الزمني بين مقطعين
def overlap(start1, end1, start2, end2):
    return max(0, min(end1, end2) - max(start1, start2))

# 3. لكل مقطع نص (من Whisper)، نلاقي المتحدث صاحب أكبر تداخل زمني معه
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

# 4. نطبع ونحفظ النتيجة النهائية المدمجة
for seg in merged_segments:
    print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}")

with open("output/jfk_transcript_with_speakers.json", "w", encoding="utf-8") as f:
    json.dump(merged_segments, f, ensure_ascii=False, indent=2)

print("\nتم حفظ الترانسكربت المدمج بملف: output/jfk_transcript_with_speakers.json")