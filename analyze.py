import sys
import os
import json
import requests

# 1. نقرأ اسم ملف الصوت الأصلي (نفس النمط اللي أسسناه بالسكربتات الثانية)
audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/jfk.flac"
base_name = os.path.splitext(os.path.basename(audio_path))[0]

input_path = f"output/{base_name}_transcript_with_speakers.json"
output_path = f"output/{base_name}_analysis.json"

with open(input_path, "r", encoding="utf-8") as f:
    segments = json.load(f)

# 2. نبني النص مع تسمية المتحدث قبل كل سطر (مو نص عادي متواصل زي قبل)
full_text = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)

print("النص مع تسميات المتحدثين:")
print(full_text)
print()

url = "http://localhost:11434/api/chat"

system_prompt = """You are an assistant that analyzes meeting or speech transcripts. The transcript includes speaker labels (e.g. SPEAKER_00, SPEAKER_01) before each line.
Always respond with ONLY valid JSON, with no extra text, no explanations, and no markdown code fences.
Use exactly this structure:

{
  "summary": "...",
  "decisions": ["..."],
  "action_items": [{"person": "...", "task": "...", "deadline": "..."}],
  "deadlines": ["..."],
  "open_questions": ["..."]
}

For "person" in action_items, use the speaker label (e.g. "SPEAKER_00") if no real name is mentioned in the transcript itself. The "summary" field should be 2-3 concise sentences. If a list category has nothing to report, return an empty list for it. Do not invent information that isn't in the transcript."""

payload = {
    "model": "qwen2.5:7b",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this transcript:\n\n{full_text}"}
    ],
    "stream": False
}

# 3. نرسل الطلب
response = requests.post(url, json=payload)
response.raise_for_status()

raw_content = response.json()["message"]["content"].strip()

if raw_content.startswith("```"):
    raw_content = raw_content.strip("`")
    if raw_content.startswith("json"):
        raw_content = raw_content[4:].strip()

try:
    analysis = json.loads(raw_content)
except json.JSONDecodeError as e:
    print("تحذير: الموديل ما رجع JSON صحيح!")
    print("الرد الخام:")
    print(raw_content)
    raise e

print("نتيجة التحليل الكاملة:")
print(json.dumps(analysis, ensure_ascii=False, indent=2))

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)

print(f"\nتم حفظ التحليل الكامل بملف: {output_path}")