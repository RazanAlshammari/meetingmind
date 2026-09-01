import json
import requests

# 1. نقرأ الترانسكربت
with open("output/sample_meeting_transcript.json", "r", encoding="utf-8") as f:
    transcript_data = json.load(f)

full_text = " ".join(segment["text"] for segment in transcript_data["segments"])

print("النص الأصلي:")
print(full_text)
print()

# 2. نجهز طلب واحد يغطي كل شي (ملخص + بنود منظمة)
url = "http://localhost:11434/api/chat"

system_prompt = """You are an assistant that analyzes meeting or speech transcripts.
Always respond with ONLY valid JSON, with no extra text, no explanations, and no markdown code fences.
Use exactly this structure:

{
  "summary": "...",
  "decisions": ["..."],
  "action_items": [{"person": "...", "task": "...", "deadline": "..."}],
  "deadlines": ["..."],
  "open_questions": ["..."]
}

The "summary" field should be 2-3 concise sentences. If a list category has nothing to report, return an empty list for it. Do not invent information that isn't in the transcript."""

payload = {
    "model": "qwen2.5:7b",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this transcript:\n\n{full_text}"}
    ],
    "stream": False
}

# 3. نرسل الطلب (مرة وحدة بس)
response = requests.post(url, json=payload)
response.raise_for_status()

raw_content = response.json()["message"]["content"].strip()

# 4. كود دفاعي زي قبل
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

# 5. نحفظها
with open("output/sample_meeting_analysis.json", "w", encoding="utf-8") as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)

print("\nتم حفظ التحليل الكامل بملف: output/sample_meeting_analysis.json")